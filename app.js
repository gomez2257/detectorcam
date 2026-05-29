const video = document.querySelector("#camera");
const overlay = document.querySelector("#overlay");
const statusLabel = document.querySelector("#status");
const motionBanner = document.querySelector("#motionBanner");
const startCameraButton = document.querySelector("#startCamera");
const recordButton = document.querySelector("#record");
const switchCameraButton = document.querySelector("#switchCamera");
const sensitivityInput = document.querySelector("#sensitivity");
const motionOnlyInput = document.querySelector("#motionOnly");
const clearListButton = document.querySelector("#clearList");
const recordingList = document.querySelector("#recordingList");

const overlayContext = overlay.getContext("2d");
const analysisCanvas = document.createElement("canvas");
const analysisContext = analysisCanvas.getContext("2d", { willReadFrequently: true });
const recordingCanvas = document.createElement("canvas");
const recordingContext = recordingCanvas.getContext("2d");

let stream = null;
let recordingStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let previousFrame = null;
let animationId = null;
let recordingAnimationId = null;
let facingMode = "environment";
let isRecording = false;
let lastMotionAt = 0;
let lastMotionBoxes = [];

const grid = {
  columns: 16,
  rows: 12,
  width: 320,
  height: 240,
};

analysisCanvas.width = grid.width;
analysisCanvas.height = grid.height;

async function startCamera() {
  stopCamera();

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode,
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: true,
    });

    video.srcObject = stream;
    await video.play();
    resizeOverlay();
    previousFrame = null;
    startCameraButton.textContent = "Reiniciar";
    recordButton.disabled = false;
    switchCameraButton.disabled = false;
    setStatus("Camara activa", "idle");
    detectMotion();
  } catch (error) {
    console.error(error);
    setStatus("Sin permiso", "idle");
    alert("No se pudo abrir la camara. Revisa permisos del navegador.");
  }
}

function stopCamera() {
  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }

  if (recordingAnimationId) {
    cancelAnimationFrame(recordingAnimationId);
    recordingAnimationId = null;
  }

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
}

function resizeOverlay() {
  const rect = video.getBoundingClientRect();
  const pixelRatio = window.devicePixelRatio || 1;
  overlay.width = Math.round(rect.width * pixelRatio);
  overlay.height = Math.round(rect.height * pixelRatio);
  overlayContext.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
}

function detectMotion() {
  if (!stream || video.readyState < 2) {
    animationId = requestAnimationFrame(detectMotion);
    return;
  }

  analysisContext.drawImage(video, 0, 0, grid.width, grid.height);
  const frame = analysisContext.getImageData(0, 0, grid.width, grid.height);
  const boxes = previousFrame ? getMotionBoxes(previousFrame, frame) : [];
  previousFrame = frame;
  lastMotionBoxes = boxes;

  drawMotion(boxes);
  updateMotionState(boxes.length > 0);
  animationId = requestAnimationFrame(detectMotion);
}

function getMotionBoxes(previous, current) {
  const cellWidth = Math.floor(grid.width / grid.columns);
  const cellHeight = Math.floor(grid.height / grid.rows);
  const boxes = [];
  const sensitivity = Number(sensitivityInput.value);

  for (let row = 0; row < grid.rows; row += 1) {
    for (let column = 0; column < grid.columns; column += 1) {
      let changedPixels = 0;
      const startX = column * cellWidth;
      const startY = row * cellHeight;

      for (let y = startY; y < startY + cellHeight; y += 3) {
        for (let x = startX; x < startX + cellWidth; x += 3) {
          const index = (y * grid.width + x) * 4;
          const previousBrightness = brightness(previous.data, index);
          const currentBrightness = brightness(current.data, index);

          if (Math.abs(currentBrightness - previousBrightness) > sensitivity) {
            changedPixels += 1;
          }
        }
      }

      if (changedPixels > 12) {
        boxes.push({ x: startX, y: startY, width: cellWidth, height: cellHeight });
      }
    }
  }

  return mergeNearbyBoxes(boxes, cellWidth, cellHeight);
}

function brightness(data, index) {
  return data[index] * 0.299 + data[index + 1] * 0.587 + data[index + 2] * 0.114;
}

function mergeNearbyBoxes(boxes, cellWidth, cellHeight) {
  if (!boxes.length) return [];

  const minX = Math.min(...boxes.map((box) => box.x));
  const minY = Math.min(...boxes.map((box) => box.y));
  const maxX = Math.max(...boxes.map((box) => box.x + box.width));
  const maxY = Math.max(...boxes.map((box) => box.y + box.height));

  return [
    {
      x: Math.max(0, minX - cellWidth),
      y: Math.max(0, minY - cellHeight),
      width: Math.min(grid.width, maxX - minX + cellWidth * 2),
      height: Math.min(grid.height, maxY - minY + cellHeight * 2),
    },
  ];
}

function drawMotion(boxes) {
  resizeOverlay();
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);

  if (!motionOnlyInput.checked) return;

  const rect = overlay.getBoundingClientRect();
  drawBoxes(overlayContext, boxes, rect.width, rect.height);
}

function drawBoxes(context, boxes, width, height) {
  const scaleX = width / grid.width;
  const scaleY = height / grid.height;

  context.lineWidth = Math.max(3, Math.round(width / 180));
  context.strokeStyle = "#2cff9a";
  context.fillStyle = "rgba(44, 255, 154, 0.14)";
  context.font = `700 ${Math.max(18, Math.round(width / 42))}px Arial`;

  boxes.forEach((box) => {
    const x = box.x * scaleX;
    const y = box.y * scaleY;
    const boxWidth = box.width * scaleX;
    const boxHeight = box.height * scaleY;
    context.fillRect(x, y, boxWidth, boxHeight);
    context.strokeRect(x, y, boxWidth, boxHeight);
    context.fillStyle = "#2cff9a";
    context.fillText("Movimiento", x + 10, Math.max(28, y - 8));
    context.fillStyle = "rgba(44, 255, 154, 0.14)";
  });
}

function updateMotionState(hasMotion) {
  const now = Date.now();

  if (hasMotion) {
    lastMotionAt = now;
    motionBanner.classList.add("visible");
    setStatus(isRecording ? "Grabando" : "Movimiento", isRecording ? "recording" : "motion");
    return;
  }

  if (now - lastMotionAt > 700) {
    motionBanner.classList.remove("visible");
    setStatus(isRecording ? "Grabando" : "Camara activa", isRecording ? "recording" : "idle");
  }
}

function toggleRecording() {
  if (!stream) return;

  if (isRecording) {
    mediaRecorder.stop();
    return;
  }

  recordedChunks = [];
  recordingStream = createRecordingStream();
  mediaRecorder = new MediaRecorder(recordingStream, { mimeType: getSupportedMimeType() });
  mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) recordedChunks.push(event.data);
  };
  mediaRecorder.onstop = saveRecording;
  mediaRecorder.start();
  isRecording = true;
  recordButton.textContent = "Detener";
  recordButton.classList.add("primary");
  setStatus("Grabando", "recording");
}

function createRecordingStream() {
  const width = video.videoWidth || 1280;
  const height = video.videoHeight || 720;
  recordingCanvas.width = width;
  recordingCanvas.height = height;

  const draw = () => {
    recordingContext.drawImage(video, 0, 0, width, height);

    if (motionOnlyInput.checked) {
      drawBoxes(recordingContext, lastMotionBoxes, width, height);
    }

    recordingAnimationId = requestAnimationFrame(draw);
  };

  draw();

  const canvasStream = recordingCanvas.captureStream(30);
  stream.getAudioTracks().forEach((track) => canvasStream.addTrack(track));
  return canvasStream;
}

function getSupportedMimeType() {
  const types = ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"];
  return types.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

function saveRecording() {
  if (recordingAnimationId) {
    cancelAnimationFrame(recordingAnimationId);
    recordingAnimationId = null;
  }

  if (recordingStream) {
    recordingStream.getVideoTracks().forEach((track) => track.stop());
    recordingStream = null;
  }

  isRecording = false;
  recordButton.textContent = "Grabar";
  recordButton.classList.remove("primary");
  setStatus("Camara activa", "idle");

  const blob = new Blob(recordedChunks, { type: "video/webm" });
  const url = URL.createObjectURL(blob);
  const item = document.createElement("article");
  const timestamp = new Date().toLocaleString();

  item.className = "recording-item";
  item.innerHTML = `
    <strong>${timestamp}</strong>
    <video src="${url}" controls playsinline></video>
    <a href="${url}" download="detectorcam-${Date.now()}.webm">Descargar video</a>
  `;

  recordingList.querySelector(".empty")?.remove();
  recordingList.prepend(item);
}

function setStatus(text, mode) {
  statusLabel.textContent = text;
  statusLabel.className = `status ${mode}`;
}

async function switchCamera() {
  facingMode = facingMode === "environment" ? "user" : "environment";
  await startCamera();
}

startCameraButton.addEventListener("click", startCamera);
recordButton.addEventListener("click", toggleRecording);
switchCameraButton.addEventListener("click", switchCamera);
clearListButton.addEventListener("click", () => {
  recordingList.innerHTML = '<p class="empty">Cuando termines una grabacion aparecera aqui.</p>';
});
window.addEventListener("resize", resizeOverlay);

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("./service-worker.js").catch(console.error);
}
