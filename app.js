const video = document.querySelector("#camera");
const overlay = document.querySelector("#overlay");
const statusLabel = document.querySelector("#status");
const motionBanner = document.querySelector("#motionBanner");
const motionStats = document.querySelector("#motionStats");
const startCameraButton = document.querySelector("#startCamera");
const recordButton = document.querySelector("#record");
const switchCameraButton = document.querySelector("#switchCamera");
const cameraSelect = document.querySelector("#cameraSelect");
const visualModeSelect = document.querySelector("#visualMode");
const sensitivityInput = document.querySelector("#sensitivity");
const detailInput = document.querySelector("#detail");
const motionOnlyInput = document.querySelector("#motionOnly");
const enhanceViewInput = document.querySelector("#enhanceView");
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
let lastMotion = { boxes: [], points: [], strength: 0 };
let trailPoints = [];

const grid = {
  columns: 30,
  rows: 22,
  width: 360,
  height: 264,
};

analysisCanvas.width = grid.width;
analysisCanvas.height = grid.height;

async function startCamera() {
  stopCamera();

  try {
    const selectedDeviceId = cameraSelect.value;
    const videoConstraints = selectedDeviceId
      ? { deviceId: { exact: selectedDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
      : { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } };

    stream = await navigator.mediaDevices.getUserMedia({
      video: videoConstraints,
      audio: true,
    });

    video.srcObject = stream;
    await video.play();
    await refreshCameraList();
    resizeOverlay();
    previousFrame = null;
    trailPoints = [];
    startCameraButton.textContent = "Reiniciar";
    recordButton.disabled = false;
    switchCameraButton.disabled = false;
    cameraSelect.disabled = false;
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

async function refreshCameraList() {
  if (!navigator.mediaDevices?.enumerateDevices) return;

  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter((device) => device.kind === "videoinput");
  const activeTrack = stream?.getVideoTracks()[0];
  const activeDeviceId = activeTrack?.getSettings().deviceId || cameraSelect.value;

  cameraSelect.innerHTML = '<option value="">Camara automatica</option>';
  cameras.forEach((camera, index) => {
    const option = document.createElement("option");
    option.value = camera.deviceId;
    option.textContent = camera.label || `Camara ${index + 1}`;
    option.selected = camera.deviceId === activeDeviceId;
    cameraSelect.append(option);
  });
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
  const motion = previousFrame ? getMotionData(previousFrame, frame) : { boxes: [], points: [], strength: 0 };
  previousFrame = frame;
  lastMotion = motion;

  updateTrail(motion.points);
  drawMotion(motion);
  updateMotionState(motion.points.length > 0, motion.strength);
  animationId = requestAnimationFrame(detectMotion);
}

function getMotionData(previous, current) {
  const cellWidth = Math.floor(grid.width / grid.columns);
  const cellHeight = Math.floor(grid.height / grid.rows);
  const boxes = [];
  const points = [];
  const sensitivity = Number(sensitivityInput.value);
  const detail = Number(detailInput.value);
  let totalDelta = 0;

  for (let row = 0; row < grid.rows; row += 1) {
    for (let column = 0; column < grid.columns; column += 1) {
      let changedPixels = 0;
      let cellDelta = 0;
      let samples = 0;
      const startX = column * cellWidth;
      const startY = row * cellHeight;

      for (let y = startY; y < startY + cellHeight; y += 3) {
        for (let x = startX; x < startX + cellWidth; x += 3) {
          const index = (y * grid.width + x) * 4;
          const previousBrightness = brightness(previous.data, index);
          const currentBrightness = brightness(current.data, index);
          const delta = Math.abs(currentBrightness - previousBrightness);
          samples += 1;
          cellDelta += delta;

          if (delta > sensitivity) {
            changedPixels += 1;
          }
        }
      }

      const averageDelta = samples ? cellDelta / samples : 0;
      totalDelta += averageDelta;

      if (changedPixels >= detail) {
        const strength = clamp((averageDelta - sensitivity) / Math.max(1, 80 - sensitivity), 0.18, 1);
        const centerX = startX + cellWidth / 2;
        const centerY = startY + cellHeight / 2;
        boxes.push({ x: startX, y: startY, width: cellWidth, height: cellHeight, strength });
        points.push({ x: centerX, y: centerY, strength, age: 1 });
      }
    }
  }

  const maxCells = grid.columns * grid.rows;
  const strength = Math.min(100, Math.round((points.length / maxCells) * 240 + (totalDelta / maxCells) * 0.45));
  return { boxes, points, strength };
}

function brightness(data, index) {
  return data[index] * 0.299 + data[index + 1] * 0.587 + data[index + 2] * 0.114;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function updateTrail(points) {
  trailPoints = trailPoints
    .map((point) => ({ ...point, age: point.age - 0.06 }))
    .filter((point) => point.age > 0);

  if (points.length) {
    trailPoints.push(...points.map((point) => ({ ...point, age: 1 })));
  }

  if (trailPoints.length > 320) {
    trailPoints = trailPoints.slice(trailPoints.length - 320);
  }
}

function drawMotion(motion) {
  resizeOverlay();
  const rect = overlay.getBoundingClientRect();
  overlayContext.clearRect(0, 0, rect.width, rect.height);

  if (!motionOnlyInput.checked) return;

  drawMotionOverlay(overlayContext, motion, rect.width, rect.height, visualModeSelect.value);
}

function drawMotionOverlay(context, motion, width, height, mode) {
  if (mode === "all" || mode === "heat") {
    drawHeatCells(context, motion.boxes, width, height, mode === "heat");
  }

  if (mode === "all" || mode === "points") {
    drawPoints(context, motion.points, width, height, 1);
  }

  if (mode === "trail") {
    drawPoints(context, trailPoints, width, height, 0.9);
  }
}

function drawHeatCells(context, boxes, width, height, heatOnly) {
  const scaleX = width / grid.width;
  const scaleY = height / grid.height;

  boxes.forEach((box) => {
    const x = box.x * scaleX;
    const y = box.y * scaleY;
    const boxWidth = box.width * scaleX;
    const boxHeight = box.height * scaleY;
    const alpha = heatOnly ? 0.1 + box.strength * 0.45 : 0.06 + box.strength * 0.16;
    context.fillStyle = `rgba(255, ${Math.round(210 - box.strength * 120)}, 30, ${alpha})`;
    context.fillRect(x, y, boxWidth, boxHeight);

    if (!heatOnly) {
      context.lineWidth = 1.6;
      context.strokeStyle = "rgba(44, 255, 154, 0.78)";
      context.strokeRect(x, y, boxWidth, boxHeight);
    }
  });
}

function drawPoints(context, points, width, height, opacityMultiplier) {
  const scaleX = width / grid.width;
  const scaleY = height / grid.height;

  points.forEach((point) => {
    const x = point.x * scaleX;
    const y = point.y * scaleY;
    const radius = 3 + point.strength * 8;
    const alpha = clamp(point.age * opacityMultiplier, 0.12, 1);

    context.beginPath();
    context.arc(x, y, radius + 4, 0, Math.PI * 2);
    context.fillStyle = `rgba(255, 255, 255, ${alpha * 0.22})`;
    context.fill();

    context.beginPath();
    context.arc(x, y, radius, 0, Math.PI * 2);
    context.fillStyle = point.strength > 0.55 ? `rgba(255, 56, 56, ${alpha})` : `rgba(255, 210, 40, ${alpha})`;
    context.fill();

    context.lineWidth = 1.5;
    context.strokeStyle = `rgba(255, 255, 255, ${alpha * 0.8})`;
    context.stroke();
  });
}

function updateMotionState(hasMotion, strength) {
  const now = Date.now();
  motionStats.textContent = `Puntos: ${lastMotion.points.length} | Intensidad: ${strength}%`;

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
    if (enhanceViewInput.checked) {
      recordingContext.filter = "contrast(1.45) brightness(1.12) saturate(0.75)";
    } else {
      recordingContext.filter = "none";
    }

    recordingContext.drawImage(video, 0, 0, width, height);
    recordingContext.filter = "none";

    if (motionOnlyInput.checked) {
      drawMotionOverlay(recordingContext, lastMotion, width, height, visualModeSelect.value);
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
  cameraSelect.value = "";
  facingMode = facingMode === "environment" ? "user" : "environment";
  await startCamera();
}

startCameraButton.addEventListener("click", startCamera);
recordButton.addEventListener("click", toggleRecording);
switchCameraButton.addEventListener("click", switchCamera);
cameraSelect.addEventListener("change", startCamera);
enhanceViewInput.addEventListener("change", () => {
  video.classList.toggle("enhanced", enhanceViewInput.checked);
});
clearListButton.addEventListener("click", () => {
  recordingList.innerHTML = '<p class="empty">Cuando termines una grabacion aparecera aqui.</p>';
});
window.addEventListener("resize", resizeOverlay);

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("./service-worker.js").catch(console.error);
}