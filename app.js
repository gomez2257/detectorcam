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
const globalFilterInput = document.querySelector("#globalFilter");
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
let backgroundBrightness = null;
let animationId = null;
let recordingAnimationId = null;
let facingMode = "environment";
let isRecording = false;
let lastMotionAt = 0;
let lastMotion = { boxes: [], points: [], strength: 0, filtered: false };
let trailPoints = [];

const grid = {
  columns: 34,
  rows: 24,
  width: 408,
  height: 288,
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
    backgroundBrightness = null;
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

  if (!backgroundBrightness) {
    backgroundBrightness = buildBrightnessFrame(frame);
  }

  const motion = previousFrame
    ? getMotionData(previousFrame, frame, backgroundBrightness)
    : { boxes: [], points: [], strength: 0, filtered: false };

  updateBackground(frame, motion.points.length > 0 ? 0.008 : 0.025);
  previousFrame = frame;
  lastMotion = motion;

  updateTrail(motion.points);
  drawMotion(motion);
  updateMotionState(motion.points.length > 0, motion.strength, motion.filtered);
  animationId = requestAnimationFrame(detectMotion);
}

function buildBrightnessFrame(frame) {
  const values = new Float32Array(grid.width * grid.height);

  for (let i = 0, pixel = 0; i < frame.data.length; i += 4, pixel += 1) {
    values[pixel] = brightness(frame.data, i);
  }

  return values;
}

function updateBackground(frame, rate) {
  if (!backgroundBrightness) return;

  for (let i = 0, pixel = 0; i < frame.data.length; i += 4, pixel += 1) {
    const current = brightness(frame.data, i);
    backgroundBrightness[pixel] = backgroundBrightness[pixel] * (1 - rate) + current * rate;
  }
}

function getMotionData(previous, current, background) {
  const cellWidth = Math.floor(grid.width / grid.columns);
  const cellHeight = Math.floor(grid.height / grid.rows);
  const detail = Number(detailInput.value);
  const sensitivity = Number(sensitivityInput.value);
  const candidates = [];
  let totalDelta = 0;

  for (let row = 0; row < grid.rows; row += 1) {
    for (let column = 0; column < grid.columns; column += 1) {
      const cell = analyzeCell(previous, current, background, column, row, cellWidth, cellHeight, sensitivity);
      totalDelta += cell.averageDelta;

      if (cell.changedPixels >= detail) {
        candidates.push(cell);
      }
    }
  }

  const filtered = filterMotionCandidates(candidates, sensitivity, detail);
  const boxes = filtered.map((cell) => ({
    x: cell.x,
    y: cell.y,
    width: cell.width,
    height: cell.height,
    strength: cell.strength,
  }));
  const points = filtered.map((cell) => ({
    x: cell.x + cell.width / 2,
    y: cell.y + cell.height / 2,
    strength: cell.strength,
    age: 1,
  }));

  const maxCells = grid.columns * grid.rows;
  const strength = Math.min(100, Math.round((points.length / maxCells) * 260 + (totalDelta / maxCells) * 0.5));
  return { boxes, points, strength, filtered: filtered.length < candidates.length };
}

function analyzeCell(previous, current, background, column, row, cellWidth, cellHeight, sensitivity) {
  const startX = column * cellWidth;
  const startY = row * cellHeight;
  let changedPixels = 0;
  let cellDelta = 0;
  let samples = 0;

  for (let y = startY; y < startY + cellHeight; y += 3) {
    for (let x = startX; x < startX + cellWidth; x += 3) {
      const pixel = y * grid.width + x;
      const index = pixel * 4;
      const previousBrightness = brightness(previous.data, index);
      const currentBrightness = brightness(current.data, index);
      const frameDelta = Math.abs(currentBrightness - previousBrightness);
      const backgroundDelta = Math.abs(currentBrightness - background[pixel]);
      const effectiveDelta = Math.max(frameDelta, backgroundDelta * 0.72);
      samples += 1;
      cellDelta += effectiveDelta;

      if (effectiveDelta > sensitivity) {
        changedPixels += 1;
      }
    }
  }

  const averageDelta = samples ? cellDelta / samples : 0;
  const strength = clamp((averageDelta - sensitivity) / Math.max(1, 82 - sensitivity), 0.16, 1);

  return {
    x: startX,
    y: startY,
    width: cellWidth,
    height: cellHeight,
    changedPixels,
    averageDelta,
    strength,
  };
}

function filterMotionCandidates(candidates, sensitivity, detail) {
  if (!candidates.length) return [];

  const sorted = [...candidates].sort((a, b) => b.averageDelta - a.averageDelta);
  const maxCells = grid.columns * grid.rows;
  const globalRatio = candidates.length / maxCells;
  const useGlobalFilter = globalFilterInput.checked;

  if (!useGlobalFilter) {
    return sorted.slice(0, 120);
  }

  const strongThreshold = Math.max(sensitivity + 8, percentile(sorted.map((cell) => cell.averageDelta), 0.72));
  let filtered = sorted.filter((cell) => cell.averageDelta >= strongThreshold && cell.changedPixels >= detail);

  if (globalRatio > 0.18) {
    const stricterThreshold = Math.max(sensitivity + 14, percentile(sorted.map((cell) => cell.averageDelta), 0.86));
    filtered = sorted.filter((cell) => cell.averageDelta >= stricterThreshold && cell.changedPixels >= detail + 2);
  }

  if (globalRatio > 0.35) {
    filtered = filtered.slice(0, 36);
  } else {
    filtered = filtered.slice(0, 80);
  }

  return removeCrowdedNeighbors(filtered);
}

function removeCrowdedNeighbors(cells) {
  const kept = [];
  const minDistance = 9;

  cells.forEach((cell) => {
    const centerX = cell.x + cell.width / 2;
    const centerY = cell.y + cell.height / 2;
    const tooClose = kept.some((saved) => {
      const savedX = saved.x + saved.width / 2;
      const savedY = saved.y + saved.height / 2;
      return Math.hypot(centerX - savedX, centerY - savedY) < minDistance;
    });

    if (!tooClose) kept.push(cell);
  });

  return kept;
}

function percentile(values, ratio) {
  if (!values.length) return 0;
  const ordered = [...values].sort((a, b) => a - b);
  const index = Math.min(ordered.length - 1, Math.max(0, Math.floor(ordered.length * ratio)));
  return ordered[index];
}

function brightness(data, index) {
  return data[index] * 0.299 + data[index + 1] * 0.587 + data[index + 2] * 0.114;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function updateTrail(points) {
  trailPoints = trailPoints
    .map((point) => ({ ...point, age: point.age - 0.055 }))
    .filter((point) => point.age > 0);

  if (points.length) {
    trailPoints.push(...points.map((point) => ({ ...point, age: 1 })));
  }

  if (trailPoints.length > 260) {
    trailPoints = trailPoints.slice(trailPoints.length - 260);
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
    const alpha = heatOnly ? 0.1 + box.strength * 0.45 : 0.04 + box.strength * 0.12;
    context.fillStyle = `rgba(255, ${Math.round(210 - box.strength * 120)}, 30, ${alpha})`;
    context.fillRect(x, y, boxWidth, boxHeight);

    if (!heatOnly) {
      context.lineWidth = 1.2;
      context.strokeStyle = "rgba(44, 255, 154, 0.42)";
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
    context.arc(x, y, radius + 5, 0, Math.PI * 2);
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

function updateMotionState(hasMotion, strength, filtered) {
  const now = Date.now();
  const filterText = filtered ? " | filtro activo" : "";
  motionStats.textContent = `Puntos: ${lastMotion.points.length} | Intensidad: ${strength}%${filterText}`;

  if (hasMotion) {
    lastMotionAt = now;
    motionBanner.classList.add("visible");
    setStatus(isRecording ? "Grabando" : "Cambio", isRecording ? "recording" : "motion");
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
    recordingContext.filter = enhanceViewInput.checked ? "contrast(1.45) brightness(1.12) saturate(0.75)" : "none";
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