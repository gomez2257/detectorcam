import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs";

const video = document.querySelector("#camera");
const overlay = document.querySelector("#overlay");
const statusLabel = document.querySelector("#status");
const motionBanner = document.querySelector("#motionBanner");
const motionStats = document.querySelector("#motionStats");
const startCameraButton = document.querySelector("#startCamera");
const recordButton = document.querySelector("#record");
const switchCameraButton = document.querySelector("#switchCamera");
const cameraSelect = document.querySelector("#cameraSelect");
const aiModeSelect = document.querySelector("#aiMode");
const visualModeSelect = document.querySelector("#visualMode");
const sensitivityInput = document.querySelector("#sensitivity");
const detailInput = document.querySelector("#detail");
const drawOverlayInput = document.querySelector("#drawOverlay");
const globalFilterInput = document.querySelector("#globalFilter");
const enhanceViewInput = document.querySelector("#enhanceView");
const clearListButton = document.querySelector("#clearList");
const recordingList = document.querySelector("#recordingList");

const overlayContext = overlay.getContext("2d");
const analysisCanvas = document.createElement("canvas");
const analysisContext = analysisCanvas.getContext("2d", { willReadFrequently: true });
const recordingCanvas = document.createElement("canvas");
const recordingContext = recordingCanvas.getContext("2d");

const MEDIAPIPE_VERSION = "0.10.35";
const WASM_ROOT = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
const MODEL_URLS = {
  pose: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
  hands: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
  face: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
};

const POSE_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10], [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21],
  [17, 19], [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28],
  [27, 29], [28, 30], [29, 31], [30, 32], [27, 31], [28, 32],
];

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
  [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

const FACE_CONNECTIONS = [
  [33, 7], [7, 163], [163, 144], [144, 145], [145, 153], [153, 154], [154, 155], [155, 133],
  [362, 382], [382, 381], [381, 380], [380, 374], [374, 373], [373, 390], [390, 249], [249, 263],
  [61, 146], [146, 91], [91, 181], [181, 84], [84, 17], [17, 314], [314, 405], [405, 321], [321, 375], [375, 291],
  [10, 338], [338, 297], [297, 332], [332, 284], [284, 251], [251, 389], [389, 356], [356, 454], [454, 323], [323, 361], [361, 288],
  [10, 109], [109, 67], [67, 103], [103, 54], [54, 21], [21, 162], [162, 127], [127, 234], [234, 93], [93, 132], [132, 58],
];

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
let lastInferenceAt = 0;
let visionReady = false;
let poseLandmarker = null;
let handLandmarker = null;
let faceLandmarker = null;
let aiState = { pose: "cargando", hands: "cargando", face: "cargando" };
let lastAi = { poses: [], hands: [], faces: [] };
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

initAi();

async function initAi() {
  setStatus("Cargando IA", "idle");
  motionStats.textContent = "Cargando modelos de cuerpo, manos y cara...";

  try {
    const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
    const results = await Promise.allSettled([
      createPoseLandmarker(vision),
      createHandLandmarker(vision),
      createFaceLandmarker(vision),
    ]);

    poseLandmarker = results[0].status === "fulfilled" ? results[0].value : null;
    handLandmarker = results[1].status === "fulfilled" ? results[1].value : null;
    faceLandmarker = results[2].status === "fulfilled" ? results[2].value : null;

    aiState = {
      pose: poseLandmarker ? "lista" : "error",
      hands: handLandmarker ? "lista" : "error",
      face: faceLandmarker ? "lista" : "error",
    };

    visionReady = Boolean(poseLandmarker || handLandmarker || faceLandmarker);
    setStatus(visionReady ? "IA lista" : "IA no cargo", "idle");
    motionStats.textContent = getAiLoadText();
    results.forEach((result) => {
      if (result.status === "rejected") console.error(result.reason);
    });
  } catch (error) {
    console.error(error);
    aiState = { pose: "error", hands: "error", face: "error" };
    visionReady = false;
    setStatus("IA no cargo", "idle");
    motionStats.textContent = "IA no cargo. Revisa internet y recarga.";
  }
}

function createPoseLandmarker(vision) {
  return PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URLS.pose, delegate: "CPU" },
    runningMode: "VIDEO",
    numPoses: 2,
    minPoseDetectionConfidence: 0.3,
    minPosePresenceConfidence: 0.3,
    minTrackingConfidence: 0.3,
  });
}

function createHandLandmarker(vision) {
  return HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URLS.hands, delegate: "CPU" },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.3,
    minHandPresenceConfidence: 0.3,
    minTrackingConfidence: 0.3,
  });
}

function createFaceLandmarker(vision) {
  return FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URLS.face, delegate: "CPU" },
    runningMode: "VIDEO",
    numFaces: 1,
    minFaceDetectionConfidence: 0.35,
    minFacePresenceConfidence: 0.35,
    minTrackingConfidence: 0.35,
  });
}

function getAiLoadText() {
  return `IA cuerpo: ${aiState.pose} | manos: ${aiState.hands} | cara: ${aiState.face}`;
}

async function startCamera() {
  stopCamera();

  try {
    const selectedDeviceId = cameraSelect.value;
    const videoConstraints = selectedDeviceId
      ? { deviceId: { exact: selectedDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
      : { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } };

    stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: true });
    video.srcObject = stream;
    await video.play();
    await refreshCameraList();
    resizeOverlay();
    previousFrame = null;
    backgroundBrightness = null;
    trailPoints = [];
    lastAi = { poses: [], hands: [], faces: [] };
    startCameraButton.textContent = "Reiniciar";
    recordButton.disabled = false;
    switchCameraButton.disabled = false;
    cameraSelect.disabled = false;
    setStatus(visionReady ? "Camara activa" : "Sin IA", "idle");
    detectLoop();
  } catch (error) {
    console.error(error);
    setStatus("Sin permiso", "idle");
    alert("No se pudo abrir la camara. Revisa permisos del navegador.");
  }
}

function stopCamera() {
  if (animationId) cancelAnimationFrame(animationId);
  if (recordingAnimationId) cancelAnimationFrame(recordingAnimationId);
  animationId = null;
  recordingAnimationId = null;

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

function detectLoop() {
  if (!stream || video.readyState < 2) {
    animationId = requestAnimationFrame(detectLoop);
    return;
  }

  updateMotionDetection();

  const now = performance.now();
  if (visionReady && now - lastInferenceAt > 120) {
    runAiDetection(now);
    lastInferenceAt = now;
  }

  drawScene();
  updateStatus();
  animationId = requestAnimationFrame(detectLoop);
}

function runAiDetection(now) {
  const mode = aiModeSelect.value;
  const poses = [];
  const hands = [];
  const faces = [];

  try {
    if (poseLandmarker && (mode === "holistic" || mode === "poseHands" || mode === "pose")) {
      const poseResult = poseLandmarker.detectForVideo(video, now);
      poses.push(...(poseResult.landmarks || []));
    }

    if (handLandmarker && (mode === "holistic" || mode === "poseHands")) {
      const handResult = handLandmarker.detectForVideo(video, now);
      hands.push(...(handResult.landmarks || []));
    }

    if (faceLandmarker && mode === "holistic") {
      const faceResult = faceLandmarker.detectForVideo(video, now);
      faces.push(...(faceResult.faceLandmarks || []));
    }

    lastAi = { poses, hands, faces };
  } catch (error) {
    console.warn(error);
  }
}

function updateMotionDetection() {
  analysisContext.drawImage(video, 0, 0, grid.width, grid.height);
  const frame = analysisContext.getImageData(0, 0, grid.width, grid.height);

  if (!backgroundBrightness) backgroundBrightness = buildBrightnessFrame(frame);

  lastMotion = previousFrame
    ? getMotionData(previousFrame, frame, backgroundBrightness)
    : { boxes: [], points: [], strength: 0, filtered: false };

  updateBackground(frame, lastMotion.points.length > 0 ? 0.008 : 0.025);
  previousFrame = frame;
  updateTrail(lastMotion.points);
}

function drawScene() {
  resizeOverlay();
  const rect = overlay.getBoundingClientRect();
  overlayContext.clearRect(0, 0, rect.width, rect.height);

  if (!drawOverlayInput.checked) return;

  const mode = aiModeSelect.value;
  const visual = visualModeSelect.value;

  if (mode === "motion" || visual === "trail" || visual === "heat") {
    drawMotionOverlay(overlayContext, lastMotion, rect.width, rect.height, visual);
  }

  if (mode !== "motion" && visual !== "trail" && visual !== "heat") {
    drawAiOverlay(overlayContext, lastAi, rect.width, rect.height, visual);
  }
}

function drawAiOverlay(context, ai, width, height, visual) {
  const transform = getVideoTransform(width, height);

  ai.poses.forEach((pose) => {
    drawConnections(context, pose, POSE_CONNECTIONS, transform, "#2cff9a", 4, 0.25);
    drawLandmarkPoints(context, pose, transform, "#ffdf3d", "#ffffff", 5, 0.25);
  });

  ai.hands.forEach((hand) => {
    drawConnections(context, hand, HAND_CONNECTIONS, transform, "#ff4fd8", 3, 0.2);
    drawLandmarkPoints(context, hand, transform, "#ff4b4b", "#ffffff", 4, 0.2);
  });

  ai.faces.forEach((face) => {
    drawConnections(context, face, FACE_CONNECTIONS, transform, "#67d7ff", 2, 0.15);
    drawLandmarkPoints(context, face, transform, "#67d7ff", "#ffffff", visual === "points" ? 2.2 : 1.4, 0.15, visual === "points" ? 2 : 8);
  });
}

function drawConnections(context, landmarks, connections, transform, color, lineWidth, minVisibility) {
  context.save();
  context.lineWidth = lineWidth;
  context.strokeStyle = color;
  context.shadowColor = "rgba(0, 0, 0, 0.72)";
  context.shadowBlur = 8;

  connections.forEach(([startIndex, endIndex]) => {
    const start = landmarks[startIndex];
    const end = landmarks[endIndex];
    if (!isVisible(start, minVisibility) || !isVisible(end, minVisibility)) return;
    const a = toCanvasPoint(start, transform);
    const b = toCanvasPoint(end, transform);
    context.beginPath();
    context.moveTo(a.x, a.y);
    context.lineTo(b.x, b.y);
    context.stroke();
  });

  context.restore();
}

function drawLandmarkPoints(context, landmarks, transform, fill, stroke, radius, minVisibility, skip = 1) {
  context.save();
  context.lineWidth = 1.4;

  landmarks.forEach((landmark, index) => {
    if (index % skip !== 0 || !isVisible(landmark, minVisibility)) return;
    const point = toCanvasPoint(landmark, transform);
    context.beginPath();
    context.arc(point.x, point.y, radius, 0, Math.PI * 2);
    context.fillStyle = fill;
    context.fill();
    context.strokeStyle = stroke;
    context.stroke();
  });

  context.restore();
}

function isVisible(landmark, minVisibility) {
  return landmark && (landmark.visibility === undefined || landmark.visibility >= minVisibility);
}

function getVideoTransform(width, height) {
  const videoWidth = video.videoWidth || width;
  const videoHeight = video.videoHeight || height;
  const scale = Math.max(width / videoWidth, height / videoHeight);
  const drawWidth = videoWidth * scale;
  const drawHeight = videoHeight * scale;
  return {
    x: (width - drawWidth) / 2,
    y: (height - drawHeight) / 2,
    width: drawWidth,
    height: drawHeight,
  };
}

function toCanvasPoint(landmark, transform) {
  return {
    x: transform.x + landmark.x * transform.width,
    y: transform.y + landmark.y * transform.height,
  };
}

function updateStatus() {
  const humanCount = lastAi.poses.length;
  const handCount = lastAi.hands.length;
  const faceCount = lastAi.faces.length;
  const motionCount = lastMotion.points.length;
  const hasAi = humanCount + handCount + faceCount > 0;
  const hasMotion = motionCount > 0;
  const now = Date.now();

  if (visionReady) {
    motionStats.textContent = `Cuerpos: ${humanCount} | Manos: ${handCount} | Caras: ${faceCount} | Puntos: ${motionCount}`;
  } else {
    motionStats.textContent = `${getAiLoadText()} | Puntos: ${motionCount}`;
  }

  if (hasAi || hasMotion) {
    lastMotionAt = now;
    motionBanner.textContent = hasAi ? "Esqueleto detectado" : "Cambio detectado";
    motionBanner.classList.add("visible");
    setStatus(isRecording ? "Grabando" : hasAi ? "IA activa" : "Cambio", isRecording ? "recording" : "motion");
    return;
  }

  if (now - lastMotionAt > 700) {
    motionBanner.classList.remove("visible");
    setStatus(isRecording ? "Grabando" : "Camara activa", isRecording ? "recording" : "idle");
  }
}

function buildBrightnessFrame(frame) {
  const values = new Float32Array(grid.width * grid.height);
  for (let i = 0, pixel = 0; i < frame.data.length; i += 4, pixel += 1) values[pixel] = brightness(frame.data, i);
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
      if (cell.changedPixels >= detail) candidates.push(cell);
    }
  }

  const filtered = filterMotionCandidates(candidates, sensitivity, detail);
  const boxes = filtered.map((cell) => ({ x: cell.x, y: cell.y, width: cell.width, height: cell.height, strength: cell.strength }));
  const points = filtered.map((cell) => ({ x: cell.x + cell.width / 2, y: cell.y + cell.height / 2, strength: cell.strength, age: 1 }));
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
      const frameDelta = Math.abs(brightness(current.data, index) - brightness(previous.data, index));
      const backgroundDelta = Math.abs(brightness(current.data, index) - background[pixel]);
      const effectiveDelta = Math.max(frameDelta, backgroundDelta * 0.72);
      samples += 1;
      cellDelta += effectiveDelta;
      if (effectiveDelta > sensitivity) changedPixels += 1;
    }
  }

  const averageDelta = samples ? cellDelta / samples : 0;
  const strength = clamp((averageDelta - sensitivity) / Math.max(1, 82 - sensitivity), 0.16, 1);
  return { x: startX, y: startY, width: cellWidth, height: cellHeight, changedPixels, averageDelta, strength };
}

function filterMotionCandidates(candidates, sensitivity, detail) {
  if (!candidates.length) return [];
  const sorted = [...candidates].sort((a, b) => b.averageDelta - a.averageDelta);
  const maxCells = grid.columns * grid.rows;
  const globalRatio = candidates.length / maxCells;

  if (!globalFilterInput.checked) return sorted.slice(0, 120);

  let threshold = Math.max(sensitivity + 8, percentile(sorted.map((cell) => cell.averageDelta), 0.72));
  let filtered = sorted.filter((cell) => cell.averageDelta >= threshold && cell.changedPixels >= detail);

  if (globalRatio > 0.18) {
    threshold = Math.max(sensitivity + 14, percentile(sorted.map((cell) => cell.averageDelta), 0.86));
    filtered = sorted.filter((cell) => cell.averageDelta >= threshold && cell.changedPixels >= detail + 2);
  }

  return removeCrowdedNeighbors(filtered.slice(0, globalRatio > 0.35 ? 36 : 80));
}

function removeCrowdedNeighbors(cells) {
  const kept = [];
  const minDistance = 9;
  cells.forEach((cell) => {
    const centerX = cell.x + cell.width / 2;
    const centerY = cell.y + cell.height / 2;
    const tooClose = kept.some((saved) => Math.hypot(centerX - (saved.x + saved.width / 2), centerY - (saved.y + saved.height / 2)) < minDistance);
    if (!tooClose) kept.push(cell);
  });
  return kept;
}

function drawMotionOverlay(context, motion, width, height, visual) {
  if (visual === "heat") drawHeatCells(context, motion.boxes, width, height);
  if (visual === "trail") drawMotionPoints(context, trailPoints, width, height, 0.9);
  if (visual === "points" || visual === "skeleton") drawMotionPoints(context, motion.points, width, height, 1);
}

function drawHeatCells(context, boxes, width, height) {
  const scaleX = width / grid.width;
  const scaleY = height / grid.height;
  boxes.forEach((box) => {
    context.fillStyle = `rgba(255, ${Math.round(210 - box.strength * 120)}, 30, ${0.1 + box.strength * 0.45})`;
    context.fillRect(box.x * scaleX, box.y * scaleY, box.width * scaleX, box.height * scaleY);
  });
}

function drawMotionPoints(context, points, width, height, opacityMultiplier) {
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

function updateTrail(points) {
  trailPoints = trailPoints.map((point) => ({ ...point, age: point.age - 0.055 })).filter((point) => point.age > 0);
  if (points.length) trailPoints.push(...points.map((point) => ({ ...point, age: 1 })));
  if (trailPoints.length > 260) trailPoints = trailPoints.slice(trailPoints.length - 260);
}

function brightness(data, index) {
  return data[index] * 0.299 + data[index + 1] * 0.587 + data[index + 2] * 0.114;
}

function percentile(values, ratio) {
  if (!values.length) return 0;
  const ordered = [...values].sort((a, b) => a - b);
  return ordered[Math.min(ordered.length - 1, Math.max(0, Math.floor(ordered.length * ratio)))];
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function resizeOverlay() {
  const rect = video.getBoundingClientRect();
  const pixelRatio = window.devicePixelRatio || 1;
  overlay.width = Math.round(rect.width * pixelRatio);
  overlay.height = Math.round(rect.height * pixelRatio);
  overlayContext.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
}

function toggleRecording() {
  if (!stream) return;
  if (isRecording) {
    mediaRecorder.stop();
    return;
  }

  recordedChunks = [];
  recordingStream = createRecordingStream();
  const mimeType = getSupportedMimeType();
  mediaRecorder = new MediaRecorder(recordingStream, mimeType ? { mimeType } : undefined);
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
    if (drawOverlayInput.checked) {
      if (aiModeSelect.value === "motion" || visualModeSelect.value === "trail" || visualModeSelect.value === "heat") {
        drawMotionOverlay(recordingContext, lastMotion, width, height, visualModeSelect.value);
      } else {
        drawAiOverlay(recordingContext, lastAi, width, height, visualModeSelect.value);
      }
    }
    recordingAnimationId = requestAnimationFrame(draw);
  };

  draw();
  const canvasStream = recordingCanvas.captureStream(30);
  stream.getAudioTracks().forEach((track) => canvasStream.addTrack(track));
  return canvasStream;
}

function getSupportedMimeType() {
  const types = [
    "video/mp4;codecs=h264,aac",
    "video/mp4",
    "video/webm;codecs=vp9,opus",
    "video/webm;codecs=vp8,opus",
    "video/webm",
  ];
  return types.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

async function saveRecording() {
  if (recordingAnimationId) cancelAnimationFrame(recordingAnimationId);
  recordingAnimationId = null;
  if (recordingStream) recordingStream.getVideoTracks().forEach((track) => track.stop());
  recordingStream = null;
  isRecording = false;
  recordButton.textContent = "Grabar";
  recordButton.classList.remove("primary");
  setStatus("Camara activa", "idle");

  const mimeType = recordedChunks[0]?.type || getSupportedMimeType() || "video/webm";
  const extension = mimeType.includes("mp4") ? "mp4" : "webm";
  const blob = new Blob(recordedChunks, { type: mimeType });
  const fileName = `detectorcam-${new Date().toISOString().replace(/[:.]/g, "-")}.${extension}`;
  const url = URL.createObjectURL(blob);
  const file = new File([blob], fileName, { type: mimeType });
  addRecordingItem({ url, file, fileName, mimeType, blob });
}

function addRecordingItem(recording) {
  const item = document.createElement("article");
  item.className = "recording-item";
  item.innerHTML = `
    <strong>${new Date().toLocaleString()}</strong>
    <video src="${recording.url}" controls playsinline></video>
    <div class="recording-actions">
      <button type="button" class="primary save-video">Guardar en el movil</button>
      <button type="button" class="ghost share-video">Compartir</button>
      <button type="button" class="danger delete-video">Eliminar</button>
    </div>
    <a class="download-link" href="${recording.url}" download="${recording.fileName}">Descarga alternativa</a>
  `;

  const saveButton = item.querySelector(".save-video");
  const shareButton = item.querySelector(".share-video");
  const deleteButton = item.querySelector(".delete-video");
  const downloadLink = item.querySelector(".download-link");

  saveButton.addEventListener("click", async () => {
    await saveVideoToDevice(recording, downloadLink);
  });

  shareButton.addEventListener("click", async () => {
    await shareVideo(recording, downloadLink);
  });

  deleteButton.addEventListener("click", () => {
    URL.revokeObjectURL(recording.url);
    item.remove();
    if (!recordingList.querySelector(".recording-item")) {
      recordingList.innerHTML = '<p class="empty">Cuando termines una grabacion aparecera aqui.</p>';
    }
  });

  if (!navigator.canShare?.({ files: [recording.file] })) {
    shareButton.disabled = true;
  }

  recordingList.querySelector(".empty")?.remove();
  recordingList.prepend(item);
}

async function saveVideoToDevice(recording, fallbackLink) {
  if (window.showSaveFilePicker) {
    try {
      const handle = await window.showSaveFilePicker({
        suggestedName: recording.fileName,
        types: [{ description: "Video", accept: { [recording.mimeType]: [`.${recording.fileName.split(".").pop()}`] } }],
      });
      const writable = await handle.createWritable();
      await writable.write(recording.blob);
      await writable.close();
      return;
    } catch (error) {
      if (error.name === "AbortError") return;
    }
  }

  fallbackLink.click();
}

async function shareVideo(recording, fallbackLink) {
  if (navigator.canShare?.({ files: [recording.file] })) {
    try {
      await navigator.share({ files: [recording.file], title: "DetectorCam", text: "Video DetectorCam" });
      return;
    } catch (error) {
      if (error.name === "AbortError") return;
    }
  }

  fallbackLink.click();
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
aiModeSelect.addEventListener("change", () => {
  lastAi = { poses: [], hands: [], faces: [] };
});
enhanceViewInput.addEventListener("change", () => video.classList.toggle("enhanced", enhanceViewInput.checked));
clearListButton.addEventListener("click", () => {
  recordingList.querySelectorAll("video").forEach((videoItem) => URL.revokeObjectURL(videoItem.src));
  recordingList.innerHTML = '<p class="empty">Cuando termines una grabacion aparecera aqui.</p>';
});
window.addEventListener("resize", resizeOverlay);

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("./service-worker.js").catch(console.error);
}