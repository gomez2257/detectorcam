import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/vision_bundle.mjs";

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
const motionGateInput = document.querySelector("#motionGate");
const globalFilterInput = document.querySelector("#globalFilter");
const enhanceViewInput = document.querySelector("#enhanceView");
const clearListButton = document.querySelector("#clearList");
const recordingList = document.querySelector("#recordingList");
const diagAi = document.querySelector("#diagAi");
const diagCamera = document.querySelector("#diagCamera");
const diagRecorder = document.querySelector("#diagRecorder");
const diagStorage = document.querySelector("#diagStorage");

const overlayContext = overlay.getContext("2d");
const analysisCanvas = document.createElement("canvas");
const analysisContext = analysisCanvas.getContext("2d", { willReadFrequently: true });
const recordingCanvas = document.createElement("canvas");
const recordingContext = recordingCanvas.getContext("2d");

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
let recordingAudioStream = null;
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
let lastAi = { poses: [], hands: [], faces: [] };
let lastVisibleAi = { poses: [], hands: [], faces: [] };
let lastStaticAi = { poses: [], hands: [], faces: [] };
let previousAiForMotion = { poses: [], hands: [], faces: [] };
let lastMotion = { boxes: [], points: [], strength: 0, filtered: false };
let trailPoints = [];

const grid = {
  columns: 34,
  rows: 24,
  width: 408,
  height: 288,
};

const DB_NAME = "detectorcam-db";
const DB_VERSION = 1;
const RECORDING_STORE = "recordings";
const MAX_STORED_RECORDINGS = 8;
const MAX_STORED_BYTES = 180 * 1024 * 1024;

let dbPromise = null;
let storedRecordingCount = 0;
let storedRecordingBytes = 0;
const activeRecordingUrls = new Set();

analysisCanvas.width = grid.width;
analysisCanvas.height = grid.height;

initAi();
initRecordings();
updateDiagnostics();

async function initAi() {
  try {
    setStatus("Cargando IA", "idle");
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm"
    );

    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URLS.pose, delegate: "GPU" },
      runningMode: "VIDEO",
      numPoses: 2,
      minPoseDetectionConfidence: 0.35,
      minPosePresenceConfidence: 0.35,
      minTrackingConfidence: 0.35,
    });

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URLS.hands, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.35,
      minHandPresenceConfidence: 0.35,
      minTrackingConfidence: 0.35,
    });

    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URLS.face, delegate: "GPU" },
      runningMode: "VIDEO",
      numFaces: 1,
      minFaceDetectionConfidence: 0.4,
      minFacePresenceConfidence: 0.4,
      minTrackingConfidence: 0.4,
    });

    visionReady = true;
    setStatus("IA lista", "idle");
    motionStats.textContent = "IA lista | abre la cámara";
  } catch (error) {
    console.error(error);
    setStatus("IA no cargó", "idle");
    motionStats.textContent = "IA no cargó. Revisa internet y recarga.";
  }
}

async function startCamera() {
  if (isRecording) {
    setStatus("Detén la grabación antes de reiniciar", "recording");
    return;
  }

  stopCamera();

  try {
    const selectedDeviceId = cameraSelect.value;
    const videoConstraints = selectedDeviceId
      ? { deviceId: { exact: selectedDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
      : { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } };

    stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: false });
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
    setStatus(visionReady ? "Cámara activa" : "Sin IA", "idle");
    detectLoop();
  } catch (error) {
    console.error(error);
    setStatus("Sin permiso", "idle");
    alert("No se pudo abrir la cámara. Revisa permisos del navegador.");
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

  cameraSelect.innerHTML = '<option value="">Cámara automática</option>';
  cameras.forEach((camera, index) => {
    const option = document.createElement("option");
    option.value = camera.deviceId;
    option.textContent = camera.label || `Cámara ${index + 1}`;
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
  if (visionReady && now - lastInferenceAt > 90) {
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
    if (mode === "holistic" || mode === "poseHands" || mode === "pose") {
      const poseResult = poseLandmarker.detectForVideo(video, now);
      poses.push(...(poseResult.landmarks || []));
    }

    if (mode === "holistic" || mode === "poseHands") {
      const handResult = handLandmarker.detectForVideo(video, now);
      hands.push(...(handResult.landmarks || []));
    }

    if (mode === "holistic") {
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

  const mode = aiModeSelect.value;
  const visual = visualModeSelect.value;
  classifyAiForMotion();

  if (!drawOverlayInput.checked) return;

  if (mode === "motion" || visual === "trail" || visual === "heat") {
    drawMotionOverlay(overlayContext, lastMotion, rect.width, rect.height, visual);
  }

  if (mode !== "motion" && visual !== "trail" && visual !== "heat") {
    drawAiOverlay(overlayContext, lastStaticAi, rect.width, rect.height, visual, "static");
    drawAiOverlay(overlayContext, lastVisibleAi, rect.width, rect.height, visual, "active");
  }
}

function classifyAiForMotion() {
  if (!motionGateInput?.checked) {
    lastVisibleAi = lastAi;
    lastStaticAi = { poses: [], hands: [], faces: [] };
    previousAiForMotion = cloneAiLandmarks(lastAi);
    return;
  }

  const classified = {
    active: { poses: [], hands: [], faces: [] },
    static: { poses: [], hands: [], faces: [] },
  };

  classifyLandmarkGroups(lastAi.poses, previousAiForMotion.poses, "pose").forEach((item) => {
    classified[item.active ? "active" : "static"].poses.push(item.landmarks);
  });
  classifyLandmarkGroups(lastAi.hands, previousAiForMotion.hands, "hand").forEach((item) => {
    classified[item.active ? "active" : "static"].hands.push(item.landmarks);
  });
  classifyLandmarkGroups(lastAi.faces, previousAiForMotion.faces, "face").forEach((item) => {
    classified[item.active ? "active" : "static"].faces.push(item.landmarks);
  });

  lastVisibleAi = classified.active;
  lastStaticAi = classified.static;
  previousAiForMotion = cloneAiLandmarks(lastAi);
}

function classifyLandmarkGroups(groups, previousGroups, type) {
  return groups.map((landmarks, index) => {
    const previous = previousGroups[index];
    return {
      landmarks,
      active: landmarkGroupHasTrueMotion(landmarks, previous, type),
    };
  });
}

function cloneAiLandmarks(ai) {
  return {
    poses: ai.poses.map(cloneLandmarks),
    hands: ai.hands.map(cloneLandmarks),
    faces: ai.faces.map(cloneLandmarks),
  };
}

function cloneLandmarks(landmarks) {
  return landmarks.map((landmark) => ({
    x: landmark.x,
    y: landmark.y,
    z: landmark.z,
    visibility: landmark.visibility,
  }));
}

function landmarkGroupHasTrueMotion(landmarks, previous, type) {
  if (!landmarks?.length) return false;

  const motionScore = getMotionScoreInsideLandmarks(landmarks, type);
  const landmarkShift = getLandmarkShift(landmarks, previous, type);
  const requiredMotion = type === "pose" ? 2.2 : type === "hand" ? 0.7 : 1.4;
  const requiredShift = type === "pose" ? 0.018 : type === "hand" ? 0.012 : 0.01;

  if (motionScore >= requiredMotion) return true;
  if (!lastMotion.points.length) return false;

  const localMotionRatio = motionScore / requiredMotion;
  if (lastMotion.filtered && localMotionRatio < 0.7) return false;

  return landmarkShift >= requiredShift && localMotionRatio >= 0.35;
}
function getMotionScoreInsideLandmarks(landmarks, type) {
  if (!lastMotion.points.length) return 0;

  const visible = landmarks.filter((landmark) => isVisible(landmark, type === "face" ? 0.12 : 0.2));
  if (!visible.length) return 0;

  const padding = type === "hand" ? 0.06 : type === "face" ? 0.05 : 0.07;
  const minX = Math.max(0, Math.min(...visible.map((landmark) => landmark.x)) - padding);
  const maxX = Math.min(1, Math.max(...visible.map((landmark) => landmark.x)) + padding);
  const minY = Math.max(0, Math.min(...visible.map((landmark) => landmark.y)) - padding);
  const maxY = Math.min(1, Math.max(...visible.map((landmark) => landmark.y)) + padding);

  return lastMotion.points.reduce((score, point) => {
    const x = point.x / grid.width;
    const y = point.y / grid.height;
    if (x < minX || x > maxX || y < minY || y > maxY) return score;
    if (point.strength < 0.38) return score;
    return score + point.strength;
  }, 0);
}

function getLandmarkShift(landmarks, previous, type) {
  if (!previous || previous.length !== landmarks.length) return 0;

  const shifts = landmarks
    .map((landmark, index) => {
      const old = previous[index];
      if (!isVisible(landmark, type === "face" ? 0.12 : 0.2) || !old) return 0;
      return Math.hypot(landmark.x - old.x, landmark.y - old.y);
    })
    .sort((a, b) => b - a);

  const sampleSize = type === "face" ? 20 : type === "hand" ? 8 : 10;
  const sample = shifts.slice(0, sampleSize);
  if (!sample.length) return 0;
  return sample.reduce((sum, value) => sum + value, 0) / sample.length;
}

function drawAiOverlay(context, ai, width, height, visual, state = "active") {
  const transform = getVideoTransform(width, height);
  const active = state === "active";
  const poseColor = active ? "#2cff9a" : "rgba(180, 205, 220, 0.62)";
  const posePoint = active ? "#ffdf3d" : "rgba(180, 205, 220, 0.56)";
  const handColor = active ? "#ff4fd8" : "rgba(180, 205, 220, 0.58)";
  const handPoint = active ? "#ff4b4b" : "rgba(180, 205, 220, 0.52)";
  const faceColor = active ? "#67d7ff" : "rgba(180, 205, 220, 0.5)";

  ai.poses.forEach((pose) => {
    drawConnections(context, pose, POSE_CONNECTIONS, transform, poseColor, active ? 4 : 2, 0.35, !active);
    drawLandmarkPoints(context, pose, transform, posePoint, active ? "#ffffff" : "rgba(255, 255, 255, 0.35)", active ? 5 : 3, 0.35);
    if (!active) drawLandmarkLabel(context, pose, transform, "Forma quieta", faceColor);
  });

  ai.hands.forEach((hand) => {
    drawConnections(context, hand, HAND_CONNECTIONS, transform, handColor, active ? 3 : 2, 0.25, !active);
    drawLandmarkPoints(context, hand, transform, handPoint, active ? "#ffffff" : "rgba(255, 255, 255, 0.35)", active ? 4 : 3, 0.25);
    if (!active) drawLandmarkLabel(context, hand, transform, "Mano/forma quieta", faceColor);
  });

  ai.faces.forEach((face) => {
    drawConnections(context, face, FACE_CONNECTIONS, transform, faceColor, active ? 2 : 1.4, 0.2, !active);
    if (visual === "points" || !active) drawLandmarkPoints(context, face, transform, faceColor, "rgba(255, 255, 255, 0.45)", active ? 2.2 : 1.5, 0.2, active ? 2 : 10);
    if (!active) drawLandmarkLabel(context, face, transform, "Cara/forma quieta", faceColor);
  });
}

function drawLandmarkLabel(context, landmarks, transform, text, color) {
  const visible = landmarks.filter((landmark) => isVisible(landmark, 0.12));
  if (!visible.length) return;
  const x = Math.min(...visible.map((landmark) => toCanvasPoint(landmark, transform).x));
  const y = Math.min(...visible.map((landmark) => toCanvasPoint(landmark, transform).y));
  context.save();
  context.font = "700 14px Arial";
  context.fillStyle = color;
  context.shadowColor = "rgba(0, 0, 0, 0.75)";
  context.shadowBlur = 6;
  context.fillText(text, Math.max(8, x), Math.max(24, y - 8));
  context.restore();
}

function drawConnections(context, landmarks, connections, transform, color, lineWidth, minVisibility, dashed = false) {
  context.save();
  context.lineWidth = lineWidth;
  context.strokeStyle = color;
  context.shadowColor = "rgba(0, 0, 0, 0.72)";
  context.shadowBlur = 8;
  if (dashed) context.setLineDash([7, 6]);

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
  const movingHumanCount = lastVisibleAi.poses.length;
  const movingHandCount = lastVisibleAi.hands.length;
  const movingFaceCount = lastVisibleAi.faces.length;
  const staticCount = lastStaticAi.poses.length + lastStaticAi.hands.length + lastStaticAi.faces.length;
  const motionCount = lastMotion.points.length;
  const hasMovingAi = movingHumanCount + movingHandCount + movingFaceCount > 0;
  const hasMotion = motionCount > 0;
  const now = Date.now();

  motionStats.textContent = `Mov: ${movingHumanCount}/${movingHandCount}/${movingFaceCount} | Quietas: ${staticCount} | Puntos: ${motionCount}`;

  if (hasMovingAi || (aiModeSelect.value === "motion" && hasMotion)) {
    lastMotionAt = now;
    motionBanner.textContent = hasMovingAi ? "Movimiento con esqueleto" : "Cambio detectado";
    motionBanner.classList.add("visible");
    setStatus(isRecording ? "Grabando" : hasMovingAi ? "IA en movimiento" : "Cambio", isRecording ? "recording" : "motion");
    return;
  }

  if (staticCount > 0) {
    motionBanner.classList.remove("visible");
    setStatus(isRecording ? "Grabando" : "Forma quieta", isRecording ? "recording" : "idle");
    return;
  }

  if (now - lastMotionAt > 700) {
    motionBanner.classList.remove("visible");
    setStatus(isRecording ? "Grabando" : "Cámara activa", isRecording ? "recording" : "idle");
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

async function toggleRecording() {
  if (!stream) return;

  if (isRecording) {
    if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
    return;
  }

  if (typeof MediaRecorder === "undefined") {
    alert("Este navegador no soporta grabación con MediaRecorder.");
    return;
  }

  try {
    recordedChunks = [];
    recordingStream = await createRecordingStream();
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
    startCameraButton.disabled = true;
    switchCameraButton.disabled = true;
    cameraSelect.disabled = true;
    setStatus("Grabando", "recording");
  } catch (error) {
    console.error(error);
    cleanupRecordingTracks();
    setStatus("Error al grabar", "idle");
    alert("No se pudo iniciar la grabación en este navegador.");
  }
}

async function createRecordingStream() {
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
        drawAiOverlay(recordingContext, lastStaticAi, width, height, visualModeSelect.value, "static");
        drawAiOverlay(recordingContext, lastVisibleAi, width, height, visualModeSelect.value, "active");
      }
    }
    recordingAnimationId = requestAnimationFrame(draw);
  };

  draw();
  const canvasStream = recordingCanvas.captureStream(30);

  try {
    recordingAudioStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    recordingAudioStream.getAudioTracks().forEach((track) => canvasStream.addTrack(track));
  } catch (error) {
    console.info("Grabacion sin audio:", error);
  }

  return canvasStream;
}

function cleanupRecordingTracks() {
  if (recordingAnimationId) cancelAnimationFrame(recordingAnimationId);
  recordingAnimationId = null;
  if (recordingStream) recordingStream.getTracks().forEach((track) => track.stop());
  if (recordingAudioStream) recordingAudioStream.getTracks().forEach((track) => track.stop());
  recordingStream = null;
  recordingAudioStream = null;
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
  cleanupRecordingTracks();
  isRecording = false;
  recordButton.textContent = "Grabar";
  recordButton.classList.remove("primary");
  startCameraButton.disabled = false;
  switchCameraButton.disabled = !stream;
  cameraSelect.disabled = !stream;
  setStatus(stream ? "Cámara activa" : "Cámara inactiva", "idle");

  if (!recordedChunks.length) {
    updateDiagnostics();
    return;
  }

  const mimeType = recordedChunks[0]?.type || getSupportedMimeType() || "video/webm";
  const extension = mimeType.includes("mp4") ? "mp4" : "webm";
  const blob = new Blob(recordedChunks, { type: mimeType });
  const recording = {
    blob,
    mimeType,
    size: blob.size,
    createdAt: Date.now(),
    fileName: `detectorcam-${Date.now()}.${extension}`,
  };

  try {
    await saveRecordingToHistory(recording);
    await purgeOldRecordings();
    await renderStoredRecordings();
  } catch (error) {
    console.warn(error);
    renderEphemeralRecording(recording);
  }
}

async function initRecordings() {
  try {
    await purgeOldRecordings();
    await renderStoredRecordings();
  } catch (error) {
    console.warn(error);
    updateDiagnostics();
  }
}

function openRecordingsDb() {
  if (!window.indexedDB) return Promise.reject(new Error("IndexedDB no disponible"));
  if (dbPromise) return dbPromise;

  dbPromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(RECORDING_STORE)) {
        const store = db.createObjectStore(RECORDING_STORE, { keyPath: "id", autoIncrement: true });
        store.createIndex("createdAt", "createdAt");
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });

  return dbPromise;
}

function requestToPromise(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function getStoredRecordings() {
  const db = await openRecordingsDb();
  const tx = db.transaction(RECORDING_STORE, "readonly");
  const records = await requestToPromise(tx.objectStore(RECORDING_STORE).getAll());
  return records.sort((a, b) => b.createdAt - a.createdAt);
}

async function saveRecordingToHistory(recording) {
  const db = await openRecordingsDb();
  const tx = db.transaction(RECORDING_STORE, "readwrite");
  await requestToPromise(tx.objectStore(RECORDING_STORE).add(recording));
}

async function deleteStoredRecording(id) {
  const db = await openRecordingsDb();
  const tx = db.transaction(RECORDING_STORE, "readwrite");
  await requestToPromise(tx.objectStore(RECORDING_STORE).delete(id));
}

async function clearStoredRecordings() {
  const db = await openRecordingsDb();
  const tx = db.transaction(RECORDING_STORE, "readwrite");
  await requestToPromise(tx.objectStore(RECORDING_STORE).clear());
}

async function purgeOldRecordings() {
  let records = await getStoredRecordings();
  for (const record of records.slice(MAX_STORED_RECORDINGS)) {
    await deleteStoredRecording(record.id);
  }

  records = await getStoredRecordings();
  let totalBytes = records.reduce((total, record) => total + (record.size || record.blob?.size || 0), 0);

  while (totalBytes > MAX_STORED_BYTES && records.length > 1) {
    const oldest = records.pop();
    totalBytes -= oldest.size || oldest.blob?.size || 0;
    await deleteStoredRecording(oldest.id);
  }
}

function revokeActiveRecordingUrls() {
  activeRecordingUrls.forEach((url) => URL.revokeObjectURL(url));
  activeRecordingUrls.clear();
}

async function renderStoredRecordings() {
  revokeActiveRecordingUrls();
  const records = await getStoredRecordings();
  storedRecordingCount = records.length;
  storedRecordingBytes = records.reduce((total, record) => total + (record.size || record.blob?.size || 0), 0);

  recordingList.innerHTML = "";
  if (!records.length) {
    recordingList.innerHTML = '<p class="empty">Cuando termines una grabación aparecerá aqui.</p>';
    updateDiagnostics();
    return;
  }

  records.forEach((record) => addRecordingItem(record));
  updateDiagnostics();
}

function addRecordingItem(recording) {
  const url = URL.createObjectURL(recording.blob);
  activeRecordingUrls.add(url);
  const file = new File([recording.blob], recording.fileName, { type: recording.mimeType });
  const canShare = navigator.canShare?.({ files: [file] });
  const item = document.createElement("article");
  item.className = "recording-item";
  item.innerHTML = `
    <strong>${new Date(recording.createdAt).toLocaleString()}</strong>
    <video src="${url}" controls playsinline></video>
    <div class="recording-actions">
      <a href="${url}" download="${recording.fileName}">Descargar video</a>
      <button type="button" class="ghost save-video">Guardar / compartir</button>
      <button type="button" class="danger delete-video">Eliminar</button>
    </div>
  `;

  item.querySelector(".save-video").addEventListener("click", async () => {
    if (canShare) {
      await navigator.share({ files: [file], title: "DetectorCam", text: "Video DetectorCam" });
    } else {
      item.querySelector("a").click();
    }
  });

  item.querySelector(".delete-video").addEventListener("click", async () => {
    if (recording.id !== undefined) await deleteStoredRecording(recording.id);
    URL.revokeObjectURL(url);
    activeRecordingUrls.delete(url);
    item.remove();
    await renderStoredRecordings();
  });

  recordingList.append(item);
}

function renderEphemeralRecording(recording) {
  const url = URL.createObjectURL(recording.blob);
  activeRecordingUrls.add(url);
  const item = document.createElement("article");
  item.className = "recording-item";
  item.innerHTML = `
    <strong>${new Date(recording.createdAt).toLocaleString()}</strong>
    <video src="${url}" controls playsinline></video>
    <div class="recording-actions">
      <a href="${url}" download="${recording.fileName}">Descargar video</a>
      <button type="button" class="danger delete-video">Eliminar</button>
    </div>
  `;
  item.querySelector(".delete-video").addEventListener("click", () => {
    URL.revokeObjectURL(url);
    activeRecordingUrls.delete(url);
    item.remove();
    if (!recordingList.querySelector(".recording-item")) {
      recordingList.innerHTML = '<p class="empty">Cuando termines una grabación aparecerá aqui.</p>';
    }
  });
  recordingList.querySelector(".empty")?.remove();
  recordingList.prepend(item);
  updateDiagnostics();
}

function formatBytes(bytes) {
  if (!bytes) return "0 MB";
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function updateDiagnostics() {
  if (diagAi) diagAi.textContent = visionReady ? "IA cargada" : "IA cargando/error";
  if (diagCamera) diagCamera.textContent = stream ? "Cámara activa" : "Cámara inactiva";
  if (diagRecorder) {
    diagRecorder.textContent = typeof MediaRecorder === "undefined"
      ? "No soportado"
      : isRecording
        ? "Grabando"
        : "Grabador listo";
  }
  if (diagStorage) {
    diagStorage.textContent = `${storedRecordingCount}/${MAX_STORED_RECORDINGS} videos | ${formatBytes(storedRecordingBytes)}`;
  }
}

function setStatus(text, mode) {
  statusLabel.textContent = text;
  statusLabel.className = `status ${mode}`;
  updateDiagnostics();
}

async function switchCamera() {
  if (isRecording) {
    setStatus("Detén la grabación antes de cambiar cámara", "recording");
    return;
  }
  cameraSelect.value = "";
  facingMode = facingMode === "environment" ? "user" : "environment";
  await startCamera();
}

startCameraButton.addEventListener("click", startCamera);
recordButton.addEventListener("click", toggleRecording);
switchCameraButton.addEventListener("click", switchCamera);
cameraSelect.addEventListener("change", () => {
  if (isRecording) {
    setStatus("Detén la grabación antes de cambiar cámara", "recording");
    return;
  }
  startCamera();
});
enhanceViewInput.addEventListener("change", () => video.classList.toggle("enhanced", enhanceViewInput.checked));
motionGateInput?.addEventListener("change", () => {
  classifyAiForMotion();
  updateStatus();
});
clearListButton.addEventListener("click", async () => {
  try {
    await clearStoredRecordings();
  } catch (error) {
    console.warn(error);
  }
  revokeActiveRecordingUrls();
  storedRecordingCount = 0;
  storedRecordingBytes = 0;
  recordingList.innerHTML = '<p class="empty">Cuando termines una grabación aparecerá aqui.</p>';
  updateDiagnostics();
});
window.addEventListener("resize", resizeOverlay);

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("./service-worker.js").catch(console.error);
}