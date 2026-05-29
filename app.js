import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  ObjectDetector,
  PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

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
const diagEvent = document.querySelector("#diagEvent");

const overlayContext = overlay.getContext("2d");
const analysisCanvas = document.createElement("canvas");
const analysisContext = analysisCanvas.getContext("2d", { willReadFrequently: true });
const recordingCanvas = document.createElement("canvas");
const recordingContext = recordingCanvas.getContext("2d");

const MODEL_URLS = {
  pose: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
  hands: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
  face: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
  object: "https://storage.googleapis.com/mediapipe-tasks/object_detector/efficientdet_lite0_uint8.tflite",
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
let aiStatusText = "IA cargando";
let aiErrorText = "";
let poseLandmarker = null;
let handLandmarker = null;
let faceLandmarker = null;
let objectDetector = null;
let lastObjectInferenceAt = 0;
let lastAi = { poses: [], hands: [], faces: [] };
let lastVisibleAi = { poses: [], hands: [], faces: [] };
let lastStaticAi = { poses: [], hands: [], faces: [] };
let previousAiForMotion = { poses: [], hands: [], faces: [] };
let lastMotion = { boxes: [], points: [], strength: 0, filtered: false };
let lastObjects = [];
let lastEventSummary = { type: "none", label: "Sin evento", alert: false };
let recordingEventTypes = new Set();
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

const ANIMAL_LABELS = new Set([
  "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"
]);

const OBJECT_LABELS_ES = {
  person: "persona",
  bird: "ave",
  cat: "gato",
  dog: "perro",
  horse: "caballo",
  sheep: "oveja",
  cow: "vaca",
  elephant: "elefante",
  bear: "oso",
  zebra: "cebra",
  giraffe: "jirafa",
  car: "carro",
  motorcycle: "moto",
  bicycle: "bicicleta",
  bus: "bus",
  truck: "camion",
  chair: "silla",
  couch: "sofa",
  bed: "cama",
  backpack: "morral",
  handbag: "bolso",
  suitcase: "maleta",
  bottle: "botella",
  cup: "vaso",
  cell_phone: "celular",
  laptop: "portatil",
  tv: "televisor",
  book: "libro",
};

const EVENT_LABELS = {
  person: "Persona en movimiento",
  animal: "Animal en movimiento",
  object: "Objeto en movimiento",
  anomaly: "Movimiento no identificado",
  quietPerson: "Persona quieta",
  quietObject: "Objeto/forma quieta",
  none: "Sin evento",
};

analysisCanvas.width = grid.width;
analysisCanvas.height = grid.height;

initAi();
initRecordings();
updateDiagnostics();

async function initAi() {
  setStatus("Cargando IA", "idle");
  aiStatusText = "IA cargando";
  aiErrorText = "";
  motionStats.textContent = "IA cargando... puedes abrir la camara";
  updateDiagnostics();

  const loadErrors = [];

  async function loadModel(label, factory, gpuOptions) {
    try {
      return await factory(gpuOptions);
    } catch (gpuError) {
      console.warn(`${label} no cargo con GPU. Intentando CPU.`, gpuError);
      loadErrors.push(`${label} GPU: ${gpuError?.message || gpuError}`);
      try {
        const cpuOptions = {
          ...gpuOptions,
          baseOptions: {
            ...gpuOptions.baseOptions,
            delegate: "CPU",
          },
        };
        return await factory(cpuOptions);
      } catch (cpuError) {
        console.warn(`${label} no disponible`, cpuError);
        loadErrors.push(`${label} CPU: ${cpuError?.message || cpuError}`);
        return null;
      }
    }
  }

  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );

    poseLandmarker = await loadModel(
      "Cuerpo",
      (options) => PoseLandmarker.createFromOptions(vision, options),
      {
        baseOptions: { modelAssetPath: MODEL_URLS.pose, delegate: "GPU" },
        runningMode: "VIDEO",
        numPoses: 2,
        minPoseDetectionConfidence: 0.25,
        minPosePresenceConfidence: 0.25,
        minTrackingConfidence: 0.25,
      }
    );

    handLandmarker = await loadModel(
      "Manos",
      (options) => HandLandmarker.createFromOptions(vision, options),
      {
        baseOptions: { modelAssetPath: MODEL_URLS.hands, delegate: "GPU" },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.25,
        minHandPresenceConfidence: 0.25,
        minTrackingConfidence: 0.25,
      }
    );

    faceLandmarker = await loadModel(
      "Cara",
      (options) => FaceLandmarker.createFromOptions(vision, options),
      {
        baseOptions: { modelAssetPath: MODEL_URLS.face, delegate: "GPU" },
        runningMode: "VIDEO",
        numFaces: 1,
        minFaceDetectionConfidence: 0.25,
        minFacePresenceConfidence: 0.25,
        minTrackingConfidence: 0.25,
      }
    );

    objectDetector = await loadModel(
      "Objetos",
      (options) => ObjectDetector.createFromOptions(vision, options),
      {
        baseOptions: { modelAssetPath: MODEL_URLS.object, delegate: "GPU" },
        runningMode: "VIDEO",
        maxResults: 8,
        scoreThreshold: 0.42,
      }
    );

    visionReady = Boolean(poseLandmarker || handLandmarker || faceLandmarker || objectDetector);
    const loaded = [
      poseLandmarker ? "cuerpo" : null,
      handLandmarker ? "manos" : null,
      faceLandmarker ? "cara" : null,
      objectDetector ? "objetos" : null,
    ].filter(Boolean);

    if (visionReady) {
      aiStatusText = `IA lista: ${loaded.join(" + ")}`;
      aiErrorText = loadErrors.length ? loadErrors.slice(-2).join(" | ") : "";
      setStatus("IA lista", "idle");
      motionStats.textContent = `${aiStatusText} | abre la camara`;
    } else {
      aiStatusText = "IA sin modelos";
      aiErrorText = loadErrors.slice(-3).join(" | ");
      setStatus("IA no cargo", "idle");
      motionStats.textContent = "IA sin modelos. La camara puede grabar sin esqueleto.";
    }
  } catch (error) {
    console.error(error);
    visionReady = false;
    aiStatusText = "IA error base";
    aiErrorText = error?.message || String(error);
    setStatus("IA no cargo", "idle");
    motionStats.textContent = `IA error: ${aiErrorText}`;
  }
}

async function startCamera() {
  stopCamera();

  try {
    const selectedDeviceId = cameraSelect.value;
    const videoConstraints = selectedDeviceId
      ? { deviceId: { exact: selectedDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
      : { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } };

    stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: false });
    video.srcObject = stream;
    await video.play();
    startCameraButton.textContent = "Reiniciar";
    recordButton.disabled = false;
    switchCameraButton.disabled = false;
    cameraSelect.disabled = false;
    try {
      await refreshCameraList();
    } catch (cameraListError) {
      console.warn("No se pudo listar camaras", cameraListError);
    }
    resizeOverlay();
    previousFrame = null;
    backgroundBrightness = null;
    trailPoints = [];
    lastAi = { poses: [], hands: [], faces: [] };
    lastVisibleAi = { poses: [], hands: [], faces: [] };
    lastStaticAi = { poses: [], hands: [], faces: [] };
    previousAiForMotion = { poses: [], hands: [], faces: [] };
    lastObjects = [];
    lastEventSummary = { type: "none", label: "Sin evento", alert: false };
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

    if (objectDetector && now - lastObjectInferenceAt > 650) {
      const objectResult = objectDetector.detectForVideo(video, now);
      lastObjects = normalizeObjectDetections(objectResult.detections || []);
      lastObjectInferenceAt = now;
    }

    lastAi = { poses, hands, faces };
  } catch (error) {
    console.warn(error);
    aiErrorText = error?.message || String(error);
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
    drawObjectOverlay(overlayContext, lastObjects, rect.width, rect.height);
    if (motionGateInput?.checked) {
      drawAiOverlay(overlayContext, lastStaticAi, rect.width, rect.height, visual, "static");
      drawAiOverlay(overlayContext, lastVisibleAi, rect.width, rect.height, visual, "active");
    } else {
      drawAiOverlay(overlayContext, lastAi, rect.width, rect.height, visual, "active");
    }
  } else if (mode === "motion") {
    drawObjectOverlay(overlayContext, lastObjects, rect.width, rect.height);
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

function mergeAi(a, b) {
  return {
    poses: [...(a?.poses || []), ...(b?.poses || [])],
    hands: [...(a?.hands || []), ...(b?.hands || [])],
    faces: [...(a?.faces || []), ...(b?.faces || [])],
  };
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

function normalizeObjectDetections(detections) {
  const frameWidth = video.videoWidth || grid.width;
  const frameHeight = video.videoHeight || grid.height;

  return detections
    .map((detection) => {
      const categories = [...(detection.categories || [])].sort((a, b) => (b.score || 0) - (a.score || 0));
      const category = categories[0] || {};
      const rawLabel = String(category.categoryName || category.displayName || "object").toLowerCase().replace(/ /g, "_");
      const box = normalizeBoundingBox(detection.boundingBox, frameWidth, frameHeight);
      if (!box) return null;

      const score = category.score || 0;
      const kind = rawLabel === "person" ? "person" : ANIMAL_LABELS.has(rawLabel) ? "animal" : "object";
      const motionScore = getMotionScoreInsideNormalizedBox(box, kind === "person" ? 0.04 : 0.03);
      const movingThreshold = kind === "person" ? 1.8 : kind === "animal" ? 0.9 : 1.15;

      return {
        label: rawLabel,
        labelEs: translateObjectLabel(rawLabel),
        score,
        kind,
        box,
        moving: motionScore >= movingThreshold,
        motionScore,
      };
    })
    .filter(Boolean)
    .filter((object) => object.score >= 0.42);
}

function normalizeBoundingBox(box, frameWidth, frameHeight) {
  if (!box) return null;
  const originX = box.originX ?? box.x ?? 0;
  const originY = box.originY ?? box.y ?? 0;
  const rawWidth = box.width ?? 0;
  const rawHeight = box.height ?? 0;
  if (rawWidth <= 0 || rawHeight <= 0) return null;

  const normalized = rawWidth <= 1 && rawHeight <= 1;
  const x = normalized ? originX : originX / frameWidth;
  const y = normalized ? originY : originY / frameHeight;
  const width = normalized ? rawWidth : rawWidth / frameWidth;
  const height = normalized ? rawHeight : rawHeight / frameHeight;

  return {
    x: clamp(x, 0, 1),
    y: clamp(y, 0, 1),
    width: clamp(width, 0, 1),
    height: clamp(height, 0, 1),
  };
}

function translateObjectLabel(label) {
  return OBJECT_LABELS_ES[label] || label.replace(/_/g, " ");
}

function getMotionScoreInsideNormalizedBox(box, padding = 0.03) {
  if (!lastMotion.points.length) return 0;
  const minX = Math.max(0, box.x - padding);
  const maxX = Math.min(1, box.x + box.width + padding);
  const minY = Math.max(0, box.y - padding);
  const maxY = Math.min(1, box.y + box.height + padding);

  return lastMotion.points.reduce((score, point) => {
    const x = point.x / grid.width;
    const y = point.y / grid.height;
    if (x < minX || x > maxX || y < minY || y > maxY) return score;
    return score + point.strength;
  }, 0);
}

function isLikelyHumanPose(pose) {
  if (!pose?.length) return false;
  const visible = (index, min = 0.24) => isVisible(pose[index], min);
  const visibleCount = pose.filter((landmark) => isVisible(landmark, 0.24)).length;
  const headCount = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].filter((index) => visible(index, 0.22)).length;
  const armCount = [13, 14, 15, 16].filter((index) => visible(index, 0.22)).length;
  const legCount = [25, 26, 27, 28, 29, 30, 31, 32].filter((index) => visible(index, 0.22)).length;
  const hasShoulders = visible(11, 0.24) && visible(12, 0.24);
  const hasHips = visible(23, 0.22) && visible(24, 0.22);

  if (visibleCount < 8 || !hasShoulders) return false;

  const shoulderWidth = Math.abs(pose[11].x - pose[12].x);
  const shoulderY = (pose[11].y + pose[12].y) / 2;
  const hipY = hasHips ? (pose[23].y + pose[24].y) / 2 : shoulderY + 0.08;
  const torsoHeight = Math.abs(hipY - shoulderY);
  const hasHumanCore = hasHips || headCount >= 2 || armCount >= 2 || legCount >= 2;
  const plausibleScale = shoulderWidth > 0.035 && shoulderWidth < 0.75 && torsoHeight > 0.035;

  return hasHumanCore && plausibleScale;
}

function getDetectionSummary() {
  const movingHumanPoses = lastVisibleAi.poses.filter(isLikelyHumanPose).length;
  const movingPseudoPoses = lastVisibleAi.poses.length - movingHumanPoses;
  const staticHumanPoses = lastStaticAi.poses.filter(isLikelyHumanPose).length;
  const staticPseudoPoses = lastStaticAi.poses.length - staticHumanPoses;
  const movingHumanParts = lastVisibleAi.hands.length + lastVisibleAi.faces.length;
  const staticHumanParts = lastStaticAi.hands.length + lastStaticAi.faces.length;
  const movingPeople = movingHumanPoses > 0 || movingHumanParts > 0 ? Math.max(1, movingHumanPoses) : 0;
  const staticPeople = staticHumanPoses > 0 || staticHumanParts > 0 ? Math.max(1, staticHumanPoses) : 0;
  const movingAnimals = lastObjects.filter((object) => object.kind === "animal" && object.moving).length;
  const movingObjects = lastObjects.filter((object) => object.kind === "object" && object.moving).length;
  const quietObjects = lastObjects.filter((object) => !object.moving && object.kind !== "person").length + staticPseudoPoses;
  const hasUnclassifiedMotion = lastMotion.points.length > 0 && movingPeople === 0 && movingAnimals === 0 && movingObjects === 0;
  const anomalies = movingPseudoPoses + (hasUnclassifiedMotion ? 1 : 0);

  let type = "none";
  if (movingPeople > 0) type = "person";
  else if (movingAnimals > 0) type = "animal";
  else if (movingObjects > 0) type = "object";
  else if (anomalies > 0) type = "anomaly";
  else if (staticPeople > 0) type = "quietPerson";
  else if (quietObjects > 0 || lastStaticAi.poses.length > 0) type = "quietObject";

  return {
    type,
    label: EVENT_LABELS[type] || EVENT_LABELS.none,
    alert: ["person", "animal", "object", "anomaly"].includes(type),
    movingPeople,
    movingAnimals,
    movingObjects,
    anomalies,
    quiet: staticPeople + quietObjects,
    points: lastMotion.points.length,
  };
}

function rememberRecordingEvent(summary = lastEventSummary) {
  if (!isRecording || !summary?.type || summary.type === "none") return;
  recordingEventTypes.add(summary.type);
}

function getPrimaryRecordingEventType() {
  const priority = ["person", "animal", "object", "anomaly", "quietPerson", "quietObject"];
  return priority.find((type) => recordingEventTypes.has(type)) || "none";
}

function buildRecordingEventLabel() {
  const priority = ["person", "animal", "object", "anomaly", "quietPerson", "quietObject"];
  const labels = priority.filter((type) => recordingEventTypes.has(type)).map((type) => EVENT_LABELS[type]);
  return labels.length ? labels.join(" + ") : EVENT_LABELS.none;
}

function drawObjectOverlay(context, objects, width, height) {
  if (!objects.length) return;
  const transform = getVideoTransform(width, height);

  objects.forEach((object) => {
    if (object.kind === "person" && (lastVisibleAi.poses.length || lastStaticAi.poses.length || lastVisibleAi.faces.length)) return;
    const style = getObjectStyle(object);
    const rect = normalizedBoxToCanvasRect(object.box, transform);
    if (rect.width < 16 || rect.height < 16) return;

    context.save();
    context.lineWidth = object.moving ? 3 : 2;
    context.strokeStyle = style.color;
    context.fillStyle = style.fill;
    context.shadowColor = "rgba(0, 0, 0, 0.7)";
    context.shadowBlur = 8;
    if (!object.moving) context.setLineDash([7, 6]);
    context.strokeRect(rect.x, rect.y, rect.width, rect.height);
    context.fillRect(rect.x, rect.y, rect.width, rect.height);
    context.setLineDash([]);
    drawBoxLabel(context, rect, style.label, style.color);
    context.restore();
  });
}

function getObjectStyle(object) {
  if (object.kind === "animal") {
    return object.moving
      ? { color: "#ffb02e", fill: "rgba(255, 176, 46, 0.08)", label: `Animal en movimiento: ${object.labelEs}` }
      : { color: "rgba(103, 215, 255, 0.72)", fill: "rgba(103, 215, 255, 0.04)", label: `Animal quieto: ${object.labelEs}` };
  }

  if (object.kind === "person") {
    return object.moving
      ? { color: "#2cff9a", fill: "rgba(44, 255, 154, 0.08)", label: "Persona posible" }
      : { color: "rgba(103, 215, 255, 0.72)", fill: "rgba(103, 215, 255, 0.04)", label: "Persona quieta posible" };
  }

  return object.moving
    ? { color: "#ffd24a", fill: "rgba(255, 210, 74, 0.08)", label: `Objeto en movimiento: ${object.labelEs}` }
    : { color: "rgba(180, 205, 220, 0.62)", fill: "rgba(180, 205, 220, 0.04)", label: `Objeto quieto: ${object.labelEs}` };
}

function normalizedBoxToCanvasRect(box, transform) {
  return {
    x: transform.x + box.x * transform.width,
    y: transform.y + box.y * transform.height,
    width: box.width * transform.width,
    height: box.height * transform.height,
  };
}

function drawBoxLabel(context, rect, text, color) {
  context.save();
  context.font = "700 13px Arial";
  context.textBaseline = "top";
  const paddingX = 7;
  const paddingY = 5;
  const labelWidth = context.measureText(text).width + paddingX * 2;
  const labelHeight = 24;
  const x = Math.max(6, Math.min(rect.x, context.canvas.width - labelWidth - 6));
  const y = Math.max(6, rect.y - labelHeight - 4);
  context.fillStyle = "rgba(5, 8, 12, 0.78)";
  context.fillRect(x, y, labelWidth, labelHeight);
  context.fillStyle = color;
  context.fillText(text, x + paddingX, y + paddingY);
  context.restore();
}

function drawAiOverlay(context, ai, width, height, visual, state = "active") {
  const transform = getVideoTransform(width, height);
  const active = state === "active";

  ai.poses.forEach((pose) => {
    const likelyHuman = isLikelyHumanPose(pose);
    const poseColor = active ? (likelyHuman ? "#2cff9a" : "#b665ff") : likelyHuman ? "rgba(103, 215, 255, 0.72)" : "rgba(180, 205, 220, 0.62)";
    const posePoint = active ? (likelyHuman ? "#ffdf3d" : "#ffe66d") : likelyHuman ? "rgba(103, 215, 255, 0.72)" : "rgba(180, 205, 220, 0.56)";
    const label = active ? (likelyHuman ? "Persona en movimiento" : "Movimiento no identificado") : likelyHuman ? "Persona quieta" : "Forma quieta";
    drawConnections(context, pose, POSE_CONNECTIONS, transform, poseColor, active && likelyHuman ? 4 : 2, 0.35, !active || !likelyHuman);
    drawLandmarkPoints(context, pose, transform, posePoint, active ? "#ffffff" : "rgba(255, 255, 255, 0.35)", active && likelyHuman ? 5 : 3, 0.35);
    drawLandmarkLabel(context, pose, transform, label, poseColor);
  });

  ai.hands.forEach((hand) => {
    const handColor = active ? "#2cff9a" : "rgba(103, 215, 255, 0.68)";
    const handPoint = active ? "#ffdf3d" : "rgba(103, 215, 255, 0.62)";
    drawConnections(context, hand, HAND_CONNECTIONS, transform, handColor, active ? 3 : 2, 0.25, !active);
    drawLandmarkPoints(context, hand, transform, handPoint, active ? "#ffffff" : "rgba(255, 255, 255, 0.35)", active ? 4 : 3, 0.25);
    drawLandmarkLabel(context, hand, transform, active ? "Mano en movimiento" : "Mano quieta", handColor);
  });

  ai.faces.forEach((face) => {
    const faceColor = active ? "#2cff9a" : "rgba(103, 215, 255, 0.6)";
    drawConnections(context, face, FACE_CONNECTIONS, transform, faceColor, active ? 2 : 1.4, 0.2, !active);
    if (visual === "points" || !active) drawLandmarkPoints(context, face, transform, faceColor, "rgba(255, 255, 255, 0.45)", active ? 2.2 : 1.5, 0.2, active ? 2 : 10);
    drawLandmarkLabel(context, face, transform, active ? "Rostro en movimiento" : "Rostro quieto", faceColor);
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
  const summary = getDetectionSummary();
  lastEventSummary = summary;
  rememberRecordingEvent(summary);

  motionStats.textContent = `Per: ${summary.movingPeople} | Ani: ${summary.movingAnimals} | Obj: ${summary.movingObjects} | Anom: ${summary.anomalies} | Quietos: ${summary.quiet} | Puntos: ${summary.points}`;

  if (summary.alert) {
    lastMotionAt = Date.now();
    motionBanner.textContent = summary.label;
    motionBanner.className = `motion-banner visible ${summary.type}`;
    setStatus(isRecording ? "Grabando" : summary.label, isRecording ? "recording" : "motion");
    return;
  }

  motionBanner.className = "motion-banner";

  if (summary.type === "quietPerson" || summary.type === "quietObject") {
    setStatus(isRecording ? "Grabando" : summary.label, isRecording ? "recording" : "idle");
    return;
  }

  if (Date.now() - lastMotionAt > 700) {
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
  const nextWidth = Math.round(rect.width * pixelRatio);
  const nextHeight = Math.round(rect.height * pixelRatio);
  if (overlay.width !== nextWidth || overlay.height !== nextHeight) {
    overlay.width = nextWidth;
    overlay.height = nextHeight;
  }
  overlayContext.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
}

function toggleRecording() {
  if (!stream) return;
  if (typeof MediaRecorder === "undefined") {
    alert("Este navegador no permite grabar video desde la app. Prueba Chrome actualizado en Android.");
    setStatus("Grabador no soportado", "idle");
    return;
  }

  if (isRecording) {
    mediaRecorder?.stop();
    return;
  }

  try {
    recordedChunks = [];
    recordingEventTypes = new Set();
    rememberRecordingEvent(getDetectionSummary());
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
  } catch (error) {
    console.warn(error);
    if (recordingStream) recordingStream.getTracks().forEach((track) => track.stop());
    recordingStream = null;
    setStatus("Error grabador", "idle");
    alert("No se pudo iniciar la grabacion en este navegador.");
  }
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
        drawObjectOverlay(recordingContext, lastObjects, width, height);
        if (motionGateInput?.checked) {
          drawAiOverlay(recordingContext, lastStaticAi, width, height, visualModeSelect.value, "static");
          drawAiOverlay(recordingContext, lastVisibleAi, width, height, visualModeSelect.value, "active");
        } else {
          drawAiOverlay(recordingContext, lastAi, width, height, visualModeSelect.value, "active");
        }
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
  if (typeof MediaRecorder === "undefined") return "";
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

  if (!recordedChunks.length) {
    setStatus("Sin video", "idle");
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
    eventType: getPrimaryRecordingEventType(),
    eventLabel: buildRecordingEventLabel(),
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
    recordingList.innerHTML = '<p class="empty">Cuando termines una grabacion aparecera aqui.</p>';
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
    <div class="recording-meta">
      <strong>${new Date(recording.createdAt).toLocaleString()}</strong>
      <span class="event-tag ${recording.eventType || "none"}">${escapeHtml(recording.eventLabel || EVENT_LABELS.none)}</span>
    </div>
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
    <div class="recording-meta">
      <strong>${new Date(recording.createdAt).toLocaleString()}</strong>
      <span class="event-tag ${recording.eventType || "none"}">${escapeHtml(recording.eventLabel || EVENT_LABELS.none)}</span>
    </div>
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
      recordingList.innerHTML = '<p class="empty">Cuando termines una grabacion aparecera aqui.</p>';
    }
  });
  recordingList.querySelector(".empty")?.remove();
  recordingList.prepend(item);
  updateDiagnostics();
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" }[char]));
}

function formatBytes(bytes) {
  if (!bytes) return "0 MB";
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function updateDiagnostics() {
  if (diagAi) diagAi.textContent = aiErrorText ? `${aiStatusText} | ${aiErrorText}` : aiStatusText;
  if (diagCamera) diagCamera.textContent = stream ? "Camara activa" : "Camara inactiva";
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
  if (diagEvent) diagEvent.textContent = lastEventSummary.label || EVENT_LABELS.none;
}

function setStatus(text, mode) {
  statusLabel.textContent = text;
  statusLabel.className = `status ${mode}`;
  updateDiagnostics();
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
  recordingList.innerHTML = '<p class="empty">Cuando termines una grabacion aparecera aqui.</p>';
  updateDiagnostics();
});
window.addEventListener("resize", resizeOverlay);

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("./service-worker.js").catch(console.error);
}