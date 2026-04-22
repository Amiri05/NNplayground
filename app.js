// =========================
// State
// =========================
let learningRate = 0.03;
let hiddenSize = 4;
let isTraining = false;
let animationId = null;
let epoch = 0;
let lossHistory = [];

let trainPercent = 50;
let trainData = [];
let testData = [];
let noiseLevel = 0.08;
let datasetType = "circle";
let showTrainData = true;

let data = [];

let W1, b1, W2, b2;

// =========================
// DOM
// =========================
const canvas = document.getElementById("plot");
const ctx = canvas.getContext("2d");

const lossCanvas = document.getElementById("lossPlot");
const lossCtx = lossCanvas.getContext("2d");

const datasetSelect = document.getElementById("datasetSelect");
const trainSplitSlider = document.getElementById("trainSplit");
const trainSplitValue = document.getElementById("trainSplitValue");
const showTrainCheckbox = document.getElementById("showTrainData");
const noiseSlider = document.getElementById("noise");
const noiseValue = document.getElementById("noiseValue");
const lrSlider = document.getElementById("lr");
const hiddenSlider = document.getElementById("hidden");
const trainBtn = document.getElementById("trainBtn");
const resetBtn = document.getElementById("resetBtn");
const statusEl = document.getElementById("status");

// Initialize state from UI
datasetType = datasetSelect.value;
trainPercent = parseInt(trainSplitSlider.value, 10);
trainSplitValue.textContent = `${trainPercent}%`;

showTrainData = showTrainCheckbox.checked;

noiseLevel = parseFloat(noiseSlider.value);
noiseValue.textContent = noiseLevel.toFixed(2);

learningRate = parseFloat(lrSlider.value);
hiddenSize = parseInt(hiddenSlider.value, 10);

// =========================
// Utils
// =========================
function randn() {
  return (Math.random() * 2 - 1) * 0.5;
}

function randn_bm() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function tanh(x) {
  return Math.tanh(x);
}

function dtanh(y) {
  return 1 - y * y;
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function shuffle(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

function toCanvasX(x) {
  return ((x + 1) / 2) * canvas.width;
}

function toCanvasY(y) {
  return canvas.height - ((y + 1) / 2) * canvas.height;
}

// =========================
// Dataset Generators
// =========================
function generateCircleData(n = 400, noise = 0.1) {
  const points = [];
  const half = Math.floor(n / 2);

  for (let i = 0; i < half; i++) {
    const angle = Math.random() * Math.PI * 2;
    const radius = Math.random() * 0.35;

    let x = radius * Math.cos(angle);
    let y = radius * Math.sin(angle);

    x += randn_bm() * noise;
    y += randn_bm() * noise;

    points.push({ x, y, label: 1 });
  }

  for (let i = 0; i < half; i++) {
    const angle = Math.random() * Math.PI * 2;
    const radius = 0.6 + Math.random() * 0.35;

    let x = radius * Math.cos(angle);
    let y = radius * Math.sin(angle);

    x += randn_bm() * noise;
    y += randn_bm() * noise;

    points.push({ x, y, label: 0 });
  }

  return points;
}

function generateXORData(n = 400, noise = 0.08) {
  const points = [];

  for (let i = 0; i < n; i++) {
    let x = Math.random() * 2 - 1;
    let y = Math.random() * 2 - 1;

    x += randn_bm() * noise;
    y += randn_bm() * noise;

    const label = x * y >= 0 ? 1 : 0;
    points.push({ x, y, label });
  }

  return points;
}

function generateGaussianData(n = 400, noise = 0.1) {
  const points = [];
  const half = Math.floor(n / 2);

  const spread = 0.1; // base cluster size (always there)

  for (let i = 0; i < half; i++) {
    let x = -0.5 + randn_bm() * spread;
    let y = -0.5 + randn_bm() * spread;

    // extra noise
    x += randn_bm() * noise;
    y += randn_bm() * noise;

    points.push({ x, y, label: 0 });
  }

  for (let i = 0; i < half; i++) {
    let x = 0.5 + randn_bm() * spread;
    let y = 0.5 + randn_bm() * spread;

    x += randn_bm() * noise;
    y += randn_bm() * noise;

    points.push({ x, y, label: 1 });
  }

  return points;
}

function generateSpiralData(n = 400, noise = 0.08) {
  const points = [];
  const half = Math.floor(n / 2);

  for (let i = 0; i < half; i++) {
    const t = i / half;
    const angle = 1.75 * Math.PI * t * 2;
    const radius = t;

    let x = radius * Math.cos(angle);
    let y = radius * Math.sin(angle);

    // scale noise with radius
    const localNoise = noise * (0.1 + radius);

    x += randn_bm() * localNoise;
    y += randn_bm() * localNoise;

    points.push({ x, y, label: 0 });
  }

  for (let i = 0; i < half; i++) {
    const t = i / half;
    const angle = 1.75 * Math.PI * t * 2 + Math.PI;
    const radius = t;

    let x = radius * Math.cos(angle);
    let y = radius * Math.sin(angle);

    const localNoise = noise * (0.3 + radius);

    x += randn_bm() * localNoise;
    y += randn_bm() * localNoise;

    points.push({ x, y, label: 1 });
  }

  return points;
}

function generateDataset(type = datasetType, n = 400, noise = noiseLevel) {
  switch (type) {
    case "circle":
      return generateCircleData(n, noise);
    case "xor":
      return generateXORData(n, noise);
    case "gaussian":
      return generateGaussianData(n, noise);
    case "spiral":
      return generateSpiralData(n, noise);
    default:
      return generateCircleData(n, noise);
  }
}

function splitData(points, percent) {
  const shuffled = [...points];
  shuffle(shuffled);

  const trainSize = Math.floor((percent / 100) * shuffled.length);
  trainData = shuffled.slice(0, trainSize);
  testData = shuffled.slice(trainSize);
}

function rebuildDataset() {
  data = generateDataset(datasetType, 400, noiseLevel);
  splitData(data, trainPercent);
}

// =========================
// Network
// =========================
function initNetwork(size) {
  W1 = Array.from({ length: size }, () => [randn(), randn()]);
  b1 = Array.from({ length: size }, () => randn());
  W2 = Array.from({ length: size }, () => randn());
  b2 = randn();
}

function forward(x1, x2) {
  const hidden = [];
  const hiddenRaw = [];

  for (let i = 0; i < W1.length; i++) {
    const z = W1[i][0] * x1 + W1[i][1] * x2 + b1[i];
    hiddenRaw.push(z);
    hidden.push(tanh(z));
  }

  let outRaw = b2;
  for (let i = 0; i < W2.length; i++) {
    outRaw += W2[i] * hidden[i];
  }

  const out = sigmoid(outRaw);
  return { hiddenRaw, hidden, outRaw, out };
}

function trainStep(sample, lr) {
  const x1 = sample.x;
  const x2 = sample.y;
  const y = sample.label;

  const { hidden, out } = forward(x1, x2);
  const dOut = out - y;

  const oldW2 = [...W2];

  for (let i = 0; i < W2.length; i++) {
    W2[i] -= lr * dOut * hidden[i];
  }
  b2 -= lr * dOut;

  for (let i = 0; i < W1.length; i++) {
    const dHidden = dOut * oldW2[i] * dtanh(hidden[i]);
    W1[i][0] -= lr * dHidden * x1;
    W1[i][1] -= lr * dHidden * x2;
    b1[i] -= lr * dHidden;
  }
}

function trainEpoch(points, lr) {
  shuffle(points);
  for (const p of points) {
    trainStep(p, lr);
  }
}

// =========================
// Metrics
// =========================
function computeLoss(points) {
  if (points.length === 0) return 0;

  let totalLoss = 0;

  for (const p of points) {
    const { out } = forward(p.x, p.y);
    const clipped = Math.min(Math.max(out, 1e-7), 1 - 1e-7);

    totalLoss += -(
      p.label * Math.log(clipped) +
      (1 - p.label) * Math.log(1 - clipped)
    );
  }

  return totalLoss / points.length;
}

// =========================
// Drawing
// =========================
function drawTrainPoints(points) {
  ctx.save();
  ctx.globalAlpha = 0.85;

  for (const p of points) {
    const x = toCanvasX(p.x);
    const y = toCanvasY(p.y);

    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = p.label === 1 ? "#7172d6" : "#efa615";
    ctx.fill();
    ctx.strokeStyle = "black";
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  ctx.restore();
}

function drawTestPoints(points) {
  ctx.save();
  ctx.globalAlpha = 0.85;

  for (const p of points) {
    const x = toCanvasX(p.x);
    const y = toCanvasY(p.y);

    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = p.label === 1 ? "#7172d6" : "#efa615";
    ctx.fill();
    ctx.strokeStyle = "white";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  ctx.restore();
}

function drawDecisionBoundary() {
  const image = ctx.createImageData(canvas.width, canvas.height);

  for (let py = 0; py < canvas.height; py++) {
    for (let px = 0; px < canvas.width; px++) {
      const x = (px / canvas.width) * 2 - 1;
      const y = -((py / canvas.height) * 2 - 1);

      const { out } = forward(x, y);
      const dist = Math.abs(out - 0.5) * 2;

      let baseR, baseG, baseB;

      if (out < 0.5) {
        // blue side
        baseR = 255;
        baseG = 180;
        baseB = 100;
      } else {
        // orange side
        baseR = 70;
        baseG = 130;
        baseB = 180;
      }

      const r = lerp(255, baseR, dist);
      const g = lerp(255, baseG, dist);
      const b = lerp(255, baseB, dist);

      const idx = (py * canvas.width + px) * 4;
      image.data[idx] = Math.round(r);
      image.data[idx + 1] = Math.round(g);
      image.data[idx + 2] = Math.round(b);
      image.data[idx + 3] = 255;
    }
  }

  ctx.putImageData(image, 0, 0);
}

function drawLoss() {
  lossCtx.clearRect(0, 0, lossCanvas.width, lossCanvas.height);

  if (lossHistory.length < 2) return;

  const maxLoss = Math.max(...lossHistory);
  const minLoss = Math.min(...lossHistory);

  lossCtx.beginPath();

  lossHistory.forEach((loss, i) => {
    const x = (i / (lossHistory.length - 1)) * lossCanvas.width;
    const y =
      lossCanvas.height -
      ((loss - minLoss) / (maxLoss - minLoss + 1e-6)) * lossCanvas.height;

    if (i === 0) lossCtx.moveTo(x, y);
    else lossCtx.lineTo(x, y);
  });

  lossCtx.strokeStyle = "black";
  lossCtx.lineWidth = 2;
  lossCtx.stroke();
}

function render() {
  drawDecisionBoundary();

  if (showTrainData) {
    drawTrainPoints(trainData);
  }

  drawTestPoints(testData);
  drawLoss();
}

// =========================
// Training Loop
// =========================
function updateStatus() {
  const trainLoss = computeLoss(trainData);
  const testLoss = computeLoss(testData);

  statusEl.textContent =
    `Dataset: ${datasetType} | ` +
    `LR: ${learningRate.toFixed(3)} | ` +
    `Hidden: ${hiddenSize} | ` +
    `Epoch: ${epoch} | ` +
    `Train: ${trainPercent}% | ` +
    `Train Loss: ${trainLoss.toFixed(4)} | ` +
    `Test Loss: ${testLoss.toFixed(4)}`;
}

function trainingLoop() {
  if (!isTraining) return;

  trainEpoch(trainData, learningRate);
  epoch++;

  const loss = computeLoss(trainData);
  lossHistory.push(loss);
  if (lossHistory.length > 200) {
    lossHistory.shift();
  }

  render();
  updateStatus();

  animationId = requestAnimationFrame(trainingLoop);
}

function startTraining() {
  if (isTraining) return;
  isTraining = true;
  trainBtn.textContent = "Pause";
  trainingLoop();
}

function stopTraining() {
  isTraining = false;
  trainBtn.textContent = "Train";

  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
}

// =========================
// Event Listeners
// =========================
datasetSelect.addEventListener("change", () => {
  datasetType = datasetSelect.value;
  stopTraining();
  rebuildDataset();
  initNetwork(hiddenSize);
  epoch = 0;
  lossHistory = [];
  render();
  updateStatus();
});

trainSplitSlider.addEventListener("input", () => {
  trainPercent = parseInt(trainSplitSlider.value, 10);
  trainSplitValue.textContent = `${trainPercent}%`;

  stopTraining();
  rebuildDataset();
  initNetwork(hiddenSize);
  epoch = 0;
  lossHistory = [];
  render();
  updateStatus();
});

noiseSlider.addEventListener("input", () => {
  noiseLevel = parseFloat(noiseSlider.value);
  noiseValue.textContent = noiseLevel.toFixed(2);

  stopTraining();
  rebuildDataset();
  initNetwork(hiddenSize);
  epoch = 0;
  lossHistory = [];
  render();
  updateStatus();
});

showTrainCheckbox.addEventListener("change", () => {
  showTrainData = showTrainCheckbox.checked;
  render();
});

lrSlider.addEventListener("input", () => {
  learningRate = parseFloat(lrSlider.value);
  updateStatus();
});

hiddenSlider.addEventListener("input", () => {
  hiddenSize = parseInt(hiddenSlider.value, 10);
  stopTraining();
  initNetwork(hiddenSize);
  epoch = 0;
  lossHistory = [];
  render();
  updateStatus();
});

trainBtn.addEventListener("click", () => {
  if (isTraining) stopTraining();
  else startTraining();
});

resetBtn.addEventListener("click", () => {
  stopTraining();
  rebuildDataset();
  initNetwork(hiddenSize);
  epoch = 0;
  lossHistory = [];
  render();
  updateStatus();
});

// =========================
// Startup
// =========================
rebuildDataset();
initNetwork(hiddenSize);
render();
updateStatus();


const lrSlider = document.getElementById('lrSlider');  
  const hiddenSlider = document.getElementById('hiddenSlider');
  const trainSlider = document.getElementById('trainSlider');
  const noiseSlider = document.getElementById('noiseSlider');
  const lrVal = document.getElementById('lrVal');

  const infoDataset = document.getElementById('infoDataset');
  const infoHidden = document.getElementById('infoHidden');
  const infoTrain = document.getElementById('infoTrain');
  const infoNoise = document.getElementById('infoNoise');
  const infoTrainLoss = document.getElementById('infoTrainLoss');
  const infoTestLoss = document.getElementById('infoTestLoss');
 
  lrSlider.addEventListener('input', () => {
    lrVal.textContent = (parseInt(lrSlider.value) / 1000).toFixed(3);
  });

  hiddenSlider.addEventListener('input', () => {
  console.log("Hidden:", hiddenSlider.value);
  });

  trainSlider.addEventListener('input', () => {
    console.log("Train %:", trainSlider.value);
  });

  noiseSlider.addEventListener('input', () => {
    console.log("Noise:", noiseSlider.value);
  });

function updateInfo() {
  infoDataset.textContent =
    datasetSelect.value.charAt(0).toUpperCase() + datasetSelect.value.slice(1);
  infoHidden.textContent = hiddenSlider.value;
  infoTrain.textContent = `${trainSlider.value}%`;
  infoNoise.textContent = `${noiseSlider.value / 100}`;
}

lrSlider.addEventListener('input', () => {
  lrVal.textContent = (parseInt(lrSlider.value) / 1000).toFixed(3);
  updateInfo();
});

hiddenSlider.addEventListener('input', () => {
  updateInfo();
});

trainSlider.addEventListener('input', () => {
  updateInfo();
});

noiseSlider.addEventListener('input', () => {
  updateInfo();
});


  function syncPlaybackUI() {
    pauseIcon.innerHTML = paused ? PLAY : PAUSE;
    dot.className = paused ? 'dot off' : 'dot';
  }

  let paused = true;
let epoch = 0;

const pauseBtn = document.getElementById('pauseBtn');
const pauseIcon = document.getElementById('pauseIcon');
const epochVal = document.getElementById('epochVal');
const dot = document.getElementById('dot');
const resetBtn = document.getElementById('resetBtn');

const PLAY = '<polygon points="2.5,1.5 10,6 2.5,10.5"/>';
const PAUSE = '<rect x="1.5" y="1" width="3.5" height="10" rx="1"/><rect x="7" y="1" width="3.5" height="10" rx="1"/>';

function syncPlaybackUI() {
  pauseIcon.innerHTML = paused ? PLAY : PAUSE;
  dot.className = paused ? 'dot off' : 'dot';
}

function fmt(n) {
  return String(n).padStart(6, '0').replace(/(\d{3})(\d{3})/, '$1,$2');
}

function render() {
  epochVal.textContent = fmt(epoch);
}

pauseBtn.addEventListener('click', () => {
  paused = !paused;
  syncPlaybackUI();
});

resetBtn.addEventListener('click', () => {
  epoch = 0;
  paused = true;
  syncPlaybackUI();
  render();
});

syncPlaybackUI();
render();
updateInfo();

  // 1 epoch per tick, every 20ms = 50 epochs per second
  setInterval(() => {
    if (!paused) {
      epoch += 1;
      render();
    }
  }, 20);
 
  render();

  document.querySelectorAll('.slider-row input[type=range]').forEach(input => {
    const out = document.getElementById(input.dataset.out);
    input.addEventListener('input', () => {
      out.textContent = (parseInt(input.value) / 100).toFixed(2);
    });
  });
 
  document.querySelectorAll('.img-card input[type=file]').forEach(inp => {
    inp.addEventListener('change', e => {
      const file = e.target.files[0];
      if (!file) return;
      const idx   = inp.dataset.card;
      const card  = document.getElementById('imgCard' + idx);
      const thumb = document.getElementById('imgThumb' + idx);
      const reader = new FileReader();
      reader.onload = ev => {
        thumb.src = ev.target.result;
        card.classList.add('has-image', 'selected');
      };
      reader.readAsDataURL(file);
    });
  });