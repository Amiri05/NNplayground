let learningRate = 0.03;
let hiddenSize = 4;
let isTraining = false;
let animationId = null;
let epoch = 0;
let lossHistory = [];

let trainPercent = 50;
let trainData = [];
let testData = [];

let showTrainData = true;

const trainSplitSlider = document.getElementById("trainSplit");
const trainSplitValue = document.getElementById("trainSplitValue");

const showTrainCheckbox = document.getElementById("showTrainData");

showTrainData = showTrainCheckbox.checked;

trainSplitSlider.addEventListener("input", () => {
  trainPercent = parseInt(trainSplitSlider.value, 10);
  trainSplitValue.textContent = `${trainPercent}%`;

  stopTraining();
  data = generateCircleData();
  splitData(data, trainPercent);
  initNetwork(hiddenSize);
  epoch = 0;
  lossHistory = [];
  render();
  updateStatus();
});

trainPercent = parseInt(trainSplitSlider.value, 10);
trainSplitValue.textContent = `${trainPercent}%`;

const lossCanvas = document.getElementById("lossPlot");
const lossCtx = lossCanvas.getContext("2d");

// Dataset Generator

// Circle Dataset
function generateCircleData(n = 400, noise = 0.08) {
  const points = [];

  for (let i = 0; i < n; i++) {
    let x = Math.random() * 2 - 1;
    let y = Math.random() * 2 - 1;

    x += (Math.random() * 2 - 1) * noise;
    y += (Math.random() * 2 - 1) * noise;

    const r = Math.sqrt(x * x + y * y);
    const label = r < 0.5 ? 1 : 0;

    points.push({ x, y, label });
  }

  return points;
}

const canvas = document.getElementById("plot");
const ctx = canvas.getContext("2d");

function splitData(points, trainPercent) {
  const shuffled = [...points];
  shuffle(shuffled);

  const trainSize = Math.floor((trainPercent / 100) * shuffled.length);

  trainData = shuffled.slice(0, trainSize);
  testData = shuffled.slice(trainSize);
}

let data = generateCircleData();
splitData(data, trainPercent);

function toCanvasX(x) {
  return ((x + 1) / 2) * canvas.width;
}

function toCanvasY(y) {
  return canvas.height - ((y + 1) / 2) * canvas.height;
}

function drawTrainPoints(points) {
  for (const p of points) {
    ctx.beginPath();
    ctx.arc(toCanvasX(p.x), toCanvasY(p.y), 3.5, 0, Math.PI * 2);
    ctx.fillStyle = p.label === 1 ? "#e9b87c" : "#71ace6";
    ctx.fill();
  }
}

function drawTestPoints(points) {
  for (const p of points) {
    ctx.beginPath();
    ctx.arc(toCanvasX(p.x), toCanvasY(p.y), 3.5, 0, Math.PI * 2);
    ctx.fillStyle = p.label === 1 ? "#a66414" : "#144372";
    ctx.fill();
  }
}

let W1, b1, W2, b2;

function randn() {
  return Math.random() * 2 - 1;
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

function initNetwork(hiddenSize) {
  W1 = Array.from({ length: hiddenSize }, () => [randn(), randn()]);
  b1 = Array.from({ length: hiddenSize }, () => randn());
  W2 = Array.from({ length: hiddenSize }, () => randn());
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

function drawDecisionBoundary() {
  const image = ctx.createImageData(canvas.width, canvas.height);

  for (let py = 0; py < canvas.height; py++) {
    for (let px = 0; px < canvas.width; px++) {
      const x = (px / canvas.width) * 2 - 1;
      const y = -((py / canvas.height) * 2 - 1);

      const { out } = forward(x, y);

      const idx = (py * canvas.width + px) * 4;

      if (out > 0.5) {
        image.data[idx] = 255;
        image.data[idx + 1] = 220;
        image.data[idx + 2] = 180;
        image.data[idx + 3] = 255;
      } else {
        image.data[idx] = 180;
        image.data[idx + 1] = 220;
        image.data[idx + 2] = 255;
        image.data[idx + 3] = 255;
      }
    }
  }

  ctx.putImageData(image, 0, 0);
}

function render() {
  drawDecisionBoundary();

  if (showTrainData) {
    drawTrainPoints(trainData);
  }

  drawTestPoints(testData);
  drawLoss();
}

function computeLoss(points) {
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

    if (i === 0) {
      lossCtx.moveTo(x, y);
    } else {
      lossCtx.lineTo(x, y);
    }
  });

  lossCtx.strokeStyle = "black";
  lossCtx.lineWidth = 2;
  lossCtx.stroke();
}

/* Renderer
    - Background Decision Boundary 
    - Training Points
    - Network Diagram
*/

// UI Controller

const lrSlider = document.getElementById("lr");
const hiddenSlider = document.getElementById("hidden");
const trainBtn = document.getElementById("trainBtn");
const resetBtn = document.getElementById("resetBtn");
const statusEl = document.getElementById("status");

learningRate = parseFloat(lrSlider.value);
hiddenSize = parseInt(hiddenSlider.value, 10);

function updateStatus() {
  const trainLoss = computeLoss(trainData);
  const testLoss = computeLoss(testData);

  statusEl.textContent =
    `LR: ${learningRate.toFixed(3)} | ` +
    `Hidden: ${hiddenSize} | ` +
    `Epoch: ${epoch} | ` +
    `Train %: ${trainPercent} | ` +
    `Train Visible: ${showTrainData ? "Yes" : "No"} | ` +
    `Train Loss: ${trainLoss.toFixed(4)} | ` +
    `Test Loss: ${testLoss.toFixed(4)}`;
}

function shuffle(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

function trainEpoch(points, lr) {
  shuffle(points);
  for (const p of points) {
    trainStep(p, lr);
  }
}

function trainingLoop() {
  if (!isTraining) return;

  trainEpoch(trainData, learningRate);
  epoch++;

  const loss = computeLoss(data);
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
  trainBtn.textContent = "Train";
  initNetwork(hiddenSize);
  epoch = 0;
  render();
  updateStatus();
});

trainBtn.addEventListener("click", () => {
  if (isTraining) {
    stopTraining();
  } else {
    startTraining();
  }
});

resetBtn.addEventListener("click", () => {
  stopTraining();
  data = generateCircleData();
  splitData(data, trainPercent);
  initNetwork(hiddenSize);
  epoch = 0;
  lossHistory = [];
  render();
  updateStatus();
});

data = generateCircleData();
splitData(data, trainPercent);
initNetwork(hiddenSize);
render();
updateStatus();