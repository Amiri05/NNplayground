// Dataset Generator

// Circle Dataset
function generateCircleData(n = 400) {
  const points = [];

  for (let i = 0; i < n; i++) {
    const x = Math.random() * 2 - 1;
    const y = Math.random() * 2 - 1;
    const r = Math.sqrt(x * x + y * y);

    const label = r < 0.5 ? 1 : 0;

    points.push({ x, y, label });
  }

  return points;
}

const canvas = document.getElementById("plot");
const ctx = canvas.getContext("2d");

let data = generateCircleData();

function toCanvasX(x) {
  return ((x + 1) / 2) * canvas.width;
}

function toCanvasY(y) {
  return canvas.height - ((y + 1) / 2) * canvas.height;
}

function drawPoints(points) {
  for (const p of points) {
    ctx.beginPath();
    ctx.arc(toCanvasX(p.x), toCanvasY(p.y), 3, 0, Math.PI * 2);
    ctx.fillStyle = p.label === 1 ? "orange" : "steelblue";
    ctx.fill();
  }
}

drawPoints(data);

/* Neural Network 
    - Weights
    - Forward Pass
    - Backprop
    - Training Step
*/

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

  // Save old W2 before updating
  const oldW2 = [...W2];

  // Update output layer
  for (let i = 0; i < W2.length; i++) {
    W2[i] -= lr * dOut * hidden[i];
  }
  b2 -= lr * dOut;

  // Update hidden layer using OLD W2
  for (let i = 0; i < W1.length; i++) {
    const dHidden = dOut * oldW2[i] * dtanh(hidden[i]);
    W1[i][0] -= lr * dHidden * x1;
    W1[i][1] -= lr * dHidden * x2;
    b1[i] -= lr * dHidden;
  }
}

function trainEpoch(points, lr) {
  for (const p of points) {
    trainStep(p, lr);
  }
}

document.getElementById("trainBtn").addEventListener("click", () => {
  const lr = parseFloat(document.getElementById("lr").value);

  for (let i = 0; i < 200; i++) {
    trainEpoch(data, lr);
  }

  drawDecisionBoundary();
  drawPoints(data);
});

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
  drawPoints(data);
}

document.getElementById("resetBtn").addEventListener("click", () => {
  const hiddenSize = parseInt(document.getElementById("hidden").value, 10);
  data = generateCircleData();
  initNetwork(hiddenSize);
  render();
});

initNetwork(4);
render();

/* Renderer
    - Background Decision Boundary 
    - Training Points
    - Network Diagram
*/

// UI Controller