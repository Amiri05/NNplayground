const lrSlider = document.getElementById("lr");
const hiddenSlider = document.getElementById("hidden");
const trainSplitSlider = document.getElementById("trainSplit");
const noiseSlider = document.getElementById("noise");
const showTrainCheckbox = document.getElementById("showTrainData");

const trainBtn = document.getElementById("trainBtn");
const resetBtn = document.getElementById("resetBtn");

const datasetSelect = document.getElementById("datasetSelect");
const datasetCards = document.querySelectorAll(".dataset-card");

const lrVal = document.getElementById("lrVal");
const epochVal = document.getElementById("epochVal");
const pauseIcon = document.getElementById("pauseIcon");
const dot = document.getElementById("dot");

const trainSplitValue = document.getElementById("trainSplitValue");
const noiseValue = document.getElementById("noiseValue");

const infoDataset = document.getElementById("infoDataset");
const infoHidden = document.getElementById("infoHidden");
const infoTrain = document.getElementById("infoTrain");
const infoNoise = document.getElementById("infoNoise");
const infoTrainLoss = document.getElementById("infoTrainLoss");
const infoTestLoss = document.getElementById("infoTestLoss");

let datasetType = datasetSelect.value;
let learningRate = parseFloat(lrSlider.value);
let hiddenSize = parseInt(hiddenSlider.value, 10);
let trainPercent = parseInt(trainSplitSlider.value, 10);
let noiseLevel = parseFloat(noiseSlider.value);
let showTrainData = showTrainCheckbox.checked;

let isTraining = false;
let epoch = 0;
let timer = null;

const PLAY = '<polygon points="2.5,1.5 10,6 2.5,10.5"/>';
const PAUSE = '<rect x="1.5" y="1" width="3.5" height="10" rx="1"/><rect x="7" y="1" width="3.5" height="10" rx="1"/>';

function capitalize(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function syncPlaybackUI() {
  pauseIcon.innerHTML = isTraining ? PAUSE : PLAY;
  dot.className = isTraining ? "dot" : "dot off";
}

function updateDatasetButtons(selectedValue) {
  datasetCards.forEach((card) => {
    card.classList.toggle("active", card.dataset.dataset === selectedValue);
  });
}

function updateInfo() {
  lrVal.textContent = learningRate.toFixed(3);
  epochVal.textContent = String(epoch).padStart(4, "0");

  trainSplitValue.textContent = `${trainPercent}%`;
  noiseValue.textContent = noiseLevel.toFixed(2);

  infoDataset.textContent = capitalize(datasetType);
  infoHidden.textContent = hiddenSize;
  infoTrain.textContent = `${trainPercent}%`;
  infoNoise.textContent = noiseLevel.toFixed(2);
  
  // placeholders until training/loss math is wired in
  infoTrainLoss.textContent = "0.00000";
  infoTestLoss.textContent = "0.00000";
}

datasetCards.forEach((card) => {
  card.addEventListener("click", () => {
    datasetType = card.dataset.dataset;
    datasetSelect.value = datasetType;
    updateDatasetButtons(datasetType);
    updateInfo();
  });
});

lrSlider.addEventListener("input", () => {
  learningRate = parseFloat(lrSlider.value);
  updateInfo();
});

hiddenSlider.addEventListener("input", () => {
  hiddenSize = parseInt(hiddenSlider.value, 10);
  updateInfo();
});

trainSplitSlider.addEventListener("input", () => {
  trainPercent = parseInt(trainSplitSlider.value, 10);
  updateInfo();
});

noiseSlider.addEventListener("input", () => {
  noiseLevel = parseFloat(noiseSlider.value);
  updateInfo();
});

showTrainCheckbox.addEventListener("change", () => {
  showTrainData = showTrainCheckbox.checked;
  updateInfo();
});

trainBtn.addEventListener("click", () => {
  isTraining = !isTraining;
  syncPlaybackUI();

  if (isTraining) {
    timer = setInterval(() => {
      epoch += 1;
      updateInfo();
    }, 60);
  } else {
    clearInterval(timer);
    timer = null;
  }
});

resetBtn.addEventListener("click", () => {
  isTraining = false;
  clearInterval(timer);
  timer = null;
  epoch = 0;
  syncPlaybackUI();
  updateInfo();
});

syncPlaybackUI();
updateDatasetButtons(datasetType);
updateInfo();