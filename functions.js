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
  infoDataset.textContent = 'Circle'; // change later when dataset buttons work
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