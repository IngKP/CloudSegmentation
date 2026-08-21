(() => {
  'use strict';
  const SIZE = 512;
  const PIXELS = SIZE * SIZE;
  const $ = (id) => document.getElementById(id);
  const state = { samples: [], selected: null, scores: null, reference: null, sar: null, selection: 0 };
  const canvases = ['sar', 'score', 'prediction', 'overlay', 'reference', 'error'].reduce((items, name) => {
    items[name] = $(`${name}-canvas`);
    return items;
  }, {});

  function setStatus(text, busy = false) {
    $('run-state').textContent = text;
    $('run-button').disabled = busy;
    $('run-button').textContent = busy ? 'Running…' : 'Run inference →';
  }

  function imagePixels(url, callback, onError) {
    const image = new Image();
    image.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = canvas.height = SIZE;
      const context = canvas.getContext('2d', { willReadFrequently: true });
      context.drawImage(image, 0, 0, SIZE, SIZE);
      callback(context.getImageData(0, 0, SIZE, SIZE).data);
    };
    image.onerror = () => onError?.(new Error(`Could not load ${url}`));
    image.src = url;
  }

  function drawPixels(canvas, pixels) {
    const context = canvas.getContext('2d');
    const output = context.createImageData(SIZE, SIZE);
    output.data.set(pixels);
    context.putImageData(output, 0, 0);
  }

  function drawSar(pixels) {
    drawPixels(canvases.sar, pixels);
    state.sar = pixels;
  }

  // Purple-orange-yellow deliberately avoids water-associated blue.
  function scoreColor(score) {
    const stops = [[16, 11, 32], [104, 39, 102], [216, 91, 63], [247, 217, 92]];
    const position = Math.min(score, 1) * (stops.length - 1);
    const lower = Math.min(Math.floor(position), stops.length - 2);
    const fraction = position - lower;
    return stops[lower].map((value, index) =>
      Math.round(value + (stops[lower + 1][index] - value) * fraction));
  }

  function render() {
    if (!state.scores || !state.reference) return;
    const threshold = Number($('threshold').value);
    $('threshold-value').value = threshold.toFixed(2);
    $('threshold-value').textContent = threshold.toFixed(2);
    $('prediction-caption').textContent = `threshold ≥ ${threshold.toFixed(2)}`;
    const scorePixels = new Uint8ClampedArray(PIXELS * 4);
    const predictionPixels = new Uint8ClampedArray(PIXELS * 4);
    const overlayPixels = new Uint8ClampedArray(PIXELS * 4);
    const referencePixels = new Uint8ClampedArray(PIXELS * 4);
    const errorPixels = new Uint8ClampedArray(PIXELS * 4);
    let tp = 0; let fp = 0; let fn = 0; let valid = 0;

    for (let index = 0; index < PIXELS; index++) {
      const score = state.scores[index];
      const reference = state.reference[index];
      const predictedWater = score >= threshold;
      const validReference = reference !== 255;
      if (validReference) {
        valid++;
        if (predictedWater && reference === 1) tp++;
        else if (predictedWater) fp++;
        else if (reference === 1) fn++;
      }
      const offset = index * 4;
      scorePixels.set([...scoreColor(score), 255], offset);
      predictionPixels.set([...(predictedWater ? [255, 155, 98] : [14, 18, 19]), 255], offset);
      const base = state.sar ? [state.sar[offset], state.sar[offset + 1], state.sar[offset + 2]] : [20, 20, 20];
      const overlay = predictedWater
        ? base.map((value, channel) => Math.round(value * 0.35 + [255, 155, 98][channel] * 0.65))
        : base;
      overlayPixels.set([...overlay, 255], offset);
      const checker = ((index % SIZE >> 3) + (Math.floor(index / SIZE) >> 3)) % 2;
      const referenceRgb = reference === 255
        ? (checker ? [105, 105, 105] : [45, 45, 45])
        : (reference === 1 ? [255, 155, 98] : [14, 18, 19]);
      referencePixels.set([...referenceRgb, 255], offset);
      let errorRgb;
      if (!validReference) errorRgb = checker ? [105, 105, 105] : [45, 45, 45];
      else if (predictedWater && reference === 1) errorRgb = [107, 229, 221];
      else if (predictedWater) errorRgb = [255, 155, 98];
      else if (reference === 1) errorRgb = [255, 100, 115];
      else errorRgb = [14, 18, 19];
      errorPixels.set([...errorRgb, 255], offset);
    }
    drawPixels(canvases.score, scorePixels);
    drawPixels(canvases.prediction, predictionPixels);
    drawPixels(canvases.overlay, overlayPixels);
    drawPixels(canvases.reference, referencePixels);
    drawPixels(canvases.error, errorPixels);
    $('iou').textContent = ((tp + fp + fn) ? tp / (tp + fp + fn) : 0).toFixed(3);
    $('precision').textContent = ((tp + fp) ? tp / (tp + fp) : 0).toFixed(3);
    $('recall').textContent = ((tp + fn) ? tp / (tp + fn) : 0).toFixed(3);
    $('valid-pixels').textContent = valid.toLocaleString();
  }

  async function run() {
    if (!state.selected) return;
    setStatus('Running inference…', true);
    $('error').classList.add('hidden');
    try {
      const response = await fetch('/predict', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sample_id: state.selected.id }),
      });
      if (!response.ok) throw new Error(`Inference failed (${response.status})`);
      const result = await response.json();
      if (!result.score_map_preview) throw new Error('The PNG8 score preview is unavailable.');
      await new Promise((resolve, reject) => imagePixels(`data:image/png;base64,${result.score_map_preview}`, (pixels) => {
        state.scores = new Float32Array(PIXELS);
        for (let index = 0; index < PIXELS; index++) state.scores[index] = pixels[index * 4] / 255;
        $('model-label').textContent = `${result.model_id} · ${result.model_variant}`;
        $('latency-label').textContent = `${Number(result.latency_ms).toFixed(1)} ms`;
        setStatus('Inference ready');
        render();
        resolve();
      }, reject));
    } catch (error) {
      $('error').textContent = error.message;
      $('error').classList.remove('hidden');
      setStatus('Inference unavailable');
    } finally {
      $('run-button').disabled = false;
    }
  }

  async function selectSample(sample) {
    const selection = ++state.selection;
    Object.assign(state, { selected: sample, scores: null, reference: null, sar: null });
    $('error').classList.add('hidden');
    document.querySelectorAll('.sample-card').forEach((element) =>
      element.classList.toggle('selected', element.dataset.id === sample.id));
    $('workspace').classList.remove('hidden');
    $('scene-label').textContent = `${sample.id} · ${sample.country} · ${sample.split}`;
    setStatus('Loading scene…', true);
    try {
      await new Promise((resolve, reject) => imagePixels(`/samples/${encodeURIComponent(sample.id)}/sar.png`, (pixels) => {
        if (selection === state.selection) drawSar(pixels);
        resolve();
      }, reject));
      await new Promise((resolve, reject) => imagePixels(`/samples/${encodeURIComponent(sample.id)}/reference.png`, (pixels) => {
        if (selection === state.selection) {
          state.reference = new Uint8Array(PIXELS);
          for (let index = 0; index < PIXELS; index++) state.reference[index] = pixels[index * 4];
        }
        resolve();
      }, reject));
      if (selection === state.selection) setStatus('Ready to run');
    } catch (error) {
      if (selection === state.selection) {
        $('error').textContent = error.message;
        $('error').classList.remove('hidden');
        setStatus('Scene unavailable');
      }
    } finally {
      if (selection === state.selection) $('run-button').disabled = false;
    }
  }

  async function init() {
    try {
      const response = await fetch('/samples');
      if (!response.ok) throw new Error(`Could not load samples (${response.status})`);
      state.samples = await response.json();
      $('sample-count').textContent = `${state.samples.length} available`;
      $('gallery').replaceChildren(...state.samples.map((sample) => {
        const card = document.createElement('button');
        card.className = 'sample-card';
        card.type = 'button';
        card.dataset.id = sample.id;
        card.innerHTML = `<span><h3>${sample.id.replaceAll('_', ' ')}</h3><p>${sample.country} · ${sample.split} · reference available</p></span><span class="sample-arrow" aria-hidden="true">↗</span>`;
        card.addEventListener('click', () => selectSample(sample));
        return card;
      }));
      if (state.samples[0]) selectSample(state.samples[0]);
    } catch (error) {
      $('gallery').textContent = error.message;
    }
  }

  $('run-button').addEventListener('click', run);
  $('threshold').addEventListener('input', render);
  init();
})();
