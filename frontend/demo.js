// Live demo logic — wires up the file picker, posts to the FastAPI service
// running on Cloud Run, and renders the prediction result.
//
// Depends on `API_BASE_URL` from config.js.

(function () {
  const fileInput   = document.getElementById('audioFile');
  const filePicker  = document.getElementById('filePicker');
  const fileHint    = document.getElementById('fileHint');
  const classifyBtn = document.getElementById('classifyBtn');
  const resetBtn    = document.getElementById('resetBtn');
  const resultEmpty   = document.getElementById('resultEmpty');
  const resultContent = document.getElementById('resultContent');

  // ── Helpers ─────────────────────────────────────────
  const LABEL_INFO = {
    high_energy: {
      title: 'High Energy — Belt',
      blurb: 'Loud, projected, harmonic-rich vocal technique typical of belt singing.',
      barClass: 'belt',
      badgeClass: 'badge-green',
    },
    low_energy: {
      title: 'Low Energy — Breathy',
      blurb: 'Soft, air-rich vocal technique with reduced harmonic content.',
      barClass: 'breathy',
      badgeClass: 'badge-lavender',
    },
  };

  function fmt(n, digits = 2) {
    if (n === null || n === undefined || Number.isNaN(n)) return '—';
    return Number(n).toFixed(digits);
  }

  function showResult(html) {
    resultEmpty.style.display = 'none';
    resultContent.innerHTML = html;
    resultContent.classList.add('visible');
  }

  function showError(message) {
    showResult(`
      <div class="demo-error">
        <span class="demo-error-icon">⚠️</span>
        <h3>Inference failed</h3>
        <p>${message}</p>
      </div>
    `);
  }

  function showLoading() {
    resultEmpty.style.display = 'none';
    resultContent.classList.add('visible');
    resultContent.innerHTML = `
      <div class="demo-loading">
        <div class="spinner"></div>
        <p>Extracting features and running inference…</p>
      </div>
    `;
  }

  function renderResult(data) {
    const info = LABEL_INFO[data.label] || LABEL_INFO.high_energy;
    const probHigh = data.probabilities.high_energy ?? 0;
    const probLow  = data.probabilities.low_energy  ?? 0;
    const f = data.features || {};

    showResult(`
      <div class="result-header">
        <span class="badge ${info.badgeClass}">${data.label}</span>
        <h3>${info.title}</h3>
        <p class="result-blurb">${info.blurb}</p>
        <div class="confidence-block">
          <span class="confidence-label">Confidence</span>
          <span class="confidence-value">${(data.confidence * 100).toFixed(1)}%</span>
        </div>
      </div>

      <div class="prob-bar-group prob-bar-group--light">
        <div class="prob-bar-row">
          <div class="prob-bar-label">
            <span>High Energy (Belt)</span>
            <span>${(probHigh * 100).toFixed(1)}%</span>
          </div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill belt" style="width: ${probHigh * 100}%;"></div>
          </div>
        </div>
        <div class="prob-bar-row">
          <div class="prob-bar-label">
            <span>Low Energy (Breathy)</span>
            <span>${(probLow * 100).toFixed(1)}%</span>
          </div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill breathy" style="width: ${probLow * 100}%;"></div>
          </div>
        </div>
      </div>

      <details class="features-details">
        <summary>Show extracted features</summary>
        <table class="features-table">
          <tbody>
            <tr><td>f0 (Hz)</td><td>${fmt(f.f0)}</td></tr>
            <tr><td>Spectral centroid (Hz)</td><td>${fmt(f.spectral_centroid)}</td></tr>
            <tr><td>Spectral rolloff (Hz)</td><td>${fmt(f.spectral_rolloff)}</td></tr>
            <tr><td>HNR (dB)</td><td>${fmt(f.hnr)}</td></tr>
          </tbody>
        </table>
      </details>
    `);
  }

  // ── File picker state ───────────────────────────────
  fileInput.addEventListener('change', function () {
    const file = fileInput.files[0];
    if (file) {
      filePicker.classList.add('has-file');
      fileHint.textContent = file.name;
      classifyBtn.disabled = false;
    } else {
      filePicker.classList.remove('has-file');
      fileHint.textContent = 'No file selected';
      classifyBtn.disabled = true;
    }
  });

  resetBtn.addEventListener('click', function () {
    fileInput.value = '';
    filePicker.classList.remove('has-file');
    fileHint.textContent = 'No file selected';
    classifyBtn.disabled = true;
    resultContent.classList.remove('visible');
    resultContent.innerHTML = '';
    resultEmpty.style.display = 'block';
  });

  // ── Submit ───────────────────────────────────────────
  classifyBtn.addEventListener('click', async function () {
    const file = fileInput.files[0];
    if (!file) return;

    classifyBtn.disabled = true;
    showLoading();

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
          const err = await res.json();
          if (err && err.detail) detail = err.detail;
        } catch (_) { /* non-JSON error body */ }
        showError(detail);
        return;
      }

      const data = await res.json();
      renderResult(data);
    } catch (err) {
      showError('Network error — could not reach the inference service.');
    } finally {
      classifyBtn.disabled = !fileInput.files[0];
    }
  });
})();
