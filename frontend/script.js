// --- Tabs ---
const tabLive = document.getElementById('tab-live');
const tabHist = document.getElementById('tab-hist');
const panelLive = document.getElementById('panel-live');
const panelHist = document.getElementById('panel-hist');

tabLive.addEventListener('click', () => {
  tabLive.classList.add('active'); tabHist.classList.remove('active');
  panelLive.classList.remove('hidden'); panelHist.classList.add('hidden');
});
tabHist.addEventListener('click', () => {
  tabHist.classList.add('active'); tabLive.classList.remove('active');
  panelHist.classList.remove('hidden'); panelLive.classList.add('hidden');
});

// --- Common outputs ---
const alertBox = document.getElementById('alertBox');
const statsEl = document.getElementById('stats');
const imgBefore = document.getElementById('img_before');
const imgAfter  = document.getElementById('img_after');

function renderResult(data) {
  // Images
  imgBefore.src = data.images.before;
  imgAfter.src  = data.images.after;

  // Stats
  statsEl.textContent = `• Total pixels: ${data.stats.pixels}\n• Loss percentage: ${data.stats.loss_pct}%`;

  // Alert
  alertBox.classList.remove('hidden', 'red', 'green');
  if (data.alert) {
    alertBox.classList.add('red');
    alertBox.innerHTML = `<span>⚠️</span> ${data.alert_reason}`;
  } else {
    alertBox.classList.add('green');
    alertBox.innerHTML = `<span>✅</span> ${data.alert_reason}`;
  }
}

// --- Live ---
const runLiveBtn = document.getElementById('run-live');
const statusLive = document.getElementById('status-live');

runLiveBtn.addEventListener('click', async () => {
  const bboxStr = document.getElementById('live-bbox').value.trim();
  const baselineStr = document.getElementById('live-baseline').value.trim();

  const bbox = bboxStr.split(',').map(Number);
  let payload = { bbox };

  if (baselineStr) {
    const [bs, be] = baselineStr.split(',').map(s => s.trim());
    payload.baseline = { start: bs, end: be };
  }

  runLiveBtn.disabled = true;
  statusLive.textContent = 'status: fetching live...';

  try {
    const res = await fetch('/api/gee/live', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || 'unknown error');
    renderResult(data);
    statusLive.textContent = 'status: done';
  } catch (e) {
    console.error(e);
    statusLive.textContent = 'status: error';
    alert('Live failed: ' + e.message);
  } finally {
    runLiveBtn.disabled = false;
  }
});

// --- Historical ---
const runHistBtn = document.getElementById('run-hist');
const statusHist = document.getElementById('status-hist');

runHistBtn.addEventListener('click', async () => {
  const bboxStr = document.getElementById('hist-bbox').value.trim();
  const beforeStr = document.getElementById('hist-before').value.trim();
  const afterStr  = document.getElementById('hist-after').value.trim();

  const bbox = bboxStr.split(',').map(Number);
  const [bs, be] = beforeStr.split(',').map(s => s.trim());
  const [as, ae] = afterStr.split(',').map(s => s.trim());

  runHistBtn.disabled = true;
  statusHist.textContent = 'status: fetching historical...';

  try {
    const res = await fetch('/api/gee/change', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        bbox,
        before: { start: bs, end: be },
        after:  { start: as, end: ae }
      })
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || 'unknown error');
    renderResult(data);
    statusHist.textContent = 'status: done';
  } catch (e) {
    console.error(e);
    statusHist.textContent = 'status: error';
    alert('Historical failed: ' + e.message);
  } finally {
    runHistBtn.disabled = false;
  }
});

// --- Monitoring ---
const startMonitorBtn = document.getElementById('start-monitor');
const stopMonitorBtn  = document.getElementById('stop-monitor');
const statusMonitor   = document.getElementById('status-monitor');

startMonitorBtn.addEventListener('click', async () => {
  const bboxStr = document.getElementById('monitor-bbox').value.trim();
  const interval = parseInt(document.getElementById('monitor-interval').value.trim(), 10);
  const threshold = parseFloat(document.getElementById('monitor-threshold').value.trim());

  const bbox = bboxStr.split(',').map(Number);

  startMonitorBtn.disabled = true;
  statusMonitor.textContent = 'status: starting monitoring...';

  try {
    const res = await fetch('/api/monitor/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ bbox, interval, threshold })
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || 'unknown error');
    statusMonitor.textContent = data.message || 'Monitoring started';
  } catch (e) {
    console.error(e);
    statusMonitor.textContent = 'status: error';
    alert('Start monitoring failed: ' + e.message);
  } finally {
    startMonitorBtn.disabled = false;
  }
});

stopMonitorBtn.addEventListener('click', async () => {
  stopMonitorBtn.disabled = true;
  statusMonitor.textContent = 'status: stopping monitoring...';

  try {
    const res = await fetch('/api/monitor/stop', {
      method: 'POST'
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || 'unknown error');
    statusMonitor.textContent = data.message || 'Monitoring stopped';
  } catch (e) {
    console.error(e);
    statusMonitor.textContent = 'status: error';
    alert('Stop monitoring failed: ' + e.message);
  } finally {
    stopMonitorBtn.disabled = false;
  }
});
