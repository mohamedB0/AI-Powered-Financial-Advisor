/**
 * Voice Assistant — Frontend Controller
 * ──────────────────────────────────────
 * Flow:
 *  1. User clicks orb → MediaRecorder starts capturing mic audio
 *  2. User clicks again (or silence detected) → recording stops
 *  3. Audio blob sent over WebSocket to backend
 *  4. Backend replies with:
 *     • JSON { type:"transcript" }  → show user text
 *     • JSON { type:"token" }       → stream AI text into panel
 *     • JSON { type:"status" }      → update status bar
 *     • JSON { type:"error" }       → show error toast
 *     • JSON { type:"done" }        → finalise & add to history
 *     • raw bytes                   → queue audio chunk for playback
 */

// Guard against double-include (fixes "already declared" SyntaxError)
if (typeof window._ariaLoaded !== 'undefined') {
  throw new Error('script.js loaded twice — check index.html');
}
window._ariaLoaded = true;

// ── Config ────────────────────────────────────────────────────────────────────
const WS_URL = (location.protocol === 'https:' ? 'wss' : 'ws')
             + `://${location.host}/ws/voice`;

const SILENCE_THRESHOLD_MS = 3000;   // auto-stop after 3 s of silence
const MIN_RECORD_MS        = 500;    // don't stop before 500 ms

// ── DOM refs ──────────────────────────────────────────────────────────────────
const orb            = document.getElementById('orb');
const micIcon        = document.getElementById('micIcon');
const stopIcon       = document.getElementById('stopIcon');
const orbHint        = document.getElementById('orbHint');
const statusDot      = document.getElementById('statusDot');
const statusText     = document.getElementById('statusText');
const transcriptPanel = document.getElementById('transcriptPanel');
const replyPanel     = document.getElementById('replyPanel');
const historyBody    = document.getElementById('historyBody');
const clearBtn       = document.getElementById('clearBtn');

// Error toast (create dynamically)
const toast = document.createElement('div');
toast.id = 'errorToast';
document.body.appendChild(toast);

// ── State ─────────────────────────────────────────────────────────────────────
let ws            = null;
let mediaRecorder = null;
let audioChunks   = [];
let isRecording   = false;
let audioCtx      = null;
let audioQueue    = [];     // array of ArrayBuffer chunks from TTS
let isPlaying     = false;  // single declaration
let currentReply  = '';
let currentTranscript = '';

// Level bar (audio visualisation)
const levelWrap   = document.createElement('div');
levelWrap.className = 'level-bar-wrap';
const levelBar    = document.createElement('div');
levelBar.className = 'level-bar';
levelWrap.appendChild(levelBar);
document.querySelector('.orb-section').insertBefore(
  levelWrap,
  document.querySelector('.orb-hint')
);

let analyser      = null;
let levelRafId    = null;

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connectWS() {
  setStatus('Connecting…', '');
  ws = new WebSocket(WS_URL);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    setStatus('Ready', 'connected');
    startPing();
  };

  ws.onclose = () => {
    setStatus('Disconnected', '');
    setBodyState('');
    setTimeout(connectWS, 3000);  // auto-reconnect
  };

  ws.onerror = (e) => {
    console.error('WS error', e);
    showError('Connection error — retrying…');
  };

  ws.onmessage = (ev) => {
    if (typeof ev.data === 'string') {
      handleTextMessage(JSON.parse(ev.data));
    } else {
      // Binary → audio chunk
      audioQueue.push(ev.data);
      drainAudioQueue();
    }
  };
}

// Keep-alive ping
let pingInterval = null;
function startPing() {
  clearInterval(pingInterval);
  pingInterval = setInterval(() => {
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'ping' }));
    }
  }, 20000);
}

// ── Message handling ──────────────────────────────────────────────────────────
function handleTextMessage(msg) {
  switch (msg.type) {
    case 'pong':
      break;

    case 'status':
      setStatus(msg.message, msg.message.toLowerCase().includes('think') ? 'thinking' : 'connected');
      break;

    case 'transcript':
      currentTranscript = msg.text;
      transcriptPanel.innerHTML = '';
      transcriptPanel.textContent = msg.text;
      setStatus('Thinking…', 'thinking');
      setBodyState('thinking');
      break;

    case 'token':
      currentReply += msg.text;
      replyPanel.textContent = currentReply;
      setStatus('Speaking…', 'speaking');
      setBodyState('speaking');
      break;

    case 'done':
      addToHistory(currentTranscript, currentReply);
      currentReply = '';
      currentTranscript = '';
      nextStartTime = 0;
      setStatus('Ready', 'connected');
      setBodyState('');
      break;

    case 'error':
      showError(msg.message);
      setStatus('Error', 'connected');
      setBodyState('');
      break;
  }
}

// ── Audio playback queue ──────────────────────────────────────────────────────
let audioBufferQueue = [];
let nextStartTime = 0;

async function drainAudioQueue() {
    if (audioQueue.length === 0) return;

    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        nextStartTime = audioCtx.currentTime;
    }

    // Resume if suspended
    if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
    }

    while (audioQueue.length > 0) {
        const chunk = audioQueue.shift();
        try {
            const buffer = await audioCtx.decodeAudioData(chunk.slice(0));
            const source = audioCtx.createBufferSource();
            source.buffer = buffer;
            source.connect(audioCtx.destination);

            const startAt = Math.max(nextStartTime, audioCtx.currentTime);
            source.start(startAt);
            nextStartTime = startAt + buffer.duration;

        } catch (e) {
            console.warn('Audio decode error:', e);
        }
    }
}

function playBuffer(buffer) {
  return new Promise((resolve) => {
    const source = audioCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(audioCtx.destination);
    source.onended = resolve;
    source.start();
  });
}

// ── Recording ─────────────────────────────────────────────────────────────────
orb.addEventListener('click', toggleRecording);

async function toggleRecording() {
  if (isRecording) {
    stopRecording();
  } else {
    await startRecording();
  }
}

async function startRecording() {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    showError('Not connected to server');
    return;
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    showError('Microphone access denied');
    return;
  }

  audioChunks = [];
  isRecording = true;
  currentReply = '';
  replyPanel.innerHTML = '<span class="placeholder">Waiting for response…</span>';
  transcriptPanel.innerHTML = '<span class="placeholder">Listening…</span>';

  setBodyState('recording');
  setStatus('Recording…', 'recording');
  micIcon.classList.add('hidden');
  stopIcon.classList.remove('hidden');
  orbHint.textContent = 'Click to send';

  // Audio visualisation
  setupLevelMeter(stream);

  // Choose a mime type the browser supports
  const mimeType = getSupportedMimeType();
  mediaRecorder = new MediaRecorder(stream, { mimeType });
  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) audioChunks.push(e.data);
  };
  mediaRecorder.onstop = () => {
    stream.getTracks().forEach(t => t.stop());
    stopLevelMeter();
    sendAudio();
  };

  mediaRecorder.start(200);  // collect every 200 ms
}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;
  mediaRecorder?.stop();

  micIcon.classList.remove('hidden');
  stopIcon.classList.add('hidden');
  orbHint.textContent = 'Click to speak';
  levelBar.style.width = '0%';
}

function sendAudio() {
  const blob = new Blob(audioChunks, { type: getSupportedMimeType() });
  blob.arrayBuffer().then(buf => {
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(buf);
    }
  });
}

function getSupportedMimeType() {
  const types = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/mp4',
  ];
  return types.find(t => MediaRecorder.isTypeSupported(t)) || 'audio/webm';
}

// ── Level meter ───────────────────────────────────────────────────────────────
function setupLevelMeter(stream) {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioCtx.createMediaStreamSource(stream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 512;
  source.connect(analyser);

  const data = new Uint8Array(analyser.frequencyBinCount);

  function tick() {
    analyser.getByteFrequencyData(data);
    const avg = data.reduce((s, v) => s + v, 0) / data.length;
    levelBar.style.width = Math.min(100, avg * 2.5) + '%';
    levelRafId = requestAnimationFrame(tick);
  }
  tick();
}

function stopLevelMeter() {
  cancelAnimationFrame(levelRafId);
  analyser = null;
  levelBar.style.width = '0%';
}

// ── UI helpers ────────────────────────────────────────────────────────────────
function setStatus(text, cls) {
  statusText.textContent = text;
  statusDot.className = 'status-dot' + (cls ? ' ' + cls : '');
}

function setBodyState(state) {
  document.body.className = state;
}

function showError(msg) {
  toast.textContent = '⚠ ' + msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 4000);
}

function addToHistory(userText, aiText) {
  // Remove placeholder
  const placeholder = historyBody.querySelector('.placeholder');
  if (placeholder) placeholder.remove();

  const entry = document.createElement('div');
  entry.className = 'history-entry';
  entry.innerHTML = `
    <div class="history-role user-role">YOU</div>
    <div class="history-msg user-msg">${escapeHtml(userText)}</div>
    <div class="history-role ai-role" style="margin-top:.4rem">ARIA</div>
    <div class="history-msg ai-msg">${escapeHtml(aiText)}</div>
  `;
  historyBody.appendChild(entry);
  historyBody.scrollTop = historyBody.scrollHeight;
}

function escapeHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

clearBtn.addEventListener('click', () => {
  historyBody.innerHTML = '<p class="placeholder">Conversation will appear here.</p>';
});

// ── Boot ──────────────────────────────────────────────────────────────────────
connectWS();