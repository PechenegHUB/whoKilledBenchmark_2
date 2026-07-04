/*
  Browser-side mask editor worker.

  The worker owns the visible edit canvas through OffscreenCanvas, so pointer-heavy
  drawing and PNG export do not block the Streamlit iframe main thread.
*/

let canvas = null;
let ctx = null;
let width = 0;
let height = 0;
let addColor = '#0060FF';
let removeColor = '#FF2D2D';
let strokes = [];
let activeStroke = null;
let lastExportToken = 0;

function postStatus() {
  self.postMessage({type: 'status', strokes: strokes.length});
}

function postError(message) {
  self.postMessage({type: 'error', message: String(message || 'Неизвестная ошибка worker')});
}

function setupContext() {
  if (!canvas) return;
  width = Number(width || canvas.width || 640);
  height = Number(height || canvas.height || 480);
  canvas.width = width;
  canvas.height = height;
  ctx = canvas.getContext('2d', {alpha: true, desynchronized: true});
  if (!ctx) {
    throw new Error('Не удалось получить 2D context в worker');
  }
  ctx.clearRect(0, 0, width, height);
}

function clearCanvas() {
  if (!ctx) return;
  ctx.clearRect(0, 0, width, height);
}

function drawSegment(stroke, p0, p1) {
  if (!ctx || !stroke || !p0 || !p1) return;
  ctx.save();
  ctx.globalCompositeOperation = 'source-over';
  ctx.strokeStyle = stroke.mode === 'add' ? addColor : removeColor;
  ctx.lineWidth = Math.max(1, Number(stroke.size || 1));
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.beginPath();
  ctx.moveTo(p0.x, p0.y);
  ctx.lineTo(p1.x, p1.y);
  ctx.stroke();
  ctx.restore();
}

function drawDot(stroke, p) {
  if (!ctx || !stroke || !p) return;
  const r = Math.max(0.5, Number(stroke.size || 1) / 2);
  ctx.save();
  ctx.globalCompositeOperation = 'source-over';
  ctx.fillStyle = stroke.mode === 'add' ? addColor : removeColor;
  ctx.beginPath();
  ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function redrawAll() {
  clearCanvas();
  for (const stroke of strokes) {
    if (!stroke.points || stroke.points.length === 0) continue;
    if (stroke.points.length === 1) {
      drawDot(stroke, stroke.points[0]);
      continue;
    }
    for (let i = 1; i < stroke.points.length; i += 1) {
      drawSegment(stroke, stroke.points[i - 1], stroke.points[i]);
    }
  }
}

function resetState(nextWidth, nextHeight, nextAddColor, nextRemoveColor) {
  width = Number(nextWidth || width || 640);
  height = Number(nextHeight || height || 480);
  addColor = nextAddColor || addColor;
  removeColor = nextRemoveColor || removeColor;
  strokes = [];
  activeStroke = null;
  setupContext();
  postStatus();
}

async function canvasToDataUrl() {
  if (!canvas) throw new Error('Canvas ещё не инициализирован');

  if (typeof canvas.convertToBlob === 'function') {
    const blob = await canvas.convertToBlob({type: 'image/png'});
    const buffer = await blob.arrayBuffer();
    const bytes = new Uint8Array(buffer);
    const chunkSize = 0x8000;
    let binary = '';
    for (let i = 0; i < bytes.length; i += chunkSize) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
    }
    return `data:image/png;base64,${btoa(binary)}`;
  }

  throw new Error('OffscreenCanvas.convertToBlob недоступен в этом браузере');
}

function beginStroke(payload) {
  const point = payload.point;
  activeStroke = {
    mode: payload.mode === 'remove' ? 'remove' : 'add',
    size: Math.max(1, Number(payload.size || 25)),
    points: [point],
  };
  drawDot(activeStroke, point);
}

function addPoint(payload) {
  if (!activeStroke) return;
  const point = payload.point;
  const prev = activeStroke.points[activeStroke.points.length - 1];
  activeStroke.points.push(point);
  drawSegment(activeStroke, prev, point);
}

function endStroke(payload) {
  if (!activeStroke) return;
  if (payload && payload.point) {
    const prev = activeStroke.points[activeStroke.points.length - 1];
    const point = payload.point;
    const dx = point.x - prev.x;
    const dy = point.y - prev.y;
    if ((dx * dx + dy * dy) > 0.25) {
      activeStroke.points.push(point);
      drawSegment(activeStroke, prev, point);
    }
  }
  strokes.push(activeStroke);
  activeStroke = null;
  postStatus();
}

self.onmessage = async (event) => {
  const msg = event.data || {};
  try {
    switch (msg.type) {
      case 'init':
        canvas = msg.canvas || new OffscreenCanvas(Number(msg.width || 640), Number(msg.height || 480));
        resetState(msg.width, msg.height, msg.addColor, msg.removeColor);
        self.postMessage({type: 'ready'});
        break;

      case 'reset':
        resetState(msg.width, msg.height, msg.addColor, msg.removeColor);
        break;

      case 'setColors':
        addColor = msg.addColor || addColor;
        removeColor = msg.removeColor || removeColor;
        redrawAll();
        break;

      case 'beginStroke':
        beginStroke(msg);
        break;

      case 'addPoint':
        addPoint(msg);
        break;

      case 'endStroke':
        endStroke(msg);
        break;

      case 'undo':
        if (activeStroke) activeStroke = null;
        strokes.pop();
        redrawAll();
        postStatus();
        break;

      case 'clear':
        activeStroke = null;
        strokes = [];
        clearCanvas();
        postStatus();
        break;

      case 'export': {
        const exportToken = ++lastExportToken;
        const editImage = await canvasToDataUrl();
        if (exportToken !== lastExportToken) return;
        self.postMessage({
          type: 'exported',
          action: msg.action || 'apply',
          edit_image: editImage,
          width,
          height,
          strokes: strokes.length,
          submitted_at: Date.now(),
        });
        break;
      }

      default:
        break;
    }
  } catch (error) {
    postError(error && error.message ? error.message : error);
  }
};
