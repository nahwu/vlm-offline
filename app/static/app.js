const form = document.getElementById("queryForm");
const fileInput = document.getElementById("fileInput");
const queryInput = document.getElementById("queryInput");
const maxFrames = document.getElementById("maxFrames");
const videoPipeline = document.getElementById("videoPipeline");
const coordMode = document.getElementById("coordMode");
const submitBtn = document.getElementById("submitBtn");
const statusBox = document.getElementById("status");
const errorBox = document.getElementById("error");
const fileList = document.getElementById("fileList");
const results = document.getElementById("results");
const historyList = document.getElementById("historyList");
const metricsBox = document.getElementById("metricsBox");
const refreshMetricsBtn = document.getElementById("refreshMetricsBtn");

const requestHistory = [];

function formatBytes(bytes) {
  if (!bytes) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  let size = bytes;
  while (size >= 1024 && i < units.length - 1) {
    size /= 1024;
    i += 1;
  }
  return `${size.toFixed(1)} ${units[i]}`;
}

function renderFileList(files) {
  fileList.innerHTML = "";
  if (!files.length) {
    fileList.innerHTML = "<li>No files selected.</li>";
    return;
  }

  files.forEach((file) => {
    const li = document.createElement("li");
    li.innerHTML = `<strong>${file.name}</strong><br><small>${file.type || "unknown"} • ${formatBytes(file.size)}</small>`;
    fileList.appendChild(li);
  });
}

function renderHistory() {
  historyList.innerHTML = "";
  if (!requestHistory.length) {
    historyList.innerHTML = "<li>No requests yet.</li>";
    return;
  }

  requestHistory.slice().reverse().forEach((item) => {
    const li = document.createElement("li");
    li.innerHTML = `<strong>${item.time}</strong><br><small>${item.mode} • ${item.pipeline} • ${item.fileCount} file(s)</small>`;
    historyList.appendChild(li);
  });
}

function pushHistory(mode, pipeline, fileCount) {
  requestHistory.push({
    time: new Date().toLocaleTimeString(),
    mode,
    pipeline,
    fileCount,
  });
  if (requestHistory.length > 20) {
    requestHistory.shift();
  }
  renderHistory();
}

function renderCards(payload, mode) {
  results.innerHTML = "";

  if (mode === "server_batch") {
    const ok = payload.results || [];
    const errors = payload.errors || [];

    ok.forEach((item) => {
      const card = document.createElement("article");
      card.className = "result-card";
      card.innerHTML = `<h3>${item.filename}</h3><p>${item.answer || "(empty)"}</p>`;
      results.appendChild(card);
    });

    errors.forEach((item) => {
      const card = document.createElement("article");
      card.className = "result-card error-card";
      card.innerHTML = `<h3>${item.filename}</h3><p>${item.error || "Unknown error"}</p>`;
      results.appendChild(card);
    });

    if (!ok.length && !errors.length) {
      results.innerHTML = '<article class="result-card"><p>No results returned.</p></article>';
    }
    return;
  }

  (payload.results || []).forEach((item) => {
    const card = document.createElement("article");
    card.className = "result-card";
    card.innerHTML = `<h3>${item.filename}</h3><p>${item.answer || "(empty)"}</p>`;
    results.appendChild(card);
  });

  if (!payload.results || !payload.results.length) {
    results.innerHTML = '<article class="result-card"><p>No results returned.</p></article>';
  }
}

async function sendSingle(file, query, frames, pipeline) {
  const data = new FormData();
  data.append("file", file);
  data.append("query", query);
  data.append("max_video_frames", String(frames));
  data.append("video_pipeline", pipeline);

  const res = await fetch("/v1/query", {
    method: "POST",
    body: data,
  });

  const payload = await res.json();
  if (!res.ok) {
    throw new Error(payload.detail || "Request failed");
  }

  return {
    filename: payload.filename,
    answer: payload.answer,
    requestId: res.headers.get("X-Request-ID") || "n/a",
  };
}

async function sendServerBatch(files, query, frames, pipeline) {
  const data = new FormData();
  data.append("query", query);
  data.append("max_video_frames", String(frames));
  data.append("video_pipeline", pipeline);

  files.forEach((file) => data.append("files", file));

  const res = await fetch("/v1/query/batch", {
    method: "POST",
    body: data,
  });

  const payload = await res.json();
  if (!res.ok && res.status !== 207) {
    throw new Error(payload.detail || "Batch request failed");
  }

  return {
    payload,
    requestId: res.headers.get("X-Request-ID") || "n/a",
    status: res.status,
  };
}

async function refreshMetrics() {
  try {
    const res = await fetch("/metrics");
    const payload = await res.json();
    if (!res.ok) {
      throw new Error(payload.detail || "Unable to fetch metrics");
    }
    metricsBox.textContent = JSON.stringify(payload, null, 2);
  } catch (error) {
    metricsBox.textContent = `Metrics unavailable: ${error.message || String(error)}`;
  }
}

fileInput.addEventListener("change", () => {
  renderFileList(Array.from(fileInput.files || []));
});

refreshMetricsBtn.addEventListener("click", refreshMetrics);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.textContent = "";

  const files = Array.from(fileInput.files || []);
  const query = queryInput.value.trim();
  const mode = coordMode.value;
  const pipeline = videoPipeline.value;
  const frames = Number(maxFrames.value || 8);

  if (!files.length) {
    errorBox.textContent = "Please select at least one file.";
    return;
  }
  if (!query) {
    errorBox.textContent = "Please provide a query.";
    return;
  }

  submitBtn.disabled = true;
  statusBox.textContent = "Running inference...";

  try {
    if (mode === "server_batch") {
      const { payload, requestId, status } = await sendServerBatch(files, query, frames, pipeline);
      renderCards(payload, mode);
      statusBox.innerHTML = `<span class="ok">Done.</span> HTTP ${status} • request_id: ${requestId}`;
    } else {
      const sequentialResults = [];
      const requestIds = [];
      for (const file of files) {
        const result = await sendSingle(file, query, frames, pipeline);
        sequentialResults.push(result);
        requestIds.push(result.requestId);
      }
      renderCards({ results: sequentialResults }, mode);
      statusBox.innerHTML = `<span class="ok">Done.</span> ${files.length} requests • request_ids: ${requestIds.join(", ")}`;
    }

    pushHistory(mode, pipeline, files.length);
    await refreshMetrics();
  } catch (error) {
    errorBox.textContent = error.message || String(error);
    statusBox.textContent = "";
  } finally {
    submitBtn.disabled = false;
  }
});

renderFileList([]);
renderHistory();
refreshMetrics();
