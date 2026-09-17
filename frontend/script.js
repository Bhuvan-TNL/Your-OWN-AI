// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// The FastAPI backend, run locally via `uvicorn app.main:app --reload`
// (default port 8000). Change this if you run the backend elsewhere.
const API_BASE_URL = "http://127.0.0.1:8000";

const ACCEPTED_EXTENSIONS = [".pdf", ".txt"];

// ---------------------------------------------------------------------------
// Element references
// ---------------------------------------------------------------------------

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const statusRetry = document.getElementById("statusRetry");

const uploadForm = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const fileLabel = document.querySelector(".file-label");
const fileLabelText = document.getElementById("fileLabelText");
const uploadButton = document.getElementById("uploadButton");
const uploadStatus = document.getElementById("uploadStatus");
const uploadResult = document.getElementById("uploadResult");

const questionInput = document.getElementById("questionInput");
const askButton = document.getElementById("askButton");
const queryStatus = document.getElementById("queryStatus");

const answerCard = document.getElementById("answerCard");
const answerText = document.getElementById("answerText");
const answerMeta = document.getElementById("answerMeta");

const sourcesCard = document.getElementById("sourcesCard");
const sourcesList = document.getElementById("sourcesList");

// ---------------------------------------------------------------------------
// State guards (prevent duplicate/overlapping submissions)
// ---------------------------------------------------------------------------

let isUploading = false;
let isAsking = false;

// ---------------------------------------------------------------------------
// Small shared helpers
// ---------------------------------------------------------------------------

function setStatusMessage(element, message, kind) {
  element.textContent = message || "";
  element.classList.remove("error", "success", "info");
  if (kind) {
    element.classList.add(kind);
  }
}

function setButtonLoading(button, isLoading, loadingLabel, idleLabel) {
  button.disabled = isLoading;
  button.classList.toggle("is-loading", isLoading);
  button.textContent = isLoading ? loadingLabel : idleLabel;
}

/**
 * Wraps fetch() with consistent error handling.
 * Resolves with the parsed JSON body on success.
 * Rejects with an Error whose `.message` is safe to show the user directly,
 * distinguishing "backend unreachable" from "backend returned an error".
 */
async function requestJson(path, options) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, options);
  } catch (networkError) {
    throw new Error(
      `Could not reach the backend at ${API_BASE_URL}. Is it running?`
    );
  }

  let body = null;
  try {
    body = await response.json();
  } catch (parseError) {
    body = null;
  }

  if (!response.ok) {
    const detail =
      body && typeof body.detail === "string"
        ? body.detail
        : `Request failed (HTTP ${response.status}).`;
    throw new Error(detail);
  }

  if (body === null) {
    throw new Error("The backend returned an unexpected (non-JSON) response.");
  }

  return body;
}

function fileHasAcceptedExtension(filename) {
  const lower = filename.toLowerCase();
  return ACCEPTED_EXTENSIONS.some((extension) => lower.endsWith(extension));
}

// ---------------------------------------------------------------------------
// Backend status
// ---------------------------------------------------------------------------

async function checkBackendHealth() {
  statusDot.classList.remove("online", "offline");
  statusText.textContent = "Checking backend…";
  statusRetry.hidden = true;

  try {
    const health = await requestJson("/health", { method: "GET" });
    statusDot.classList.add("online");
    statusText.textContent = `Backend connected (${health.llm_provider || "unknown provider"})`;
  } catch (error) {
    statusDot.classList.add("offline");
    statusText.textContent = "Backend unavailable";
    statusRetry.hidden = false;
  }
}

statusRetry.addEventListener("click", () => {
  checkBackendHealth();
});

// ---------------------------------------------------------------------------
// Upload
// ---------------------------------------------------------------------------

fileInput.addEventListener("change", () => {
  const file = fileInput.files && fileInput.files[0];
  uploadResult.hidden = true;
  setStatusMessage(uploadStatus, "", null);

  if (!file) {
    fileLabelText.textContent = "Choose a file…";
    fileLabel.classList.remove("has-file");
    uploadButton.disabled = true;
    return;
  }

  fileLabelText.textContent = file.name;
  fileLabel.classList.add("has-file");

  if (!fileHasAcceptedExtension(file.name)) {
    uploadButton.disabled = true;
    setStatusMessage(
      uploadStatus,
      "Only .pdf and .txt files are supported. Please choose a different file.",
      "error"
    );
    return;
  }

  uploadButton.disabled = false;
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (isUploading) {
    return;
  }

  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    setStatusMessage(uploadStatus, "Please choose a file first.", "error");
    return;
  }
  if (!fileHasAcceptedExtension(file.name)) {
    setStatusMessage(uploadStatus, "Only .pdf and .txt files are supported.", "error");
    return;
  }

  isUploading = true;
  uploadResult.hidden = true;
  setStatusMessage(uploadStatus, "", null);
  setButtonLoading(uploadButton, true, "Uploading…", "Upload");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const data = await requestJson("/api/upload", {
      method: "POST",
      body: formData,
    });
    renderUploadResult(data);
    setStatusMessage(uploadStatus, "Document uploaded successfully.", "success");
  } catch (error) {
    setStatusMessage(uploadStatus, error.message, "error");
  } finally {
    isUploading = false;
    setButtonLoading(uploadButton, false, "Uploading…", "Upload");
  }
});

function renderUploadResult(data) {
  uploadResult.innerHTML = "";

  const isDuplicate = Boolean(data.duplicate);

  const entries = [
    ["Filename", data.filename],
    ["Type", data.file_type],
    ["Pages", data.page_count],
    ["Chunks", data.chunk_count],
  ];

  for (const [label, value] of entries) {
    if (value === undefined || value === null) {
      continue;
    }
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = String(value);
    uploadResult.append(dt, dd);
  }

  const statusDt = document.createElement("dt");
  statusDt.textContent = "Status";
  const statusDd = document.createElement("dd");
  const badge = document.createElement("span");
  badge.classList.add("badge", isDuplicate ? "duplicate" : "new");
  badge.textContent = isDuplicate ? "Already indexed" : "Newly indexed";
  statusDd.append(badge);
  uploadResult.append(statusDt, statusDd);

  uploadResult.hidden = false;
}

// ---------------------------------------------------------------------------
// Ask a question
// ---------------------------------------------------------------------------

questionInput.addEventListener("input", () => {
  askButton.disabled = questionInput.value.trim().length === 0;
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (!askButton.disabled) {
      askQuestion();
    }
  }
});

askButton.addEventListener("click", () => {
  askQuestion();
});

async function askQuestion() {
  if (isAsking) {
    return;
  }

  const question = questionInput.value.trim();
  if (!question) {
    setStatusMessage(queryStatus, "Please type a question first.", "error");
    return;
  }

  isAsking = true;
  setStatusMessage(queryStatus, "", null);
  setButtonLoading(askButton, true, "Thinking…", "Ask");
  answerCard.hidden = true;
  sourcesCard.hidden = true;

  try {
    const data = await requestJson("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    renderAnswer(data);
    renderSources(data);
  } catch (error) {
    setStatusMessage(queryStatus, error.message, "error");
  } finally {
    isAsking = false;
    setButtonLoading(askButton, false, "Thinking…", "Ask");
  }
}

function renderAnswer(data) {
  if (typeof data.answer !== "string") {
    setStatusMessage(queryStatus, "The backend returned an unexpected response.", "error");
    return;
  }

  answerText.textContent = data.answer;

  const metaParts = [];
  if (data.provider) {
    metaParts.push(`Provider: ${data.provider}`);
  }
  if (data.model) {
    metaParts.push(`Model: ${data.model}`);
  }
  if (data.timing && typeof data.timing.total_ms === "number") {
    metaParts.push(`Answered in ${Math.round(data.timing.total_ms)} ms`);
  }

  if (metaParts.length > 0) {
    answerMeta.textContent = metaParts.join(" · ");
    answerMeta.hidden = false;
  } else {
    answerMeta.hidden = true;
  }

  answerCard.hidden = false;
}

function renderSources(data) {
  sourcesList.innerHTML = "";

  // `results` carries the actual retrieved text; `sources` is a lighter
  // reference list without text. Prefer `results`, fall back to `sources`.
  let items = [];
  if (Array.isArray(data.results) && data.results.length > 0) {
    items = data.results;
  } else if (Array.isArray(data.sources) && data.sources.length > 0) {
    items = data.sources;
  }

  if (items.length === 0) {
    sourcesCard.hidden = true;
    return;
  }

  for (const item of items) {
    const li = document.createElement("li");
    li.className = "source-item";

    const header = document.createElement("div");
    header.className = "source-item-header";

    const filename = document.createElement("span");
    filename.className = "source-filename";
    filename.textContent = item.filename || "Unknown document";
    header.append(filename);

    const metaBits = [];
    if (item.page_number !== undefined && item.page_number !== null) {
      metaBits.push(`page ${item.page_number}`);
    }
    if (item.chunk_id) {
      metaBits.push(`chunk ${item.chunk_id}`);
    }
    if (typeof item.score === "number") {
      metaBits.push(`relevance ${item.score.toFixed(2)}`);
    }
    if (metaBits.length > 0) {
      const meta = document.createElement("span");
      meta.className = "source-meta";
      meta.textContent = metaBits.join(" · ");
      header.append(meta);
    }

    li.append(header);

    if (typeof item.text === "string" && item.text.trim().length > 0) {
      const snippet = document.createElement("p");
      snippet.className = "source-snippet";
      snippet.textContent = item.text;
      li.append(snippet);
    }

    sourcesList.append(li);
  }

  sourcesCard.hidden = false;
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

checkBackendHealth();