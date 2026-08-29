const state = { result: {}, model: null, samples: [] };

const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
}[char]));
const number = (value, digits = 3) => value === null || value === undefined || Number.isNaN(Number(value)) ? "—" : Number(value).toFixed(digits);
const metric = (metrics, name) => metrics?.[name] ?? null;

function formatModel(name) {
  return String(name || "model").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function getComparisonRows() {
  const result = state.result || {};
  if (result.metric_summary) {
    return Object.entries(result.metric_summary).map(([name, values]) => ({
      name,
      metrics: Object.fromEntries(Object.entries(values).map(([key, value]) => [key, value?.mean ?? null])),
      resource: result.folds?.[0]?.models?.[name]?.resource || null,
      parameters: result.folds?.[0]?.models?.[name]?.parameters || {}
    }));
  }
  if (result.repeated_evaluation?.metric_summary) {
    return Object.entries(result.repeated_evaluation.metric_summary).map(([name, values]) => ({
      name,
      metrics: Object.fromEntries(Object.entries(values).map(([key, value]) => [key, value?.mean ?? null])),
      resource: result.models?.[name]?.resource || null,
      parameters: result.models?.[name]?.parameters || {}
    }));
  }
  if (result.rows) {
    return result.rows.map((row) => ({ name: row.model, metrics: row.metrics || {}, resource: row.resource, parameters: {} }));
  }
  return Object.entries(result.models || {}).map(([name, model]) => ({
    name,
    metrics: model.metrics || {},
    resource: model.resource || null,
    parameters: model.parameters || {}
  }));
}

function datasetInfo() {
  const result = state.result || {};
  return result.dataset || {};
}

function renderShell() {
  const result = state.result || {};
  const dataset = datasetInfo();
  const split = result.split || {};
  const execution = result.execution || result.study || {};
  const profile = dataset.task_profile;
  const isFixture = String(dataset.name || "").includes("fixture") || String(dataset.name || "").includes("synthetic");
  const mode = profile ? (isFixture ? "profile fixture" : "profiled cohort") : "benchmark record";
  $("#mode-label").textContent = mode;
  $("#record-status").textContent = result.study ? "study complete" : result.models ? "record loaded" : "waiting";
  $("#dataset-value").textContent = dataset.name || "no record";
  $("#split-value").textContent = split.strategy || execution.split_strategy || "—";
  $("#backend-value").textContent = execution.resolved_backend || execution.backend || execution.backend_mode || "—";
  $("#task-value").textContent = profile?.name || (dataset.name ? "diagnostic benchmark" : "—");
  $("#positive-value").textContent = dataset.positive_label || "—";
  $("#test-value").textContent = split.test_rows ?? (result.folds ? result.folds.reduce((sum, fold) => sum + (fold.test_rows || 0), 0) : "—");
  if (profile?.horizon_days) {
    $("#hero-lede").textContent = `${profile.outcome_definition || "Profiled outcome"}. The console keeps cohort boundaries, operating point, and circuit cost visible together.`;
  }
}

function renderComparison() {
  const rows = getComparisonRows();
  const mount = $("#comparison");
  if (!rows.length) {
    mount.innerHTML = '<div class="loading-state">No model results in this artifact. Run qhealth or qhealth-study first.</div>';
    $("#evidence-summary").innerHTML = "";
    return;
  }
  const best = Math.max(...rows.map((row) => Number(row.metrics.balanced_accuracy ?? -Infinity)));
  const headers = ["Model", "Balanced accuracy", "Sensitivity", "Specificity", "ROC-AUC", "PR-AUC", "Brier"];
  const body = rows.map((row) => {
    const isBest = Number(row.metrics.balanced_accuracy) === best && Number.isFinite(best);
    return `<tr class="${isBest ? "best" : ""}"><td>${escapeHtml(formatModel(row.name))}</td><td>${number(row.metrics.balanced_accuracy)}</td><td>${number(row.metrics.sensitivity)}</td><td>${number(row.metrics.specificity)}</td><td>${number(row.metrics.roc_auc)}</td><td>${number(row.metrics.pr_auc)}</td><td>${number(row.metrics.brier_score)}</td></tr>`;
  }).join("");
  mount.innerHTML = `<table class="comparison-table"><thead><tr>${headers.map((header) => `<th>${header}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table><p class="table-caption">${state.result.study ? "Outer-fold mean values. Paired deltas and fold details are preserved in the JSON record." : "Values are held-out metrics; a dash means the model did not produce that score type."}</p>`;

  const bestRow = rows.find((row) => Number(row.metrics.balanced_accuracy) === best) || rows[0];
  const quantum = rows.find((row) => String(row.name).includes("qsvc") || String(row.name) === "vqc");
  const classical = rows.find((row) => !String(row.name).includes("qsvc") && String(row.name) !== "vqc");
  const delta = quantum && classical && quantum.metrics.balanced_accuracy != null && classical.metrics.balanced_accuracy != null
    ? Number(quantum.metrics.balanced_accuracy) - Number(classical.metrics.balanced_accuracy) : null;
  $("#evidence-summary").innerHTML = [
    ["LEADING HELD-OUT SIGNAL", number(bestRow.metrics.balanced_accuracy), `${formatModel(bestRow.name)} · balanced accuracy`],
    ["OPERATING POINT", number(bestRow.metrics.sensitivity), `${formatModel(bestRow.name)} · sensitivity`],
    ["QUANTUM / CLASSICAL DELTA", delta === null ? "—" : `${delta >= 0 ? "+" : ""}${number(delta)}`, delta === null ? "Comparable pair unavailable" : "balanced accuracy difference"]
  ].map(([label, value, caption]) => `<div class="summary-item"><small>${label}</small><strong>${value}</strong><p>${escapeHtml(caption)}</p></div>`).join("");
  renderResource(bestRow);
  renderFeatureSummary(bestRow);
}

function renderResource(row) {
  const resource = row?.resource || state.result.models?.[row?.name]?.resource || {};
  const probe = resource.circuit_probe || state.result.hardware_probe || {};
  const entries = [
    ["backend", resource.backend || state.result.execution?.resolved_backend || state.result.study?.backend || "classical path"],
    ["qubits", resource.qubits ?? state.result.study?.qubits ?? probe.logical_qubits ?? "n/a"],
    ["estimated kernel pairs", resource.estimated_kernel_pairs ?? "n/a"],
    ["training rows", resource.training_rows ?? state.result.split?.train_rows ?? "—"],
    ["two-qubit gates", probe.two_qubit_gates ?? "n/a"]
  ];
  $("#resource-list").innerHTML = entries.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("");
}

function renderFeatureSummary(row) {
  const model = state.result.models?.[row?.name];
  const explanation = model?.explanation;
  const mount = $("#feature-impact");
  if (!explanation || explanation.status !== "ok" || !explanation.features?.length) {
    mount.innerHTML = '<div class="panel-kicker"><span class="panel-index">F</span><span>FEATURE SENSITIVITY</span></div><div class="loading-state">Explanation appears after a model or row is scored.</div>';
    return;
  }
  const features = explanation.features.slice(0, 8);
  const max = Math.max(...features.map((item) => Math.abs(Number(item.mean_signed_score_delta || item.mean_abs_score_delta || 0))), 1e-9);
  mount.innerHTML = `<div class="feature-panel-heading"><div><div class="panel-kicker"><span class="panel-index">F</span><span>FEATURE SENSITIVITY</span></div><h3>${escapeHtml(formatModel(row.name))}</h3></div><span>${escapeHtml(explanation.method || "reference replacement")}</span></div><div class="feature-bars">${features.map((item) => { const value = Number(item.mean_signed_score_delta || 0); return `<div class="feature-bar-row"><span>${escapeHtml(item.feature)}</span><div class="feature-bar-track"><div class="feature-bar-fill ${value < 0 ? "negative" : ""}" style="width:${Math.max(3, Math.min(100, Math.abs(value) / max * 100))}%"></div></div><span>${value >= 0 ? "+" : ""}${number(value)}</span></div>`; }).join("")}</div>`;
}

function renderSamples() {
  const select = $("#sample-select");
  if (!state.samples.length) return;
  select.innerHTML = '<option value="">Choose a supplied row…</option>' + state.samples.map((row, index) => `<option value="${index}">${escapeHtml(row.row_id || `row-${index + 1}`)}</option>`).join("");
  select.addEventListener("change", () => {
    const row = state.samples[Number(select.value)];
    if (row) $("#features-input").value = JSON.stringify(row.features, null, 2);
  });
}

function renderPrediction(result) {
  const prediction = result.predictions?.[0];
  const score = result.scores?.[0];
  const abstained = result.abstained?.[0];
  const badge = abstained ? "ABSTAIN / REVIEW" : prediction === 1 ? "POSITIVE SIGNAL" : "NEGATIVE SIGNAL";
  const badgeClass = abstained ? "abstain" : prediction === 1 ? "positive" : "";
  const rowExplanation = result.explanation?.rows?.[0];
  const impacts = rowExplanation?.top_features || [];
  const max = Math.max(...impacts.map((item) => Math.abs(Number(item.score_delta || 0))), 1e-9);
  $("#prediction-output").innerHTML = `<div class="prediction-card"><div class="prediction-card-heading"><div><div class="panel-kicker"><span class="panel-index">✓</span><span>${escapeHtml(formatModel(result.model_name))}</span></div><h3>Local score returned.</h3></div><div class="prediction-score"><strong>${number(score)}</strong><span>${escapeHtml(result.score_type || "score")}</span></div></div><div class="prediction-badge ${badgeClass}">${badge}</div><div class="prediction-meta"><div><span>row</span><strong>${escapeHtml(rowExplanation?.row_id || "dashboard-row")}</strong></div><div><span>threshold</span><strong>${number(result.threshold)}</strong></div><div><span>coverage</span><strong>${number(result.abstention?.coverage, 2)}</strong></div></div>${impacts.length ? `<p class="explanation-title">Top input sensitivities</p><div class="impact-list">${impacts.map((item) => { const value = Number(item.score_delta || 0); return `<div class="impact-row"><span class="impact-name">${escapeHtml(item.feature)}</span><div class="impact-track"><div class="impact-bar ${value < 0 ? "negative" : ""}" style="width:${Math.max(4, Math.min(100, Math.abs(value) / max * 100))}%"></div></div><span class="impact-value">${value >= 0 ? "+" : ""}${number(value)}</span></div>`; }).join("")}</div>` : '<p class="field-help">No continuous explanation was available for this model.</p>'}</div>`;
}

async function submitPrediction(event) {
  event.preventDefault();
  const status = $("#prediction-status");
  status.textContent = "";
  if (!state.model?.available) {
    status.textContent = "Start qhealth-dashboard with --model to enable local prediction.";
    return;
  }
  try {
    const features = JSON.parse($("#features-input").value);
    if (!features || Array.isArray(features) || typeof features !== "object") throw new Error("feature input must be a JSON object");
    status.textContent = "Scoring locally…";
    const response = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ features, explain: true, row_id: "dashboard-row" }) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "prediction failed");
    status.textContent = "Score complete.";
    renderPrediction(payload);
  } catch (error) {
    status.textContent = error.message;
  }
}

async function load() {
  try {
    const [result, model, samples] = await Promise.all([
      fetch("/api/result").then((response) => response.json()),
      fetch("/api/model").then((response) => response.json()),
      fetch("/api/samples").then((response) => response.json())
    ]);
    state.result = result || {};
    state.model = model || null;
    state.samples = samples?.rows || [];
    renderShell();
    renderComparison();
    renderSamples();
    if (state.model?.available && state.samples[0]) $("#features-input").value = JSON.stringify(state.samples[0].features, null, 2);
    if (!state.model?.available) $("#prediction-status").textContent = "Prediction is disabled until a saved model is supplied.";
  } catch (error) {
    $("#mode-label").textContent = "record unavailable";
    $("#comparison").innerHTML = `<div class="loading-state">${escapeHtml(error.message)}</div>`;
  }
}

$("#prediction-form").addEventListener("submit", submitPrediction);
$("#clear-input").addEventListener("click", () => { $("#features-input").value = ""; $("#prediction-status").textContent = ""; });
load();
