const API = "/api";

const modulesTbody = document.getElementById("modules-tbody");
const detailPanel = document.getElementById("detail-panel");
const detailTbody = document.getElementById("detail-tbody");
const detailModuleCode = document.getElementById("detail-module-code");
const detailSparkline = document.getElementById("detail-sparkline");
const flaggedList = document.getElementById("flagged-list");
const statFlagged = document.getElementById("stat-flagged");
const statTotal = document.getElementById("stat-total");
const statCohorts = document.getElementById("stat-cohorts");
const toast = document.getElementById("toast");

let toastTimer = null;

function showToast(message, isError = false) {
  toast.textContent = message;
  toast.classList.toggle("toast-error", isError);
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, 3500);
}

async function apiGet(path) {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `Request failed (${res.status})`);
  }
  return res.json();
}

async function apiPost(path, payload) {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `Request failed (${res.status})`);
  return body;
}

function pct(value) {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function statusBadge(isDifficult) {
  if (isDifficult === 1) return `<span class="badge badge-risk">Historically difficult</span>`;
  if (isDifficult === 0) return `<span class="badge badge-safe">Historically stable</span>`;
  return `<span class="badge badge-watch">No prediction yet</span>`;
}

function rateCell(rate) {
  const widthPct = Math.max(0, Math.min(100, (rate ?? 0) * 100));
  return `
    <div class="rate-cell">
      <div class="rate-bar"><div class="rate-bar-fill" style="width:${widthPct}%"></div></div>
      <span class="rate-value">${pct(rate)}</span>
    </div>`;
}

async function loadModules() {
  modulesTbody.innerHTML = `<tr><td colspan="6" class="empty-row">Loading modules…</td></tr>`;
  try {
    const modules = await apiGet("/modules");

    statTotal.textContent = modules.length;
    statCohorts.textContent = modules.reduce((sum, m) => sum + m.n_cohorts, 0);
    statFlagged.textContent = modules.filter((m) => m.latest_is_difficult === 1).length;

    if (modules.length === 0) {
      modulesTbody.innerHTML = `<tr><td colspan="6" class="empty-row">No modules seeded yet.</td></tr>`;
      return;
    }

    modulesTbody.innerHTML = "";
    modules.forEach((m) => {
      const tr = document.createElement("tr");
      tr.className = "clickable";
      tr.innerHTML = `
        <td><span class="module-code">${m.code_module}</span></td>
        <td>${m.n_cohorts}</td>
        <td>${m.latest_presentation}</td>
        <td>${rateCell(m.latest_difficulty_rate)}</td>
        <td>${statusBadge(m.latest_is_difficult)}</td>
        <td><button class="btn btn-predict" data-module="${m.code_module}" data-presentation="${m.latest_presentation}">Predict next</button></td>
      `;
      tr.addEventListener("click", (evt) => {
        if (evt.target.closest("button")) return;
        openDetail(m.code_module);
      });
      modulesTbody.appendChild(tr);
    });

    modulesTbody.querySelectorAll(".btn-predict").forEach((btn) => {
      btn.addEventListener("click", (evt) => {
        evt.stopPropagation();
        runPrediction(btn.dataset.module, btn.dataset.presentation, btn);
      });
    });
  } catch (err) {
    modulesTbody.innerHTML = `<tr><td colspan="6" class="empty-row">Couldn't load modules: ${err.message}</td></tr>`;
  }
}

async function runPrediction(codeModule, latestPresentation, btn) {
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Predicting…";
  try {
    // Bump the presentation year by one as a simple "next" guess (e.g. 2014J -> 2015J).
    const nextPresentation = latestPresentation.replace(/^(\d{4})/, (y) => String(Number(y) + 1));
    const result = await apiPost("/predict", {
      code_module: codeModule,
      code_presentation: nextPresentation,
    });
    showToast(
      `${codeModule} ${nextPresentation}: ${result.predicted_difficult ? "flagged" : "not flagged"} ` +
      `(${pct(result.predicted_probability)} risk)`,
      result.predicted_difficult
    );
    await Promise.all([loadModules(), loadFlagged()]);
  } catch (err) {
    showToast(`Prediction failed: ${err.message}`, true);
  } finally {
    btn.disabled = false;
    btn.textContent = original;
  }
}

function renderSparkline(cohorts) {
  const width = 560, height = 60, pad = 10;
  const rates = cohorts.map((c) => c.difficulty_rate);
  const min = Math.min(...rates), max = Math.max(...rates);
  const span = max - min || 1;

  const points = cohorts.map((c, i) => {
    const x = pad + (i / Math.max(1, cohorts.length - 1)) * (width - pad * 2);
    const y = height - pad - ((c.difficulty_rate - min) / span) * (height - pad * 2);
    return { x, y, difficult: c.is_difficult };
  });

  const path = points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  const dots = points
    .map((p) => `<circle class="sparkline-point${p.difficult ? " difficult" : ""}" cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="3.5"></circle>`)
    .join("");

  detailSparkline.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}" preserveAspectRatio="none">
      <path d="${path}" fill="none" stroke="#D6DBCE" stroke-width="1.5"></path>
      ${dots}
    </svg>`;
}

async function openDetail(codeModule) {
  detailModuleCode.textContent = codeModule;
  detailPanel.hidden = false;
  detailPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  detailTbody.innerHTML = `<tr><td colspan="5" class="empty-row">Loading…</td></tr>`;

  try {
    const cohorts = await apiGet(`/modules/${codeModule}/cohorts`);
    renderSparkline(cohorts);
    detailTbody.innerHTML = cohorts.map((c) => `
      <tr>
        <td>${c.code_presentation}</td>
        <td>${c.enrollment_count}</td>
        <td>${rateCell(c.difficulty_rate)}</td>
        <td>${c.hist_n_prior_cohorts}</td>
        <td>${statusBadge(c.is_difficult)}</td>
      </tr>
    `).join("");
  } catch (err) {
    detailTbody.innerHTML = `<tr><td colspan="5" class="empty-row">Couldn't load history: ${err.message}</td></tr>`;
  }
}

document.getElementById("close-detail").addEventListener("click", () => {
  detailPanel.hidden = true;
});

async function loadFlagged() {
  try {
    const preds = await apiGet("/flagged");
    if (preds.length === 0) {
      flaggedList.innerHTML = `<li class="empty-row">No modules currently flagged. Run a prediction from the table above.</li>`;
      return;
    }
    flaggedList.innerHTML = preds.map((p) => `
      <li class="flagged-item">
        <div class="flagged-main">
          <span class="flagged-code">${p.code_module}</span>
          <span class="flagged-meta">${p.code_presentation || "—"} · scored ${new Date(p.created_at).toLocaleString()}</span>
        </div>
        <span class="flagged-prob">${pct(p.predicted_probability)}</span>
      </li>
    `).join("");
  } catch (err) {
    flaggedList.innerHTML = `<li class="empty-row">Couldn't load flagged modules: ${err.message}</li>`;
  }
}

document.getElementById("refresh-btn").addEventListener("click", () => {
  loadModules();
  loadFlagged();
});

loadModules();
loadFlagged();
