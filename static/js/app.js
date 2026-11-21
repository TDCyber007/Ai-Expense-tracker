/* ============================================================
   AI Expense Tracker — frontend logic (vanilla JS + Chart.js)
   ============================================================ */
"use strict";

const $ = (id) => document.getElementById(id);

const CATEGORIES = window.CATEGORIES || [];
const money = (n) =>
  "$" + Number(n || 0).toLocaleString("en-US",
    { minimumFractionDigits: 2, maximumFractionDigits: 2 });

// Nice colors for the common categories; anything else gets a cycling color.
const CAT_COLORS = {
  "Food & Dining": "#ff9f43",
  "Groceries": "#2ecc71",
  "Transportation": "#6c8cff",
  "Housing": "#a06bff",
  "Utilities": "#00cfe8",
  "Entertainment": "#ff5c8a",
  "Health & Fitness": "#20c997",
  "Shopping": "#e05bd6",
  "Travel": "#f0b429",
  "Subscriptions": "#8a94ff",
  "Education": "#26c6da",
  "Personal Care": "#ffb37a",
  "Other": "#8891b3",
};
const PALETTE = ["#6c8cff", "#ff9f43", "#2ecc71", "#ff5c8a", "#00cfe8",
  "#a06bff", "#f0b429", "#20c997", "#e05bd6", "#8a94ff", "#ffb37a",
  "#26c6da", "#7ee787"];
function catColor(cat) {
  if (CAT_COLORS[cat]) return CAT_COLORS[cat];
  const i = Math.max(0, CATEGORIES.indexOf(cat));
  return PALETTE[i % PALETTE.length];
}
const shortDate = (iso) => {
  const d = new Date(iso + "T00:00:00");
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
};

// ------------------------------------------------------------ Toast -------
let toastTimer;
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 2200);
}

// ------------------------------------------------------------ Charts ------
let barChart, doughnutChart;
Chart.defaults.color = "#98a0bd";
Chart.defaults.font.family =
  "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif";
Chart.defaults.font.size = 11;

function renderBarChart(history) {
  const ctx = $("barChart");
  const labels = history.map((h) => h.month);
  const data = history.map((h) => h.total);
  if (barChart) barChart.destroy();
  barChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Spend",
        data,
        backgroundColor: "rgba(108,140,255,0.55)",
        hoverBackgroundColor: "rgba(160,107,255,0.85)",
        borderRadius: 6,
        maxBarThickness: 34,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false },
        tooltip: { callbacks: { label: (c) => money(c.parsed.y) } } },
      scales: {
        x: { grid: { display: false }, border: { display: false } },
        y: { grid: { color: "rgba(255,255,255,0.06)" },
          border: { display: false },
          ticks: { callback: (v) => "$" + v.toLocaleString() } },
      },
    },
  });
}

function renderDoughnut(byCategory) {
  const ctx = $("doughnut");
  const labels = byCategory.map((c) => c.category);
  const data = byCategory.map((c) => c.amount);
  const colors = byCategory.map((c) => catColor(c.category));
  if (doughnutChart) doughnutChart.destroy();
  if (!data.length) {
    doughnutChart = new Chart(ctx, {
      type: "doughnut",
      data: { labels: ["No data"], datasets: [{ data: [1],
        backgroundColor: ["#232744"], borderWidth: 0 }] },
      options: { responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        cutout: "70%" },
    });
    return;
  }
  doughnutChart = new Chart(ctx, {
    type: "doughnut",
    data: { labels, datasets: [{ data, backgroundColor: colors,
      borderColor: "#14182c", borderWidth: 2, hoverOffset: 6 }] },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: "66%",
      plugins: {
        legend: { position: "bottom",
          labels: { boxWidth: 10, boxHeight: 10, padding: 10,
            usePointStyle: true, pointStyle: "circle" } },
        tooltip: { callbacks: { label: (c) =>
          ` ${c.label}: ${money(c.parsed)}` } },
      },
    },
  });
}

// ------------------------------------------------------------ Render ------
function renderSummary(s) {
  $("kpiMonth").textContent = money(s.month_total);
  if (s.month_change_pct === null || s.month_change_pct === undefined) {
    $("kpiMonthSub").innerHTML =
      `<span class="muted">${s.count} txns this month</span>`;
  } else {
    const up = s.month_change_pct >= 0;
    $("kpiMonthSub").innerHTML =
      `<span class="${up ? "up" : "down"}">${up ? "&#9650;" : "&#9660;"} ` +
      `${Math.abs(s.month_change_pct)}%</span> vs last month`;
  }

  $("kpiProjected").textContent = money(s.projected_total);
  const fcRange =
    s.projected_total > 0 && s.days_in_month
      ? `${s.days_elapsed}/${s.days_in_month} days in`
      : "forecast by the model";
  $("kpiProjectedSub").textContent = fcRange;

  $("kpiAvg").textContent = money(s.avg_daily);
  $("kpiAvgSub").textContent = `day ${s.days_elapsed || 1} of ${s.days_in_month}`;

  if (s.top_category) {
    $("kpiAvgSub").textContent =
      `day ${s.days_elapsed || 1}/${s.days_in_month} · top: ${s.top_category}`;
  }
}

function renderForecast(fc) {
  if (!fc || !fc.has_data) {
    $("fcTotal").textContent = "$0.00";
    $("fcBand").textContent = "add expenses to see a forecast";
    $("fcFill").style.width = "0%";
    $("catForecast").innerHTML = "";
    $("fcNext").textContent = "";
    return;
  }
  const c = fc.current_month;
  $("fcTotal").textContent = money(c.projected_total);
  $("fcBand").textContent = `range ${money(c.lower)} – ${money(c.upper)}`;
  const pct = c.projected_total > 0
    ? Math.min(100, (c.so_far / c.projected_total) * 100) : 0;
  $("fcFill").style.width = pct + "%";
  $("fcSoFar").textContent = money(c.so_far) + " spent so far";
  $("fcRemaining").textContent =
    `${c.days_remaining} days left · ~${money(c.daily_burn)}/day`;

  const maxProj = Math.max(...fc.by_category.map((x) => x.projected_total), 1);
  $("catForecast").innerHTML = fc.by_category.map((x) => `
    <div class="cf-row">
      <div class="cf-top">
        <span class="cf-name">
          <span class="dot" style="background:${catColor(x.category)}"></span>
          ${x.category}
        </span>
        <span class="cf-amt">${money(x.projected_total)}</span>
      </div>
      <div class="mini-track">
        <div class="mini-fill" style="width:${(x.projected_total / maxProj) * 100}%;
          background:${catColor(x.category)}"></div>
      </div>
    </div>`).join("");

  $("fcNext").textContent = fc.next_month > 0
    ? `Next month projected at ${money(fc.next_month)} based on recent trend.`
    : "";
}

function renderTips(rec) {
  const tips = rec.tips || [];
  const emoji = {
    rising: "&#11014;", discretionary: "&#9986;", subscriptions: "&#128257;",
    budget: "&#128680;", pace: "&#9201;", on_track: "&#9989;",
  };
  $("tips").innerHTML = tips.map((t) => `
    <div class="tip ${t.severity}">
      <div class="emoji">${emoji[t.type] || "&#128161;"}</div>
      <div class="body">
        <div class="title">${t.title}</div>
        <div class="detail">${t.detail}</div>
        ${t.potential_saving > 0
          ? `<div class="save">Potential saving: ${money(t.potential_saving)}</div>`
          : ""}
      </div>
    </div>`).join("");

  const total = rec.total_potential_saving || 0;
  $("kpiSave").textContent = money(total);
  $("kpiSaveSub").textContent = total > 0
    ? `from ${tips.length} AI tips`
    : "you're on track";
  $("tipsTotal").hidden = !(total > 0);
  $("tipsTotalAmt").textContent = money(total);
}

function renderTxns(expenses) {
  const body = $("txnBody");
  $("txnCount").textContent = expenses.length ? `(${expenses.length})` : "";
  $("txnEmpty").hidden = expenses.length > 0;
  body.innerHTML = expenses.map((e) => {
    const options = CATEGORIES.map((c) =>
      `<option value="${c}" ${c === e.category ? "selected" : ""}>${c}</option>`
    ).join("");
    return `<tr>
      <td class="muted" style="white-space:nowrap">${shortDate(e.date)}</td>
      <td class="txn-desc">${escapeHtml(e.description)}
        ${e.confidence && e.source === "manual"
          ? `<div class="txn-conf">${Math.round(e.confidence * 100)}% conf.</div>`
          : ""}
      </td>
      <td>
        <span class="badge">
          <span class="dot" style="background:${catColor(e.category)}"></span>
          <select data-id="${e.id}" class="recat">${options}</select>
        </span>
      </td>
      <td class="txn-amt">${money(e.amount)}</td>
      <td><button class="icon-btn del" data-id="${e.id}" title="Delete">&#128465;</button></td>
    </tr>`;
  }).join("");

  body.querySelectorAll(".del").forEach((b) =>
    b.addEventListener("click", () => deleteExpense(b.dataset.id)));
  body.querySelectorAll(".recat").forEach((sel) =>
    sel.addEventListener("change", () => recategorize(sel.dataset.id, sel.value)));
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "\u0026amp;")
    .replaceAll("<", "\u0026lt;")
    .replaceAll(">", "\u0026gt;")
    .replaceAll('"', "\u0026quot;")
    .replaceAll("'", "\u0026#39;");
}

// ------------------------------------------------------------ Data --------
async function loadData() {
  const [summary, insights, txns] = await Promise.all([
    fetch("/api/summary").then((r) => r.json()),
    fetch("/api/insights").then((r) => r.json()),
    fetch("/api/expenses?limit=25").then((r) => r.json()),
  ]);
  renderSummary(summary);
  renderForecast(insights.forecast);
  renderBarChart(insights.forecast.monthly_history || []);
  renderDoughnut(summary.by_category || []);
  renderTips(insights.recommendations);
  renderTxns(txns.expenses || []);
}

// ------------------------------------------------------------ Actions ----
async function deleteExpense(id) {
  await fetch(`/api/expenses/${id}`, { method: "DELETE" });
  toast("Deleted");
  loadData();
}

async function recategorize(id, category) {
  await fetch(`/api/expenses/${id}/recategorize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ category }),
  });
  toast(`Moved to ${category} · model re-trained`);
  loadData();
}

$("addForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const desc = $("desc").value.trim();
  const amount = $("amount").value;
  const date = $("date").value;
  if (!desc || !amount) return;
  await fetch("/api/expenses", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ description: desc, amount, date }),
  });
  $("desc").value = "";
  $("amount").value = "";
  $("catPreview").textContent = "detecting…";
  toast("Expense added");
  loadData();
});

$("demoBtn").addEventListener("click", async () => {
  $("demoBtn").disabled = true;
  $("demoBtn").textContent = "Loading…";
  const r = await fetch("/api/seed", { method: "POST" }).then((x) => x.json());
  toast(`Added ${r.added} demo transactions`);
  $("demoBtn").disabled = false;
  $("demoBtn").textContent = "Load demo data";
  loadData();
});

$("resetBtn").addEventListener("click", async () => {
  await fetch("/api/reset", { method: "POST" });
  toast("Data reset");
  loadData();
});

// Live category preview while typing.
let previewTimer;
$("desc").addEventListener("input", () => {
  clearTimeout(previewTimer);
  const desc = $("desc").value.trim();
  if (!desc) { $("catPreview").textContent = "detecting…"; return; }
  previewTimer = setTimeout(async () => {
    const r = await fetch("/api/predict?description=" +
      encodeURIComponent(desc)).then((x) => x.json());
    const el = $("catPreview");
    if (r.category) {
      el.textContent = r.category;
      el.style.color = catColor(r.category);
    } else {
      el.textContent = "—";
      el.style.color = "";
    }
  }, 250);
});

// ------------------------------------------------------------ Init --------
(function init() {
  $("date").value = new Date().toISOString().slice(0, 10);
  loadData();
})();