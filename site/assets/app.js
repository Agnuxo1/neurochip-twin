/* NeuroChip Twin v2 — project site script.
   Plain JavaScript, no dependencies, no tracking, no external requests.
   Every number shown is read at load time from data/*.json (written by scripts/precompute_site.py). */

/* =========================================================================================
   PUBLICATION CONFIG — replace the four placeholders at publication (keep the quotes).
   A value that still looks like "SOMETHING_URL" is treated as not yet published: the button
   is shown greyed out with the label "soon".
   ========================================================================================= */
const CONFIG = {
  REPORT_URL: "https://github.com/Agnuxo1/neurochip-twin/releases/download/v2.0.0/neurochip_twin_v2.pdf",          // technical report (PDF)
  REPO_URL: "https://github.com/Agnuxo1/neurochip-twin",              // public code repository
  VIDEO_URL: "https://youtu.be/yLzpvGqP_2A",            // hosted demo video page
  KAGGLE_URL: "https://www.kaggle.com/code/franciscoangulo/neurochip-twin-v2-demo",          // Kaggle Writeup / notebook
  // VIDEO SLOT: once the demo video is at site/video/teaser.mp4, set VIDEO_FILE to "video/teaser.mp4".
  // While it is empty the page shows a placeholder and makes no request for the file (clean console).
  VIDEO_FILE: "video/teaser.mp4",
  VIDEO_CAPTIONS: ""                 // optional WebVTT captions, e.g. "video/teaser.en.vtt"
};

(function () {
  "use strict";

  /* ------------------------------------------------------------------ helpers */
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const SVGNS = "http://www.w3.org/2000/svg";
  const DIVS_FALLBACK = [5, 7, 9, 12];
  const isSet = (v) => typeof v === "string" && v.length > 0 && !/^[A-Z_]+_URL$/.test(v);
  const esc = (x) => String(x).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const num = (x) => (typeof x === "number" ? x.toLocaleString("en-US") : String(x));
  const pct = (x, d = 1) => (x == null ? "–" : (100 * x).toFixed(d) + "%");
  const fx = (x, d = 2) => (x == null || !isFinite(x) ? "–" : (x < 0 ? "−" : "") + Math.abs(x).toFixed(d));

  function svg(tag, attrs, parent) {
    const e = document.createElementNS(SVGNS, tag);
    if (attrs) for (const k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function html(tag, attrs, inner) {
    const e = document.createElement(tag);
    if (attrs) for (const k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
    if (inner != null) e.innerHTML = inner;
    return e;
  }
  async function getJSON(url) {
    const r = await fetch(url, { cache: "no-cache" });
    if (!r.ok) throw new Error(url + " → HTTP " + r.status);
    return r.json();
  }
  function fmtConc(logc) {
    const c = Math.pow(10, logc);
    if (c >= 100) return String(Math.round(c));
    return String(Number(c.toPrecision(2)));
  }
  function decadeLabel(n) {
    if (n >= 0) return String(Math.pow(10, n));
    return "0." + "0".repeat(-n - 1) + "1";
  }
  function niceStep(span, target) {
    const raw = span / Math.max(1, target);
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const m = raw / p;
    return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
  }

  /* ------------------------------------------------------------------ theme */
  function effectiveTheme() {
    const a = document.documentElement.getAttribute("data-theme");
    if (a === "light" || a === "dark") return a;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function setupTheme() {
    const btn = $("#theme-toggle");
    const sync = () => {
      const t = effectiveTheme();
      btn.setAttribute("aria-label", t === "dark" ? "Switch to light theme" : "Switch to dark theme");
      btn.title = btn.getAttribute("aria-label");
    };
    btn.addEventListener("click", () => {
      const next = effectiveTheme() === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("nct-theme", next); } catch (e) { /* storage unavailable: theme still switches */ }
      sync();
    });
    if (window.matchMedia) {
      const mq = window.matchMedia("(prefers-color-scheme: dark)");
      if (mq.addEventListener) mq.addEventListener("change", sync);
    }
    sync();
  }

  /* ------------------------------------------------------------------ navigation */
  function setupNav() {
    const nav = $("#site-nav"), btn = $("#menu-btn");
    btn.addEventListener("click", () => {
      const open = !nav.classList.contains("open");
      nav.classList.toggle("open", open);
      btn.setAttribute("aria-expanded", String(open));
    });
    $$("a", nav).forEach((a) => a.addEventListener("click", () => {
      nav.classList.remove("open");
      btn.setAttribute("aria-expanded", "false");
    }));
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && nav.classList.contains("open")) { nav.classList.remove("open"); btn.setAttribute("aria-expanded", "false"); btn.focus(); }
    });
    if ("IntersectionObserver" in window) {
      const links = new Map($$("a", nav).map((a) => [a.getAttribute("href").slice(1), a]));
      const io = new IntersectionObserver((entries) => {
        entries.forEach((en) => {
          if (en.isIntersecting && links.has(en.target.id)) {
            links.forEach((a) => a.removeAttribute("aria-current"));
            links.get(en.target.id).setAttribute("aria-current", "true");
          }
        });
      }, { rootMargin: "-45% 0px -50% 0px" });
      links.forEach((_, id) => { const s = document.getElementById(id); if (s) io.observe(s); });
    }
  }

  /* ------------------------------------------------------------------ publication links */
  function setupLinks() {
    $$("[data-link]").forEach((a) => {
      const key = a.getAttribute("data-link");
      const url = CONFIG[key];
      if (isSet(url)) {
        a.href = url;
        a.target = "_blank";
        a.rel = "noopener";
      } else {
        a.classList.add("is-pending");
        a.setAttribute("aria-disabled", "true");
        a.title = "Link added at publication";
        a.removeAttribute("href");
        a.setAttribute("role", "link");
      }
    });
    $$("[data-config]").forEach((s) => {
      const v = CONFIG[s.getAttribute("data-config")];
      s.textContent = isSet(v) ? v : "<repository-url>";
    });
  }

  /* ------------------------------------------------------------------ video slot */
  async function setupVideo() {
    const v = $("#teaser"), ph = $("#video-placeholder");
    if (!v || !CONFIG.VIDEO_FILE) return;
    try {
      const r = await fetch(CONFIG.VIDEO_FILE, { method: "HEAD", cache: "no-cache" });
      if (!r.ok) return;
      v.src = CONFIG.VIDEO_FILE;
      if (CONFIG.VIDEO_CAPTIONS) {
        try {
          const c = await fetch(CONFIG.VIDEO_CAPTIONS, { method: "HEAD", cache: "no-cache" });
          const tr = v.querySelector("track");
          if (c.ok && tr) tr.src = CONFIG.VIDEO_CAPTIONS; else if (tr) tr.remove();
        } catch (e) { /* captions optional */ }
      }
      v.hidden = false;
      ph.hidden = true;
    } catch (e) { /* keep the placeholder */ }
  }

  /* ------------------------------------------------------------------ lightbox, copy */
  function setupLightbox() {
    const dlg = $("#lightbox"), img = $("#lb-img"), cap = $("#lb-cap");
    if (!dlg || typeof dlg.showModal !== "function") return;
    $$("[data-zoom]").forEach((b) => b.addEventListener("click", () => {
      const src = b.querySelector("img");
      img.src = src.currentSrc || src.src;
      img.alt = src.alt;
      const fc = b.closest("figure") && b.closest("figure").querySelector("figcaption");
      cap.textContent = fc ? fc.textContent : "";
      dlg.showModal();
    }));
    dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
  }
  function setupCopy() {
    $$("[data-copy]").forEach((b) => b.addEventListener("click", async () => {
      const pre = document.getElementById(b.getAttribute("data-copy"));
      const text = pre ? pre.innerText : "";
      let ok = false;
      try { await navigator.clipboard.writeText(text); ok = true; } catch (e) {
        const r = document.createRange(); r.selectNodeContents(pre);
        const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(r);
        try { ok = document.execCommand("copy"); } catch (e2) { ok = false; }
        sel.removeAllRanges();
      }
      b.textContent = ok ? "Copied" : "Press Ctrl+C";
      setTimeout(() => { b.textContent = "Copy"; }, 1600);
    }));
  }

  /* ------------------------------------------------------------------ key numbers and tables */
  function renderKPIs(S) {
    const box = $("#kpis");
    box.innerHTML = "";
    (S.key_numbers || []).forEach((c) => {
      const card = html("article", { class: "kpi", "aria-label": c.title });
      card.appendChild(html("div", { class: "kpi-v" }, esc(c.value)));
      card.appendChild(html("p", { class: "kpi-t" }, esc(c.title)));
      card.appendChild(html("p", { class: "kpi-d" }, esc(c.detail)));
      const d = html("details");
      d.appendChild(html("summary", null, "Source"));
      const ul = html("ul");
      (c.sources || []).forEach((s) => ul.appendChild(html("li", null, "results/" + esc(s))));
      d.appendChild(ul);
      card.appendChild(d);
      box.appendChild(card);
    });
    const sc = S.scale || {};
    $("#scale-strip").innerHTML = [
      `<span><b>${num(sc.chemicals)}</b> chemicals</span>`,
      `<span><b>${num(sc.wells)}</b> exposed wells</span>`,
      `<span><b>${num(sc.features)}</b> network features × <b>${(sc.divs || DIVS_FALLBACK).length}</b> days in vitro</span>`,
      `<span><b>${num(sc.recordings)}</b> chip recordings, <b>${num(sc.axons)}</b> sorted axons</span>`
    ].join("");
  }

  function bindValues(ctx) {
    $$("[data-bind]").forEach((el) => {
      let v = ctx;
      for (const p of el.getAttribute("data-bind").split(".")) v = v == null ? v : v[p];
      if (v != null) el.textContent = num(v);
    });
  }

  const METHOD_ORDER = ["neurotrajectory", "loglinear_interp", "analog_knn", "hill_per_endpoint", "context_mean", "zero"];
  const SHORT = { neurotrajectory: "Twin", loglinear_interp: "Log-linear", analog_knn: "Analog kNN",
    hill_per_endpoint: "Hill", context_mean: "Context mean", zero: "Zero effect" };
  const METHOD_COLOR = { neurotrajectory: "#0072B2", loglinear_interp: "#E69F00", analog_knn: "#009E73",
    hill_per_endpoint: "#CC79A7", context_mean: "#56B4E9", zero: "#8C8C8C" };

  function renderCvTable(S) {
    const T = S.trajectory, L = S.method_labels || {};
    const t = $("#tbl-cv");
    const head = "<thead><tr><th scope=\"col\">k</th>" + METHOD_ORDER.map((m) =>
      `<th scope="col" title="${esc(L[m] || m)}"><span class="swatch" style="background:${METHOD_COLOR[m]}"></span>${esc(SHORT[m] || L[m] || m)}</th>`).join("") +
      "<th scope=\"col\">Twin − best [95% CI]</th></tr></thead>";
    const rows = Object.keys(T.by_k).sort((a, b) => +a - +b).map((k) => {
      const e = T.by_k[k], c = e.curve_mae;
      const vals = METHOD_ORDER.map((m) => c[m]);
      const best = Math.min(...vals.filter((v) => v != null));
      const p = e.paired_vs_best;
      const diff = p && p.mean_diff != null
        ? `${fx(p.mean_diff, 3)} [${fx(p.ci95[0], 3)}, ${fx(p.ci95[1], 3)}]<br><span class="sub">${fx(p.rel_change_pct, 1)}% vs ${esc(SHORT[e.best_baseline] || e.best_baseline)}</span>`
        : "–";
      return `<tr><th scope="row">${k === "0" ? "0 (no dose)" : k}</th>` + vals.map((v) =>
        `<td class="${v === best ? "best" : ""}">${v == null ? "–" : v.toFixed(3)}</td>`).join("") + `<td>${diff}</td></tr>`;
    }).join("");
    t.insertAdjacentHTML("beforeend", head + "<tbody>" + rows + "</tbody>");
    const folds = Array.isArray(T.outer_folds) ? T.outer_folds.length : T.outer_folds;
    const seeds = Array.isArray(T.seeds) ? T.seeds.length : T.seeds;
    $("#cv-note").textContent = `${num(T.n_chemicals)} chemicals, ${folds}-fold chemical-level cross-validation, ` +
      `${seeds}-seed ensemble, ${T.designs_per_k} random designs of measured concentrations per k. Twin = NeuroTrajectory; Hill = Hill curve per endpoint. Bold: lowest error in the row. ` +
      `Difference = twin minus best baseline, paired over chemicals (negative = twin better).`;
  }

  function renderConfTable(S) {
    const R = S.conformal.results;
    const t = $("#tbl-conf");
    const head = "<thead><tr><th scope=\"col\">Nominal</th><th scope=\"col\">Parametric band</th><th scope=\"col\">Conformal band [95% CI]</th><th scope=\"col\">Mean width</th></tr></thead>";
    const rows = Object.keys(R).map((a) => {
      const e = R[a], k3 = e.k3;
      return `<tr><th scope="row">${pct(e.nominal, 0)}</th><td>${pct(k3.parametric.coverage)}</td>` +
        `<td class="best">${pct(k3.conformal.coverage)} [${pct(k3.conformal.ci95[0])}, ${pct(k3.conformal.ci95[1])}]</td>` +
        `<td>${k3.conformal.width.toFixed(2)}</td></tr>`;
    }).join("");
    t.insertAdjacentHTML("beforeend", head + "<tbody>" + rows + "</tbody>");
    const A = S.conformal.abstention;
    $("#conf-note").textContent = `Cross-conformal, Mondrian by cytotoxicity regime × k, ${num(S.conformal.n_chemicals)} chemicals. ` +
      `Abstention flags ${pct(A.abstention_rate)} of forecasts; their curve error is ${A.curve_mae_abstained.toFixed(2)} ` +
      `against ${A.curve_mae_retained.toFixed(2)} for the retained ones.`;
  }

  function renderProvenance(S) {
    const t = $("#tbl-hashes");
    t.innerHTML = "<thead><tr><th scope=\"col\">Input</th><th scope=\"col\">SHA-256</th></tr></thead><tbody>" +
      Object.entries(S.sources || {}).map(([k, v]) => `<tr><td>${esc(k.startsWith("data_bundle") ? k : "results/" + k)}</td><td>${esc(v)}</td></tr>`).join("") +
      "</tbody>";
    const when = (S.generated_utc || "").replace("T", " ").replace("Z", " UTC");
    $("#prov-line").innerHTML = `Data generated by <code>scripts/precompute_site.py</code> on ${esc(when)} from ` +
      `${Object.keys(S.sources || {}).length} hashed inputs; nothing on this page is typed by hand.`;
    $("#foot-gen").textContent = `Data generated ${when}.`;
  }

  /* ================================================================== explorer */
  const X = {
    index: null, summary: null, byId: new Map(), byName: new Map(), cache: new Map(),
    state: { id: null, k: 3, feat: "firing_rate_mean", reveal: false, wells: true, base: true },
    doc: null, loading: 0
  };

  function initExplorer(index, summary) {
    X.index = index; X.summary = summary;
    index.chemicals.forEach((c) => { X.byId.set(c.id, c); X.byName.set(c.chem.toLowerCase(), c); });

    // chemical search
    const input = $("#chem-input"), list = $("#chem-list");
    input.placeholder = `Search ${index.chemicals.length} chemicals…`;
    list.innerHTML = index.chemicals.slice().sort((a, b) => a.chem.localeCompare(b.chem))
      .map((c) => `<option value="${esc(c.chem)}"></option>`).join("");
    const pick = () => {
      const q = input.value.trim().toLowerCase();
      if (!q) return;
      let c = X.byName.get(q);
      if (!c) c = index.chemicals.find((x) => x.chem.toLowerCase().startsWith(q)) ||
                  index.chemicals.find((x) => x.chem.toLowerCase().includes(q));
      if (c) select(c.id); else setStatus(`No chemical matches “${esc(input.value)}”.`);
    };
    input.addEventListener("change", pick);
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); pick(); } });
    $("#btn-random").addEventListener("click", () => {
      const c = index.chemicals[Math.floor(Math.random() * index.chemicals.length)];
      select(c.id);
    });

    // k selector
    const seg = $("#k-seg");
    seg.innerHTML = index.ks.map((k) => `<button type="button" role="radio" aria-checked="false" data-k="${k}">${k}</button>`).join("");
    $$("button", seg).forEach((b) => b.addEventListener("click", () => { X.state.k = +b.dataset.k; render(); }));
    seg.addEventListener("keydown", (e) => {
      if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)) return;
      e.preventDefault();
      const ks = index.ks, i = ks.indexOf(X.state.k);
      const j = (i + (e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 1) + ks.length) % ks.length;
      X.state.k = ks[j]; render();
      const b = seg.querySelector(`[data-k="${ks[j]}"]`); if (b) b.focus();
    });

    // feature select
    const sel = $("#feat-select");
    sel.innerHTML = Object.entries(index.feature_groups).map(([g, fs]) =>
      `<optgroup label="${esc(g)}">` + fs.map((f) => `<option value="${f}">${esc(index.feature_labels[f] || f)}</option>`).join("") + "</optgroup>").join("");
    sel.addEventListener("change", () => { X.state.feat = sel.value; render(); });

    // options
    $("#btn-reveal").addEventListener("click", () => { X.state.reveal = !X.state.reveal; render(); });
    $("#opt-wells").addEventListener("change", (e) => { X.state.wells = e.target.checked; render(); });
    $("#opt-base").addEventListener("change", (e) => { X.state.base = e.target.checked; render(); });

    // examples, chosen by the report's stated percentile rule, plus the first abstained forecast
    const ex = $("#examples");
    ex.innerHTML = "<span class=\"ex-lab\">Examples:</span>";
    const chips = [];
    (index.examples || []).forEach((e) => {
      const c = X.byName.get(e.chemical.toLowerCase());
      if (c) chips.push({ id: c.id, name: c.chem, role: e.role, k: 3 });
    });
    const abst = index.chemicals.filter((c) => c.k["3"] && c.k["3"].a).sort((a, b) => a.chem.localeCompare(b.chem))[0];
    if (abst) chips.push({ id: abst.id, name: abst.chem, role: "the twin abstains", k: 3 });
    chips.forEach((ch) => {
      const b = html("button", { type: "button", class: "chip", "data-id": ch.id, "aria-pressed": "false" },
        `${esc(ch.name)}<small>${esc(ch.role)}</small>`);
      b.addEventListener("click", () => { X.state.k = ch.k; select(ch.id); });
      ex.appendChild(b);
    });
    X.chips = chips;

    // initial state from the URL (?chem=…&k=…&f=…), else the median example
    const P = new URLSearchParams(location.search);
    const kq = +P.get("k");
    if (index.ks.includes(kq)) X.state.k = kq;
    const fq = P.get("f");
    if (fq && index.features.includes(fq)) X.state.feat = fq;
    sel.value = X.state.feat;
    let start = P.get("chem") && X.byName.get(P.get("chem").toLowerCase());
    if (!start) {
      const med = chips.find((c) => /median/.test(c.role)) || chips[0];
      start = med ? X.byId.get(med.id) : index.chemicals[0];
    }
    select(start.id, true);

    // re-draw charts on resize (widths are measured, so text stays crisp and legible)
    if ("ResizeObserver" in window) {
      let raf = 0, lastW = 0;
      new ResizeObserver((ents) => {
        const w = Math.round(ents[0].contentRect.width);
        if (Math.abs(w - lastW) < 2) return;
        lastW = w;
        cancelAnimationFrame(raf);
        raf = requestAnimationFrame(() => { if (X.doc) drawCharts(); });
      }).observe($("#panels"));
    }
  }

  function setStatus(msg) { $("#ex-head").innerHTML = `<p class="status">${msg}</p>`; }

  async function select(id, initial) {
    X.state.id = id;
    X.state.reveal = false;
    const c = X.byId.get(id);
    $("#chem-input").value = c.chem;
    const ticket = ++X.loading;
    if (!X.cache.has(id)) setStatus(`Loading ${esc(c.chem)}…`);
    try {
      let d = X.cache.get(id);
      if (!d) { d = await getJSON(`data/chem/${id}.json`); X.cache.set(id, d); }
      if (ticket !== X.loading) return;
      X.doc = d;
      render(initial);
    } catch (e) {
      setStatus(`Could not load the data for ${esc(c.chem)} (${esc(e.message)}). Serve the folder with a static server; opening the file directly blocks data loading.`);
    }
  }

  /* ---------- data access */
  const V = (x) => (x == null ? null : x / 100);
  function kEntry() {
    const D = X.doc, k = String(X.state.k);
    return D.k[k] || D.k[String(Math.max(...Object.keys(D.k).map(Number)))];
  }
  function baselineCurve(D, E, f, d, xs) {
    const pts = E.ctx.map((j) => [D.levels[j], V(D.level_mean[f][d][j])]).filter((p) => p[1] != null).sort((a, b) => a[0] - b[0]);
    if (!pts.length) return xs.map(() => 0);
    if (pts.length === 1) return xs.map(() => pts[0][1]);
    return xs.map((x) => {
      if (x <= pts[0][0]) return pts[0][1];
      if (x >= pts[pts.length - 1][0]) return pts[pts.length - 1][1];
      let i = 1; while (pts[i][0] < x) i++;
      const [x0, y0] = pts[i - 1], [x1, y1] = pts[i];
      return y0 + (y1 - y0) * (x - x0) / (x1 - x0 || 1);
    });
  }
  function interpGrid(grid, arr, x) {
    if (x <= grid[0]) return arr[0];
    if (x >= grid[grid.length - 1]) return arr[arr.length - 1];
    let i = 1; while (grid[i] < x) i++;
    const w = (x - grid[i - 1]) / (grid[i] - grid[i - 1]);
    return arr[i - 1] * (1 - w) + arr[i] * w;
  }

  /* ---------- main render */
  function render() {
    const D = X.doc; if (!D) return;
    const st = X.state, E = kEntry(), I = X.index;
    const fi = I.features.indexOf(st.feat);
    X.fi = fi;

    // controls state
    $$("#k-seg button").forEach((b) => {
      const on = +b.dataset.k === st.k;
      b.setAttribute("aria-checked", String(on));
      b.tabIndex = on ? 0 : -1;
    });
    $$("#examples .chip").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.id === st.id)));
    const rb = $("#btn-reveal");
    rb.textContent = st.reveal ? "Hide the held-out wells" : "Reveal the held-out wells";
    rb.setAttribute("aria-pressed", String(st.reveal));

    // header
    const fold = I.folds[String(D.fold)] || {};
    const lab = D.label === "positive" ? "DNT positive" : D.label === "negative" ? "DNT negative" : "no reference label";
    const ctxC = E.ctx.map((j) => fmtConc(D.levels[j])).join(", ");
    const note = E.k_used < st.k ? ` (only ${D.levels.length} concentrations were tested, so k is capped at ${E.k_used})` : "";
    $("#ex-head").innerHTML =
      `<div class="ex-title"><h3>${esc(D.chem)}</h3>` +
      `<span class="pill">EPA reference: ${lab}</span></div>` +
      `<div class="ex-meta"><span><b>${D.levels.length}</b> tested concentrations, <b>${D.n_wells}</b> exposed wells</span>` +
      `<span><b class="tag tag-measured">Measured</b> and given to the twin: <b>${ctxC} µM</b>${note}</span>` +
      `<span>Model: fold ${D.fold} ensemble (${fold.n_models || 3} seeds), trained on ${fold.n_train_chemicals || "other"} chemicals, never on this one</span></div>`;

    drawCharts();
    renderReadout(D, E, I);
    renderTable(D, E, fi, I);

    const site = I.band && I.band.site_coverage_by_k && I.band.site_coverage_by_k[String(st.k)];
    $("#ex-fineprint").textContent =
      `Design: ${st.k} evenly spread measured concentration${st.k > 1 ? "s" : ""} (with k = 1 the twin sees only the lowest one). ` +
      `Band: cross-conformal 90 % interval calibrated on the chemicals of the other folds at the same k` +
      (site ? `; on this page it covers ${pct(site.conformal_coverage)} of all held-out well values at k = ${st.k} (target 90 %)` : "") +
      `. The report's intervals are additionally stratified by cytotoxicity. Units: ${I.units}.`;

    // shareable URL (debounced so fast clicking does not flood the history API)
    clearTimeout(X.urlTimer);
    X.urlTimer = setTimeout(() => {
      try {
        const u = new URL(location.href);
        u.searchParams.set("chem", D.chem); u.searchParams.set("k", String(st.k)); u.searchParams.set("f", st.feat);
        history.replaceState(null, "", u);
      } catch (e) { /* ignore */ }
    }, 400);
  }

  function drawCharts() {
    const D = X.doc, E = kEntry(), st = X.state, fi = X.fi, I = X.index;
    const divs = I.divs || DIVS_FALLBACK;
    const grid = D.grid;
    const ctxSet = new Set(E.ctx), heldSet = new Set(E.held);

    // shared y domain across the 4 DIV panels (includes held-out values so the axis does not jump on reveal)
    let lo = 0, hi = 0;
    const acc = (v) => { if (v != null && isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } };
    for (let d = 0; d < divs.length; d++) {
      const mu = E.mu[fi][d], hw = E.hw[fi][d];
      for (let i = 0; i < grid.length; i++) { acc(V(mu[i]) - V(hw[i])); acc(V(mu[i]) + V(hw[i])); }
      D.wells.y[fi][d].forEach((y) => acc(V(y)));
    }
    lo = Math.max(lo, -11.5); hi = Math.min(hi, 11.5);
    if (hi - lo < 2) { hi += 1; lo -= 1; }
    const pad = (hi - lo) * 0.06;
    const yDom = [lo - pad, hi + pad];

    const host = $("#panels");
    host.innerHTML = "";
    divs.forEach((div, d) => {
      const p = html("div", { class: "panel" });
      host.appendChild(p);
      drawPanel(p, { D, E, d, div, fi, grid, yDom, ctxSet, heldSet, st });
    });
    drawHeat(D, E, fi, divs, ctxSet, heldSet, st);
  }

  function drawPanel(p, o) {
    const { D, E, d, div, fi, grid, yDom, ctxSet, st } = o;
    const W = Math.max(150, Math.round(p.clientWidth || 280));
    const H = Math.round(Math.min(300, Math.max(190, W * 0.8)));
    const M = { l: 38, r: 10, t: 26, b: 34 };
    const x0 = grid[0], x1 = grid[grid.length - 1];
    const xs = (v) => M.l + (v - x0) / (x1 - x0) * (W - M.l - M.r);
    const ys = (v) => M.t + (yDom[1] - Math.max(yDom[0], Math.min(yDom[1], v))) / (yDom[1] - yDom[0]) * (H - M.t - M.b);
    const nHeld = E.held.length;
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img",
      "aria-label": `DIV ${div}: ${X.index.feature_labels[X.state.feat]} forecast from ${E.k_used} measured concentration(s); ` +
        `${nHeld} held-out concentration(s) ${st.reveal ? "revealed" : "hidden"}.` });
    const cid = `clip-${d}-${Math.random().toString(36).slice(2, 7)}`;
    const cp = svg("clipPath", { id: cid }, svg("defs", null, root));
    svg("rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b }, cp);

    // grid + axes
    const g = svg("g", { class: "c-axis" }, root);
    const step = niceStep(yDom[1] - yDom[0], H < 230 ? 4 : 5);
    for (let v = Math.ceil(yDom[0] / step) * step; v <= yDom[1] + 1e-9; v += step) {
      const y = ys(v);
      svg("line", { x1: M.l, x2: W - M.r, y1: y, y2: y, class: Math.abs(v) < 1e-9 ? "c-zero" : "c-grid" }, g);
      const t = svg("text", { x: M.l - 6, y: y + 3.5, "text-anchor": "end", class: "c-tick" }, g);
      t.textContent = (Math.abs(v) < 1e-9 ? "0" : (v < 0 ? "−" : "") + String(+Math.abs(v).toFixed(2)));
    }
    let decs = [];
    for (let n = Math.ceil(x0); n <= Math.floor(x1); n++) decs.push(n);
    if (decs.length > 4 && W < 230) decs = decs.filter((n, i) => i % 2 === 0);
    decs.forEach((n) => {
      const x = xs(n);
      svg("line", { x1: x, x2: x, y1: H - M.b, y2: H - M.b + 4, class: "c-tickline" }, g);
      const t = svg("text", { x, y: H - M.b + 16, "text-anchor": "middle", class: "c-tick" }, g);
      t.textContent = decadeLabel(n);
    });
    svg("line", { x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b, class: "c-base-axis" }, g);
    const xl = svg("text", { x: W - M.r, y: H - 4, "text-anchor": "end", class: "c-axlabel" }, g);
    xl.textContent = "µM";
    const tt = svg("text", { x: M.l, y: 16, class: "c-title" }, root);
    tt.textContent = `DIV ${div}`;
    if (E.metrics.abstain) {
      const ta = svg("text", { x: W - M.r, y: 16, "text-anchor": "end", class: "c-flag" }, root);
      ta.textContent = "abstains";
    }

    const plot = svg("g", { "clip-path": `url(#${cid})` }, root);
    // measured concentrations
    E.ctx.forEach((j) => svg("line", { x1: xs(D.levels[j]), x2: xs(D.levels[j]), y1: M.t, y2: H - M.b, class: "c-ctxline" }, plot));

    // band + forecast
    const mu = E.mu[fi][d].map(V), hw = E.hw[fi][d].map(V);
    let top = "", bot = "";
    grid.forEach((x, i) => { top += (i ? "L" : "M") + xs(x).toFixed(1) + "," + ys(mu[i] + hw[i]).toFixed(1); });
    for (let i = grid.length - 1; i >= 0; i--) bot += "L" + xs(grid[i]).toFixed(1) + "," + ys(mu[i] - hw[i]).toFixed(1);
    svg("path", { d: top + bot + "Z", class: "c-band" }, plot);
    if (st.base) {
      const bc = baselineCurve(D, E, fi, d, grid);
      svg("path", { d: grid.map((x, i) => (i ? "L" : "M") + xs(x).toFixed(1) + "," + ys(bc[i]).toFixed(1)).join(""), class: "c-baseline" }, plot);
    }
    svg("path", { d: grid.map((x, i) => (i ? "L" : "M") + xs(x).toFixed(1) + "," + ys(mu[i]).toFixed(1)).join(""), class: "c-pred" }, plot);

    // wells
    const lvW = D.wells.lv, yW = D.wells.y[fi][d];
    const jitter = (i) => ((i * 37) % 11 - 5) * 0.9;
    if (st.wells) {
      lvW.forEach((j, i) => {
        const y = V(yW[i]); if (y == null) return;
        const isCtx = ctxSet.has(j);
        if (!isCtx && !st.reveal) return;
        svg("circle", { cx: xs(D.levels[j]) + jitter(i), cy: ys(y), r: 2.1, class: isCtx ? "c-well-m" : "c-well-h" }, plot);
      });
    }
    // level means
    D.levels.forEach((lv, j) => {
      const m = V(D.level_mean[fi][d][j]); if (m == null) return;
      const isCtx = ctxSet.has(j);
      const cx = xs(lv), cy = ys(m);
      if (isCtx) {
        svg("circle", { cx, cy, r: 5, class: "c-mean-m" }, plot);
      } else if (st.reveal) {
        const s = 5.4;
        svg("path", { d: `M${cx},${cy - s}L${cx + s},${cy}L${cx},${cy + s}L${cx - s},${cy}Z`, class: "c-mean-h" }, plot);
      }
    });

    // hover guide
    const guide = svg("line", { y1: M.t, y2: H - M.b, class: "c-guide", visibility: "hidden" }, root);
    const dot = svg("circle", { r: 4, class: "c-guide-dot", visibility: "hidden" }, root);
    const hit = svg("rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b, fill: "transparent" }, root);
    const tip = $("#tooltip");
    const move = (ev) => {
      const r = root.getBoundingClientRect();
      const px = (ev.clientX - r.left) * (W / r.width);
      const xv = x0 + (px - M.l) / (W - M.l - M.r) * (x1 - x0);
      let gi = 0, best = Infinity;
      grid.forEach((x, i) => { const dd = Math.abs(x - xv); if (dd < best) { best = dd; gi = i; } });
      const gx = xs(grid[gi]);
      guide.setAttribute("x1", gx); guide.setAttribute("x2", gx); guide.setAttribute("visibility", "visible");
      dot.setAttribute("cx", gx); dot.setAttribute("cy", ys(mu[gi])); dot.setAttribute("visibility", "visible");
      let lj = -1, lb = 0.12;
      D.levels.forEach((lv, j) => { const dd = Math.abs(lv - xv); if (dd < lb) { lb = dd; lj = j; } });
      let meas = "";
      if (lj >= 0) {
        const m = V(D.level_mean[fi][d][lj]);
        const nW = D.level_n[lj];
        if (ctxSet.has(lj)) meas = `<br><b>Measured</b> (given): ${fx(m)} · ${nW} wells at ${fmtConc(D.levels[lj])} µM`;
        else if (st.reveal) meas = `<br><b>Measured</b> (held out): ${fx(m)} · ${nW} wells at ${fmtConc(D.levels[lj])} µM`;
        else meas = `<br>Held-out measurement at ${fmtConc(D.levels[lj])} µM: hidden`;
      }
      tip.innerHTML = `<b>${fmtConc(grid[gi])} µM · DIV ${div}</b><br><span class="t-pred"><b>Predicted</b> ${fx(mu[gi])} ` +
        `(90 % band ${fx(mu[gi] - hw[gi])} to ${fx(mu[gi] + hw[gi])})</span>${meas}`;
      tip.hidden = false;
      const tw = tip.offsetWidth, th = tip.offsetHeight;
      let tx = ev.clientX + 14, ty = ev.clientY + 14;
      if (tx + tw > window.innerWidth - 8) tx = ev.clientX - tw - 14;
      if (ty + th > window.innerHeight - 8) ty = ev.clientY - th - 14;
      tip.style.left = Math.max(8, tx) + "px"; tip.style.top = Math.max(8, ty) + "px";
    };
    const leave = () => { guide.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); tip.hidden = true; };
    hit.addEventListener("pointermove", move);
    hit.addEventListener("pointerdown", move);
    hit.addEventListener("pointerleave", leave);
    hit.addEventListener("pointercancel", leave);
    p.appendChild(root);
  }

  /* ---------- dose × day heatmaps */
  const RDBU = [[-10, [5, 48, 97]], [-7.5, [33, 102, 172]], [-5, [67, 147, 195]], [-2.5, [146, 197, 222]], [-1, [209, 229, 240]],
    [0, [247, 247, 247]], [1, [253, 219, 199]], [2.5, [244, 165, 130]], [5, [214, 96, 77]], [7.5, [178, 24, 43]], [10, [103, 0, 31]]];
  function rdbu(v) {
    const x = Math.max(-10, Math.min(10, v));
    let i = 1; while (i < RDBU.length - 1 && RDBU[i][0] < x) i++;
    const [a, ca] = RDBU[i - 1], [b, cb] = RDBU[i];
    const w = (x - a) / (b - a);
    const c = ca.map((u, k) => Math.round(u + (cb[k] - u) * w));
    return { fill: `rgb(${c[0]},${c[1]},${c[2]})`, dark: Math.abs(x) > 5.2 };
  }
  function drawHeat(D, E, fi, divs, ctxSet, heldSet, st) {
    const L = D.levels.length;
    const muHeld = new Map(E.held.map((j, h) => [j, h]));
    const make = (host, kind) => {
      host.innerHTML = "";
      const W = Math.max(220, Math.round(host.clientWidth || 320));
      const M = { l: 50, r: 6, t: 6, b: 34 };
      const cw = (W - M.l - M.r) / L, ch = 26;
      const H = M.t + ch * divs.length + M.b;
      const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img",
        "aria-label": kind === "meas" ? "Heatmap of measured level means by concentration and DIV" : "Heatmap of the twin forecast by concentration and DIV" });
      const showText = cw >= 34;
      divs.forEach((div, dr) => {
        const d = divs.length - 1 - dr;              // DIV 12 on top
        const y = M.t + dr * ch;
        const lab = svg("text", { x: M.l - 8, y: y + ch / 2 + 4, "text-anchor": "end", class: "c-tick" }, root);
        lab.textContent = `DIV ${divs[d]}`;
        for (let j = 0; j < L; j++) {
          const x = M.l + j * cw;
          let v = null, hidden = false;
          if (kind === "meas") {
            v = V(D.level_mean[fi][d][j]);
            if (heldSet.has(j) && !st.reveal) hidden = true;
          } else {
            v = muHeld.has(j) ? V(E.mu_held[fi][d][muHeld.get(j)]) : interpGrid(D.grid, E.mu[fi][d].map(V), D.levels[j]);
          }
          const rect = svg("rect", { x: x + 1, y: y + 1, width: Math.max(1, cw - 2), height: ch - 2, rx: 3 }, root);
          if (hidden) { rect.setAttribute("class", "c-cell-hidden"); }
          else if (v == null) { rect.setAttribute("class", "c-cell-na"); }
          else {
            const c = rdbu(v); rect.setAttribute("fill", c.fill);
            if (showText) {
              const t = svg("text", { x: x + cw / 2, y: y + ch / 2 + 4, "text-anchor": "middle", class: "c-cell-t", fill: c.dark ? "#fff" : "#1a1a1a" }, root);
              t.textContent = fx(v, 1);
            }
          }
          if (hidden && showText) {
            const t = svg("text", { x: x + cw / 2, y: y + ch / 2 + 4, "text-anchor": "middle", class: "c-cell-q" }, root);
            t.textContent = "?";
          }
        }
      });
      // measured-concentration boxes (as in the report's figure)
      E.ctx.forEach((j) => svg("rect", { x: M.l + j * cw + 0.5, y: M.t + 0.5, width: cw - 1, height: ch * divs.length - 1, rx: 4, class: "c-ctxbox" }, root));
      for (let j = 0; j < L; j++) {
        if (!showText && L > 6 && j % 2 === 1) continue;
        const t = svg("text", { x: M.l + j * cw + cw / 2, y: M.t + ch * divs.length + 15, "text-anchor": "middle", class: "c-tick" }, root);
        t.textContent = fmtConc(D.levels[j]);
      }
      const xl = svg("text", { x: W - M.r, y: H - 3, "text-anchor": "end", class: "c-axlabel" }, root);
      xl.textContent = "µM · boxed = measured, given to the twin";
      host.appendChild(root);
    };
    make($("#heat-meas"), "meas");
    make($("#heat-pred"), "pred");
    // colour bar
    const cb = $("#cbar");
    if (!cb.firstChild) {
      const W = 420, H = 40;
      const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H });
      const defs = svg("defs", null, root);
      const lg = svg("linearGradient", { id: "rdbu-grad", x1: "0", x2: "1", y1: "0", y2: "0" }, defs);
      RDBU.forEach(([v, c]) => svg("stop", { offset: ((v + 10) / 20).toFixed(3), "stop-color": `rgb(${c.join(",")})` }, lg));
      svg("rect", { x: 10, y: 4, width: W - 20, height: 10, rx: 3, fill: "url(#rdbu-grad)" }, root);
      [-10, -5, 0, 5, 10].forEach((v) => {
        const t = svg("text", { x: 10 + (v + 10) / 20 * (W - 20), y: 27, "text-anchor": "middle", class: "c-tick" }, root);
        t.textContent = (v < 0 ? "−" : "") + Math.abs(v);
      });
      const t = svg("text", { x: W / 2, y: 38, "text-anchor": "middle", class: "c-axlabel" }, root);
      t.textContent = "effect vs vehicle (robust SD; blue = below vehicle, red = above; clipped at ±10)";
      root.setAttribute("height", "44"); root.setAttribute("viewBox", `0 0 ${W} 44`);
      cb.appendChild(root);
    }
  }

  /* ---------- readout cards */
  function verdict(cls, text) { return `<span class="verdict ${cls}">${text}</span>`; }
  function meter(value, mark, max, cls) {
    const w = Math.max(0, Math.min(1, value / max)) * 100, m = Math.max(0, Math.min(1, mark / max)) * 100;
    return `<div class="meter" aria-hidden="true"><div class="meter-fill ${cls || ""}" style="width:${w.toFixed(1)}%"></div>` +
      `<div class="meter-mark" style="left:calc(${m.toFixed(1)}% - 1px)"></div></div>`;
  }

  function renderReadout(D, E, I) {
    const st = X.state, M = E.metrics, k = String(st.k);
    // 1. confidence
    const site = (I.abstention.site_design_by_k || {})[k];
    const mx = Math.max(M.threshold * 2, M.epi * 1.15);
    $("#r-trust").innerHTML =
      `<h4>Twin confidence</h4>` +
      `<p class="big">${M.abstain ? verdict("v-warn", "Abstains: I don’t know") : verdict("v-ok", "Confident enough to use")}</p>` +
      `<p>${M.abstain
        ? "The ensemble members disagree more than for 90 % of the chemicals of the other folds. Measure another concentration before relying on this forecast."
        : "Ensemble disagreement is below the cross-fitted threshold (90th percentile of the other folds)."}</p>` +
      meter(M.epi, M.threshold, mx, M.abstain ? "warn" : "") +
      `<div class="meter-scale"><span>epistemic score ${M.epi.toFixed(3)}</span><span>threshold ${M.threshold.toFixed(3)}</span></div>` +
      (site ? `<p class="fine">At k = ${k}, ${pct(site.rate)} of the chemicals on this page abstain; their forecast error is ${site.mae_abstained.toFixed(2)} vs ${site.mae_retained.toFixed(2)} for the rest.</p>` : "");

    // 2. score (after reveal)
    const C = M.curve_mae;
    const sd = X.summary.site_design_check && X.summary.site_design_check.by_k && X.summary.site_design_check.by_k[k];
    if (!st.reveal) {
      $("#r-score").innerHTML = `<h4>Forecast accuracy</h4>` +
        `<p class="big">${verdict("v-neutral", `${E.held.length} held-out concentrations hidden`)}</p>` +
        `<p>The twin never saw these wells. Reveal them to score the forecast against two baselines that see the same measured wells.</p>`;
    } else {
      const rows = [["NeuroTrajectory (twin)", C.model], ["Log-linear interpolation", C.interp], ["Analog kNN", C.knn]];
      const best = Math.min(...rows.map((r) => r[1]));
      const cov = M.coverage90;
      $("#r-score").innerHTML = `<h4>Forecast accuracy</h4>` +
        `<p class="big">${C.model <= Math.min(C.interp, C.knn) ? verdict("v-ok", "Twin is most accurate here") : verdict("v-warn", "A baseline wins on this chemical")}</p>` +
        `<table class="mini"><thead><tr><th scope="col">Method</th><th scope="col">Curve error</th></tr></thead><tbody>` +
        rows.map((r) => `<tr class="${r[1] === best ? "best" : ""}"><td>${r[0]}</td><td>${r[1].toFixed(3)}</td></tr>`).join("") +
        `</tbody></table>` +
        `<p class="fine">Mean |forecast − measured mean| over all 17 features × 4 DIV at the held-out concentrations (robust-SD units). ` +
        `${pct(cov, 0)} of the held-out well values fall inside the 90 % band.` +
        (sd ? ` Across all ${sd.n_chemicals} chemicals at k = ${k}, the twin beats log-linear interpolation on ${pct(sd.frac_better_than_interp, 0)} of them.` : "") + `</p>`;
    }

    // 3. hazard
    const Hz = D.hazard, HR = I.hazard && I.hazard.site_rates_on_reference_chemicals;
    if (!Hz) {
      $("#r-hazard").innerHTML = `<h4>Hazard call</h4><p>Not available for this chemical.</p>`;
    } else {
      const p = Hz.prob[k], call = Hz.call[k], pf = Hz.prob.full;
      const r = HR && HR[k];
      const refTxt = D.label === "positive" ? "DNT positive" : D.label === "negative" ? "DNT negative" : "none (not an EPA reference chemical)";
      let agree = "";
      if (D.label === "positive" || D.label === "negative") {
        const ok = (D.label === "positive") === call;
        agree = ok ? verdict("v-ok", "agrees with reference") : verdict("v-warn", "disagrees with reference");
      }
      const valid = (I.hazard.validated || []).includes(k);
      const rateTxt = r ? `flags ${r.tp} of ${r.tp + r.fn} reference positives and clears ${r.tn} of ${r.tn + r.fp} reference negatives` : "";
      $("#r-hazard").innerHTML = `<h4>Hazard call · developmental neurotoxicity</h4>` +
        `<p class="big">${call ? verdict("v-bad", "Hazard flagged") : verdict("v-ok", "No hazard flagged")} <span class="fine">p = ${p.toFixed(2)}</span></p>` +
        meter(p, 0.5, 1, call ? "bad" : "ok") +
        `<div class="meter-scale"><span>0</span><span>call threshold 0.5</span><span>1</span></div>` +
        `<p>EPA reference label: <b>${refTxt}</b> ${agree}</p>` +
        `<p>From all measured concentrations: p = ${pf.toFixed(2)} (${Hz.call.full ? "flagged" : "not flagged"}).</p>` +
        `<p class="fine">${valid
          ? `Validated protocol (k = 3, report R4). On the reference chemicals this rule ${rateTxt}.`
          : `Exploratory at k = ${k}: the report validates k = 3. At this k the same rule ${rateTxt}.`} ` +
        `Classifier trained only on reference chemicals of the other folds. A research output, not a regulatory call.</p>`;
    }

    // 4. DIV 7 gate
    const G = D.gate;
    if (!G) {
      $("#r-gate").innerHTML = `<h4>DIV 7 gate</h4>` +
        `<p class="big">${verdict("v-neutral", "Not evaluated")}</p>` +
        `<p>This chemical is in fold 0, used during development. The preregistered early-exit analysis covers the chemicals of folds 1–4 only.</p>`;
    } else {
      const row = (name, sub, e) => {
        const outcome = e.exit
          ? (G.reference_active ? verdict("v-bad", "Stop at DIV 7 · would miss an active") : verdict("v-ok", "Stop at DIV 7 · correct"))
          : (G.reference_active ? verdict("v-ok", "Continue · correct") : verdict("v-neutral", "Continue · no saving"));
        return `<div class="gate-row"><span class="gname">${name}</span>${outcome}` +
          `<span class="gsub">${sub}: score ${e.score.toFixed(2)} vs threshold ${e.threshold.toFixed(2)} (stops if below)</span></div>`;
      };
      $("#r-gate").innerHTML = `<h4>DIV 7 gate · stop the culture early?</h4>` +
        row("Throughput gate", "Persistence, DIV 7 carried forward", G.persistence) +
        row("Safety gate", "Twin forecast of DIV 9/12, upper 90 % band", G.twin_upper90) +
        `<p>Complete assay (DIV 5–12): <b>${G.reference_active ? "active" : "inactive"}</b> (score ${G.reference_score.toFixed(2)}; ≥ 1 = active).</p>` +
        `<p class="fine">Gates see only the DIV 5 and 7 wells of every tested concentration, so they do not depend on k. ` +
        `Each fold's threshold keeps 95 % of the active chemicals of the other folds.</p>`;
    }
  }

  /* ---------- accessible table */
  function renderTable(D, E, fi, I) {
    const st = X.state, divs = I.divs || DIVS_FALLBACK;
    const ctxSet = new Set(E.ctx), heldIdx = new Map(E.held.map((j, h) => [j, h]));
    let head = `<caption>${esc(I.feature_labels[st.feat])}, ${esc(D.chem)}, k = ${E.k_used}. ` +
      `Measured = mean of replicate wells; predicted = twin forecast [90 % band].</caption><thead><tr>` +
      `<th scope="col">Concentration (µM)</th><th scope="col">Role</th><th scope="col">Wells</th>` +
      divs.map((d) => `<th scope="col">DIV ${d} measured</th><th scope="col">DIV ${d} predicted</th>`).join("") + "</tr></thead><tbody>";
    const rows = D.levels.map((lv, j) => {
      const isCtx = ctxSet.has(j);
      const cells = divs.map((_, d) => {
        const m = V(D.level_mean[fi][d][j]);
        const meas = isCtx || st.reveal ? fx(m) : "hidden";
        let pred = "–";
        if (heldIdx.has(j)) {
          const h = heldIdx.get(j), mu = V(E.mu_held[fi][d][h]), hw = V(E.hw_held[fi][d][h]);
          pred = `${fx(mu)} [${fx(mu - hw)}, ${fx(mu + hw)}]`;
        }
        return `<td>${meas}</td><td>${pred}</td>`;
      }).join("");
      return `<tr><th scope="row">${fmtConc(lv)}</th><td>${isCtx ? "measured, given" : "held out"}</td><td>${D.level_n[j]}</td>${cells}</tr>`;
    }).join("");
    $("#ex-table").innerHTML = head + rows + "</tbody>";
  }

  /* ================================================================== boot */
  async function boot() {
    setupTheme(); setupNav(); setupLinks(); setupLightbox(); setupCopy(); setupVideo();
    let S, I;
    try {
      [S, I] = await Promise.all([getJSON("data/summary.json"), getJSON("data/index.json")]);
    } catch (e) {
      const msg = `Could not load the data files (${esc(e.message)}). Serve this folder with a static server, e.g. <code>python -m http.server -d site</code>; opening index.html directly blocks data loading.`;
      $("#kpis").innerHTML = `<p class="note">${msg}</p>`;
      setStatus(msg);
      return;
    }
    const safe = (fn, ...a) => { try { fn(...a); } catch (e) { console.error(e); } };
    safe(renderKPIs, S);
    safe(bindValues, { scale: S.scale });
    safe(renderCvTable, S);
    safe(renderConfTable, S);
    safe(renderProvenance, S);
    safe(initExplorer, I, S);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
