/* NeuroChip Twin static explorer.
 * Every number shown on the page is read at runtime from data/*.json, which scripts/precompute_site.py
 * builds from data_bundle/ (forecasts) and results/*.json (validation). No backend. */
(() => {
  'use strict';

  // ------------------------------------------------------------------ palette (validated, light surface)
  const C = {
    model: '#2a78d6', band: 'rgba(42,120,214,0.20)', measured: '#1b1b19', revealed: '#d63a36',
    interp: '#8f8e89', knn: '#5f5e5a', hill: '#b3b2ad', ff: '#2a78d6', fb: '#eb6834',
    grid: '#ebeae5', axis: '#a9a8a2', zero: '#b9b8b2', ink: '#1b1b19', ink2: '#4a4945', ink3: '#6b6a65', surface: '#ffffff',
  };
  const METHOD = {
    neurotrajectory: { name: 'NeuroTrajectory', color: C.model, dash: 'solid', symbol: 'circle', width: 3 },
    analog_knn: { name: 'Analog kNN', color: C.knn, dash: 'dot', symbol: 'square', width: 2 },
    loglinear_interp: { name: 'Log-linear interpolation', color: C.interp, dash: 'dash', symbol: 'diamond', width: 2 },
    hill_per_endpoint: { name: 'Hill fit per endpoint', color: C.hill, dash: 'dashdot', symbol: 'triangle-up', width: 2 },
  };
  const DIVERGING = [[0, '#104281'], [0.2, '#3987e5'], [0.42, '#cde2fb'], [0.5, '#f0efec'],
    [0.58, '#f9d3cf'], [0.8, '#e34948'], [1, '#8f1d1c']];
  const FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';

  // ------------------------------------------------------------------ state + helpers
  const S = {
    meta: null, sum: null, byId: new Map(), byName: new Map(), cache: new Map(),
    doc: null, id: null, k: 3, feats: [], revealed: false, wells: true, interp: true,
    heatJ: 0, base: 'i', geom: null, token: 0, scatterBound: false,
  };
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const num = (v) => v !== null && v !== undefined && Number.isFinite(Number(v));
  const fx = (v, n = 2) => (num(v) ? Number(v).toFixed(n) : '–');
  const pct = (v, n = 1) => (num(v) ? (100 * v).toFixed(n) + '%' : '–');
  const signed = (v, n = 3) => (num(v) ? (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(n) : '–');
  const ci = (a, n = 3) => (Array.isArray(a) ? `${signed(a[0], n)} to ${signed(a[1], n)}` : '–');
  const SC = () => S.meta.scale;
  const dq = (v) => (v === null || v === undefined ? null : v / SC());
  const flabel = (f) => (S.meta.feature_labels && S.meta.feature_labels[f]) || f.replace(/_/g, ' ');
  function uM(logc) {
    const v = Math.pow(10, logc);
    if (v >= 100) return v.toFixed(0);
    if (v >= 10) return v.toFixed(1).replace(/\.0$/, '');
    return String(Number(v.toPrecision(2)));
  }
  const labelText = (l) => (l === 'positive' ? 'positive (reference developmental neurotoxicant)'
    : l === 'negative' ? 'negative (reference non-toxicant)' : 'not in the reference set');

  async function getJSON(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
    return r.json();
  }
  function plot(el, data, layout, config) {
    if (typeof el === 'string') el = $(el);
    if (!el) return;
    if (!window.Plotly) {
      el.innerHTML = '<p class="error-box">Charts need Plotly from cdn.jsdelivr.net, which could not be loaded. The tables on this page are still complete.</p>';
      return;
    }
    const base = {
      paper_bgcolor: C.surface, plot_bgcolor: C.surface, font: { family: FONT, size: 11, color: C.ink2 },
      hoverlabel: { font: { family: FONT, size: 12 }, bgcolor: '#ffffff', bordercolor: C.axis },
    };
    return Plotly.react(el, data, Object.assign(base, layout),
      Object.assign({ displayModeBar: false, responsive: true }, config || {}));
  }
  function axis(extra) {
    return Object.assign({ gridcolor: C.grid, linecolor: C.axis, showline: true, zeroline: false,
      tickfont: { size: 10, color: C.ink2 }, fixedrange: true, automargin: true }, extra || {});
  }
  function makeSeg(el, items, current, onPick, fmt) {
    el.innerHTML = '';
    items.forEach((it) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.setAttribute('role', 'radio');
      b.dataset.v = String(it);
      b.textContent = fmt ? fmt(it) : String(it);
      el.appendChild(b);
    });
    const sync = (v) => [...el.children].forEach((b) => {
      const on = b.dataset.v === String(v);
      b.setAttribute('aria-checked', on ? 'true' : 'false');
      b.tabIndex = on ? 0 : -1;
    });
    sync(current);
    el.onclick = (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      sync(b.dataset.v);
      onPick(b.dataset.v);
    };
    el.onkeydown = (e) => {
      if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) return;
      const bs = [...el.children];
      const i = bs.findIndex((b) => b.getAttribute('aria-checked') === 'true');
      const j = (i + (e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -1 : 1) + bs.length) % bs.length;
      e.preventDefault();
      bs[j].focus();
      bs[j].click();
    };
    return sync;
  }

  // Wide tables scroll inside .table-wrap; their caption is shown above the scroller so it wraps to the
  // viewport on phones, and kept as a visually hidden <caption> for screen readers.
  function tableCaption(tableId, text) {
    const wrap = $(tableId).parentElement;
    let p = wrap.previousElementSibling;
    if (!p || !p.classList.contains('table-cap')) {
      p = document.createElement('p');
      p.className = 'table-cap';
      p.setAttribute('aria-hidden', 'true');
      wrap.parentElement.insertBefore(p, wrap);
    }
    p.textContent = text;
    return `<caption class="sr-only">${esc(text)}</caption>`;
  }

  // ------------------------------------------------------------------ boot
  async function init() {
    try {
      const [meta, sum] = await Promise.all([getJSON('data/index.json'), getJSON('data/summary.json')]);
      S.meta = meta;
      S.sum = sum;
    } catch (err) {
      const msg = `Could not load the site data (${esc(err.message)}). Open this page through a static server, for example <code>python -m http.server -d site 8000</code>, not as a local file, and make sure <code>scripts/precompute_site.py</code> has been run.`;
      $('status').innerHTML = `<span class="error-box">${msg}</span>`;
      $('hero-stats').innerHTML = `<div class="stat skeleton"><span class="stat-l">${msg}</span></div>`;
      return;
    }
    fillStatic();
    renderHero();
    setupExplorer();
    // below-the-fold sections render independently so one failure never blanks the page
    [renderResults, renderChip].forEach((fn) => { try { fn(); } catch (e) { console.error(e); } });
  }

  function fillStatic() {
    const T = S.sum.trajectory;
    document.querySelectorAll('[data-fill="designs_per_k"]').forEach((el) => { el.textContent = String(T.designs_per_k); });
    const src = Object.keys(S.sum.sources || {});
    const man = S.sum.manifest || {};
    $('foot-prov').textContent = `Site data generated by ${S.sum.generated_by} (${S.sum.generated_utc}) from ${src.join(', ')}` +
      (num(man.n_wells) ? `; bundle: ${man.n_chemicals} chemicals, ${man.n_wells} wells.` : '.');
  }

  // ------------------------------------------------------------------ hero
  function renderHero() {
    const T = S.sum.trajectory;
    const K = '3';
    const bk = T.by_k[K];
    const pv = bk.paired_vs_best;
    const conf = S.sum.conformal.results['alpha_0.1'];
    const c3 = conf && conf.k3;
    const ab = S.sum.conformal.abstention;
    const better = pv.rel_change_pct < 0;
    const tiles = [
      { v: String(T.n_chemicals), l: 'unseen chemicals, each forecast only by models that never saw it',
        s: `${T.outer_folds.length}-fold cross-validation by chemical` },
      { v: `${Math.abs(pv.rel_change_pct).toFixed(1)}<span class="unit">%</span>`,
        l: `${better ? 'lower' : 'higher'} forecast error than the best baseline (${METHOD[bk.best_baseline] ? METHOD[bk.best_baseline].name.toLowerCase() : bk.best_baseline}) from ${K} measured concentrations`,
        s: `paired 95% CI ${ci(pv.ci95)}; better on ${pct(pv.frac_chem_improved, 0)} of chemicals` },
      c3 && { v: `${(100 * c3.conformal.coverage).toFixed(1)}<span class="unit">%</span>`,
        l: `of held-out wells inside the conformal ${Math.round(100 * conf.nominal)}% band (k = ${K})`,
        s: `95% CI ${pct(c3.conformal.ci95[0])}–${pct(c3.conformal.ci95[1])}; uncalibrated ${pct(c3.parametric.coverage)}` },
      { v: `${fx(ab.curve_mae_retained)} <span class="unit">vs</span> ${fx(ab.curve_mae_abstained)}`,
        l: 'error when it answers vs when it abstains',
        s: `abstains on ${pct(ab.abstention_rate)} of forecasts (epistemic score above the 90th percentile)` },
    ].filter(Boolean);
    $('hero-stats').innerHTML = tiles.map((t) =>
      `<div class="stat"><span class="stat-v">${t.v}</span><span class="stat-l">${esc(t.l)}</span><span class="stat-s">${esc(t.s)}</span></div>`).join('');
    $('hero-note').textContent = `Error = mean absolute difference between the forecast and the replicate-well mean at each held-out concentration, in robust-SD units of the vehicle wells (${T.n_chemicals} chemicals, ${T.seeds.length}-model ensembles, ${T.designs_per_k} random designs per k). Source: results/trajectory_cv.json and results/conformal_r7.json.`;
  }

  // ------------------------------------------------------------------ explorer setup
  function setupExplorer() {
    const M = S.meta;
    M.chemicals.forEach((c) => { S.byId.set(c.id, c); S.byName.set(c.chem.toLowerCase(), c); });
    $('chem-list').innerHTML = M.chemicals.map((c) => `<option value="${esc(c.chem)}"></option>`).join('');
    $('chem-help').textContent = `${M.chemicals.length} chemicals from the EPA NFA; type to search, then pick one.`;

    const params = new URLSearchParams(location.search);
    const kq = Number(params.get('k'));
    S.k = M.ks.includes(kq) ? kq : (M.ks.includes(3) ? 3 : M.ks[M.ks.length - 1]);
    S.syncK = makeSeg($('k-seg'), M.ks, S.k, (v) => { S.k = Number(v); render(); });

    setupFeatures();
    $('opt-wells').onchange = (e) => { S.wells = e.target.checked; renderPanels(); };
    $('opt-interp').onchange = (e) => { S.interp = e.target.checked; renderPanels(); };
    $('btn-reveal').onclick = () => { S.revealed = !S.revealed; renderPanels(); renderReveal(); renderHeat(); };
    $('btn-random').onclick = () => {
      let c;
      do { c = M.chemicals[Math.floor(Math.random() * M.chemicals.length)]; } while (M.chemicals.length > 1 && c.id === S.id);
      selectChem(c.id);
    };
    const inp = $('chem-input');
    const tryPick = (loose) => {
      const v = inp.value.trim().toLowerCase();
      if (!v) return;
      let c = S.byName.get(v);
      if (!c && loose) c = M.chemicals.find((x) => x.chem.toLowerCase().includes(v));
      if (c && c.id !== S.id) selectChem(c.id);
    };
    inp.addEventListener('change', () => tryPick(false));
    inp.addEventListener('input', () => tryPick(false));
    inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); tryPick(true); } });

    S.syncBase = makeSeg($('base-seg'), ['i', 'n'], S.base, (v) => { S.base = v; renderScatter(); },
      (v) => (v === 'i' ? 'vs log-linear interpolation' : 'vs analog kNN'));
    setupExamples();

    let t = null;
    window.addEventListener('resize', () => {
      clearTimeout(t);
      t = setTimeout(() => { if (S.doc && geometry() !== S.geom) renderPanels(); }, 180);
    });

    const q = params.get('chem');
    const fromUrl = q && (S.byId.get(q) || S.byName.get(q.toLowerCase()));
    const dflt = S.byName.get('deltamethrin') || M.chemicals[0];
    selectChem((fromUrl || dflt).id);
  }

  function setupFeatures() {
    const M = S.meta;
    S.feats = M.default_features.filter((f) => M.features.includes(f));
    const grid = $('feat-grid');
    grid.insertAdjacentHTML('beforeend', M.features.map((f, i) =>
      `<label class="chk"><input type="checkbox" value="${esc(f)}" id="feat-${i}"> ${esc(flabel(f))}</label>`).join(''));
    const presets = [['Key features', M.default_features]]
      .concat(Object.entries(M.feature_groups || {}))
      .concat([[`All ${M.features.length}`, M.features]]);
    $('feat-presets').innerHTML = presets.map(([n], i) => `<button type="button" class="chip-btn" data-i="${i}">${esc(n)}</button>`).join('');
    $('feat-presets').onclick = (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      S.feats = presets[Number(b.dataset.i)][1].slice();
      syncFeat();
      renderPanels();
    };
    grid.onchange = () => {
      const on = [...grid.querySelectorAll('input:checked')].map((x) => x.value);
      if (!on.length) { syncFeat(); return; }            // keep at least one feature
      S.feats = M.features.filter((f) => on.includes(f));
      syncFeat();
      renderPanels();
    };
    syncFeat();
  }
  function syncFeat() {
    const M = S.meta;
    S.feats = M.features.filter((f) => S.feats.includes(f));
    $('feat-grid').querySelectorAll('input').forEach((x) => { x.checked = S.feats.includes(x.value); });
    $('feat-count').textContent = `${S.feats.length} of ${M.features.length}`;
  }

  function setupExamples() {
    const M = S.meta;
    const k = String(M.ks.includes(3) ? 3 : M.ks[M.ks.length - 1]);
    const rows = M.chemicals.map((c) => {
      const m = c.k[k];
      return { c, m, gain: Math.min(m.i, m.n) - m.m };
    }).filter((r) => num(r.m.m) && num(r.m.i) && num(r.m.n));
    if (!rows.length) return;
    const confident = rows.filter((r) => !r.m.a);
    const sorted = confident.slice().sort((a, b) => a.gain - b.gain);
    const pick = [];
    const add = (tag, r) => { if (r && !pick.some((p) => p.r.c.id === r.c.id)) pick.push({ tag, r }); };
    add('Demo default', rows.find((r) => r.c.chem.toLowerCase() === 'deltamethrin'));
    add('Largest gain', sorted[sorted.length - 1]);
    add('Typical', sorted[Math.floor(sorted.length / 2)]);
    add('Model loses', sorted[0]);
    add('Model abstains', rows.filter((r) => r.m.a).sort((a, b) => b.m.m - a.m.m)[0]);
    $('examples').innerHTML = `<span class="lbl">Examples (k = ${k}):</span>` + pick.map(({ tag, r }) =>
      `<button type="button" class="chip-btn" data-id="${r.c.id}" data-k="${k}" title="${esc(r.c.chem)}"><b>${esc(tag)}:</b> ${esc(r.c.chem)}</button>`).join('');
    $('examples').onclick = (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      S.k = Number(b.dataset.k);
      S.syncK(S.k);
      selectChem(b.dataset.id);
    };
  }

  async function selectChem(id) {
    const token = ++S.token;
    const c = S.byId.get(id);
    if (!c) return;
    $('status').textContent = `Loading ${c.chem}…`;
    let doc = S.cache.get(id);
    if (!doc) {
      try {
        doc = await getJSON(`data/chem/${id}.json`);
      } catch (err) {
        if (token === S.token) $('status').innerHTML = `<span class="error-box">Could not load ${esc(c.chem)}: ${esc(err.message)}</span>`;
        return;
      }
      S.cache.set(id, doc);
    }
    if (token !== S.token) return;
    S.doc = doc;
    S.id = id;
    S.revealed = false;
    S.heatJ = -1;
    $('chem-input').value = doc.chem;
    render();
  }

  // ------------------------------------------------------------------ explorer render
  const kdata = () => S.doc.k[String(S.k)];
  function render() {
    if (!S.doc) return;
    const K = kdata();
    if (S.heatJ < 0 || S.heatJ >= K.held.length) S.heatJ = K.held.length - 1;
    try { history.replaceState(null, '', `?chem=${S.id}&k=${S.k}${location.hash}`); } catch (e) { /* file:// or sandbox */ }
    renderStatus();
    renderPanels();
    renderReveal();
    renderHeat();
    renderScatter();
  }

  function renderStatus() {
    const d = S.doc;
    const K = kdata();
    const f = S.meta.folds[String(d.fold)] || {};
    const ctx = K.ctx.map((i) => uM(d.levels[i]));
    const extra = K.k_used === 1 ? ' Only the lowest concentration is given, so the forecast must extrapolate upward: the hardest case.' : '';
    $('status').innerHTML = `<strong>${esc(d.chem)}</strong> <span class="pill">held-out: fold ${d.fold} ensemble, trained without this chemical</span><br>` +
      `EPA DNT reference: ${esc(labelText(d.label))}. Tested at ${d.levels.length} concentrations (${d.n_wells} wells). ` +
      `The model is given the wells at ${K.k_used} of them (${ctx.join(', ')} µM) and forecasts the other ${K.held.length}.${extra}` +
      (num(f.n_train_chemicals) ? ` Fold ${d.fold} models were trained on ${f.n_train_chemicals} other chemicals.` : '');
    const m = K.metrics;
    const icon = (warn) => warn
      ? '<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3 L22 20 H2 Z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M12 10 v4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="12" cy="17" r="1.2" fill="currentColor"/></svg>'
      : '<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/><path d="M7.5 12.5 l3 3 l6-6.5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    if (!num(m.epi) || !num(m.threshold)) {
      $('abstain').innerHTML = '';
    } else if (m.abstain) {
      $('abstain').innerHTML = `<div class="badge warn" role="status">${icon(true)}<div><strong>Abstain: low confidence</strong>` +
        `The ensemble members disagree (epistemic score ${fx(m.epi, 3)} &gt; threshold ${fx(m.threshold, 3)}, the 90th percentile of the other folds). ` +
        'NeuroChip Twin would flag this forecast and ask for another measured concentration.</div></div>';
    } else {
      $('abstain').innerHTML = `<div class="badge ok" role="status">${icon(false)}<div><strong>Confident enough to answer</strong>` +
        `Epistemic score ${fx(m.epi, 3)} ≤ threshold ${fx(m.threshold, 3)} (90th percentile of the other folds).</div></div>`;
    }
  }

  function geometry() {
    const w = $('panels').clientWidth || window.innerWidth;
    return w < 620 ? 'narrow' : 'wide';
  }
  function xTicks(x0, x1) {
    let t = [];
    for (let v = Math.ceil(x0); v <= Math.floor(x1); v++) t.push(v);
    if (t.length > 5) t = t.filter((v, i) => i % 2 === 0);
    if (t.length < 2) t = [x0 + 0.3, x1 - 0.3].map((v) => Math.round(v * 2) / 2);
    return { tickvals: t, ticktext: t.map(uM) };
  }
  function interpCurve(d, K, f, di, xs) {
    const lm = d.level_mean[f][di];
    const pts = K.ctx.map((i) => [d.levels[i], lm[i]]).filter((p) => p[1] !== null).sort((a, b) => a[0] - b[0]);
    if (!pts.length) return xs.map(() => 0);
    return xs.map((x) => {
      if (x <= pts[0][0]) return pts[0][1] / SC();
      if (x >= pts[pts.length - 1][0]) return pts[pts.length - 1][1] / SC();
      for (let j = 1; j < pts.length; j++) {
        if (x <= pts[j][0]) {
          const [xa, ya] = pts[j - 1];
          const [xb, yb] = pts[j];
          const w = (x - xa) / (xb - xa);
          return (ya * (1 - w) + yb * w) / SC();
        }
      }
      return 0;
    });
  }

  function renderPanels() {
    const d = S.doc;
    if (!d) return;
    const K = kdata();
    const geom = geometry();
    S.geom = geom;
    const box = $('panels');
    const want = S.feats.join('|') + '#' + geom;
    if (box.dataset.key !== want) {
      box.innerHTML = S.feats.map((f) => `<div class="panel"><h4>${esc(flabel(f))} <span>${esc(f)}</span></h4>` +
        `<div class="plot" id="p-${esc(f)}" role="img" aria-label="${esc(flabel(f))}: forecast and wells at DIV ${S.meta.divs.join(', ')}"></div></div>`).join('');
      box.dataset.key = want;
    }
    const fi = Object.fromEntries(S.meta.features.map((f, i) => [f, i]));
    const ctxSet = new Set(K.ctx);
    const heldSet = new Set(K.held);
    const grid = d.grid;
    const x0 = grid[0];
    const x1 = grid[grid.length - 1];
    const ticks = xTicks(x0, x1);
    const gridLab = grid.map(uM);
    const narrow = geom === 'narrow';
    const doms = narrow
      ? [[[0, 0.46], [0.6, 1]], [[0.54, 1], [0.6, 1]], [[0, 0.46], [0, 0.4]], [[0.54, 1], [0, 0.4]]]
      : [0, 1, 2, 3].map((i) => [[i * 0.255, i * 0.255 + 0.235], [0, 1]]);

    S.feats.forEach((fname) => {
      const f = fi[fname];
      const traces = [];
      let lo = 0;
      let hi = 0;
      const layout = {
        height: narrow ? 420 : 215, margin: { l: 38, r: 6, t: 22, b: 26 }, showlegend: false,
        hovermode: 'closest', dragmode: false, annotations: [],
      };
      S.meta.divs.forEach((div, di) => {
        const s = di === 0 ? '' : String(di + 1);
        const xa = 'x' + s;
        const ya = 'y' + s;
        const mu = K.mu[f][di].map(dq);
        const hw = K.hw[f][di].map(dq);
        const up = mu.map((v, i) => v + hw[i]);
        const dn = mu.map((v, i) => v - hw[i]);
        up.forEach((v) => { hi = Math.max(hi, v); });
        dn.forEach((v) => { lo = Math.min(lo, v); });
        traces.push({ x: grid, y: up, xaxis: xa, yaxis: ya, mode: 'lines', line: { width: 0, color: C.band }, hoverinfo: 'skip' });
        traces.push({ x: grid, y: dn, xaxis: xa, yaxis: ya, mode: 'lines', line: { width: 0, color: C.band }, fill: 'tonexty', fillcolor: C.band, hoverinfo: 'skip' });
        if (S.interp) {
          traces.push({ x: grid, y: interpCurve(d, K, f, di, grid), xaxis: xa, yaxis: ya, mode: 'lines',
            line: { color: C.interp, width: 1.4, dash: 'dash' }, customdata: gridLab,
            hovertemplate: '%{customdata} µM<br>log-linear baseline %{y:.2f}<extra></extra>' });
        }
        traces.push({ x: grid, y: mu, xaxis: xa, yaxis: ya, mode: 'lines', line: { color: C.model, width: 2.2 },
          customdata: grid.map((g, i) => [gridLab[i], dn[i], up[i]]),
          hovertemplate: `DIV ${div} · %{customdata[0]} µM<br>forecast %{y:.2f}<br>90% band %{customdata[1]:.2f} to %{customdata[2]:.2f}<extra></extra>` });
        // wells
        const wy = d.wells.y[f][di];
        const wm = { x: [], y: [], t: [] };
        const wr = { x: [], y: [], t: [] };
        d.wells.lv.forEach((li, n) => {
          const v = wy[n];
          if (v === null) return;
          const tgt = ctxSet.has(li) ? wm : heldSet.has(li) ? wr : null;
          if (!tgt) return;
          tgt.x.push(d.levels[li]); tgt.y.push(v / SC()); tgt.t.push(uM(d.levels[li]));
        });
        const lm = d.level_mean[f][di];
        const means = (set) => {
          const o = { x: [], y: [], t: [] };
          [...set].sort((a, b) => a - b).forEach((li) => {
            if (lm[li] === null) return;
            o.x.push(d.levels[li]); o.y.push(lm[li] / SC()); o.t.push(`${uM(d.levels[li])} µM, n = ${d.level_n[li]} wells`);
          });
          return o;
        };
        const mm = means(ctxSet);
        mm.y.forEach((v) => { lo = Math.min(lo, v); hi = Math.max(hi, v); });
        if (S.wells) {
          traces.push({ x: wm.x, y: wm.y, xaxis: xa, yaxis: ya, mode: 'markers', marker: { size: 4, color: C.measured, opacity: 0.35 },
            customdata: wm.t, hovertemplate: 'measured well · %{customdata} µM<br>%{y:.2f}<extra></extra>' });
        }
        traces.push({ x: mm.x, y: mm.y, xaxis: xa, yaxis: ya, mode: 'markers',
          marker: { size: 9, color: C.measured, symbol: 'circle', line: { color: '#ffffff', width: 1.5 } },
          customdata: mm.t, hovertemplate: 'measured mean · %{customdata}<br>%{y:.2f}<extra></extra>' });
        if (S.revealed) {
          const rm = means(heldSet);
          rm.y.forEach((v) => { lo = Math.min(lo, v); hi = Math.max(hi, v); });
          if (S.wells) {
            traces.push({ x: wr.x, y: wr.y, xaxis: xa, yaxis: ya, mode: 'markers', marker: { size: 4, color: C.revealed, opacity: 0.4 },
              customdata: wr.t, hovertemplate: 'held-out well · %{customdata} µM<br>%{y:.2f}<extra></extra>' });
          }
          traces.push({ x: rm.x, y: rm.y, xaxis: xa, yaxis: ya, mode: 'markers',
            marker: { size: 10, color: C.revealed, symbol: 'diamond', line: { color: '#ffffff', width: 1.5 } },
            customdata: rm.t, hovertemplate: 'held-out mean · %{customdata}<br>%{y:.2f}<extra></extra>' });
        }
        layout['xaxis' + s] = axis({ domain: doms[di][0], anchor: ya, range: [x0, x1], tickvals: ticks.tickvals, ticktext: ticks.ticktext, automargin: false });
        layout['yaxis' + s] = axis({ domain: doms[di][1], anchor: xa, zeroline: true, zerolinecolor: C.zero, zerolinewidth: 1,
          showticklabels: narrow ? di % 2 === 0 : di === 0, automargin: false });
        layout.annotations.push({ text: `DIV ${div}`, xref: `${xa} domain`, yref: `${ya} domain`, x: 0.5, y: 1, yanchor: 'bottom',
          showarrow: false, font: { size: 11, color: C.ink2 } });
      });
      const pad = Math.max(0.3, 0.08 * (hi - lo));
      const lim = (num(S.meta.clip) ? S.meta.clip : 10) + 0.5;   // targets are clipped at +-clip
      const yr = [Math.max(lo - pad, -lim), Math.min(hi + pad, lim)];
      S.meta.divs.forEach((_, di) => { layout['yaxis' + (di === 0 ? '' : String(di + 1))].range = yr; });
      plot(`p-${fname}`, traces, layout);
    });
    const btn = $('btn-reveal');
    btn.setAttribute('aria-pressed', S.revealed ? 'true' : 'false');
    btn.textContent = S.revealed ? 'Hide the held-out wells' : 'Reveal the held-out wells';
  }

  function renderReveal() {
    const box = $('score');
    if (!S.revealed) { box.hidden = true; return; }
    box.hidden = false;
    const d = S.doc;
    const K = kdata();
    const m = K.metrics.curve_mae;
    const rows = [
      ['NeuroTrajectory (this model)', m.model, C.model],
      ['Log-linear interpolation', m.interp, C.interp],
      ['Analog kNN (nearest training chemicals)', m.knn, C.knn],
    ];
    const best = Math.min(...rows.map((r) => r[1]).filter(num));
    $('score-table').innerHTML = $('score-table').querySelector('caption').outerHTML +
      '<thead><tr><th scope="col">Method</th><th scope="col">Curve MAE</th><th scope="col">vs NeuroTrajectory</th></tr></thead><tbody>' +
      rows.map(([n, v, col], i) => `<tr class="${v === best ? 'hl' : ''}"><td><span class="sw-inline" style="background:${col}"></span>${esc(n)}${v === best ? ' <span class="pill">best</span>' : ''}</td>` +
        `<td>${fx(v, 3)}</td><td>${i === 0 ? '–' : (num(v) && num(m.model) ? `${signed(v - m.model, 3)} (${v > m.model ? 'model ' + (100 * (1 - m.model / v)).toFixed(0) + '% lower' : 'model ' + (100 * (m.model / v - 1)).toFixed(0) + '% higher'})` : '–')}</td></tr>`).join('') +
      '</tbody>';
    const kk = S.meta.chemicals;
    const key = String(S.k);
    const winsI = kk.filter((c) => c.k[key].m < c.k[key].i).length;
    const winsN = kk.filter((c) => c.k[key].m < c.k[key].n).length;
    const verdict = m.model === best
      ? 'NeuroTrajectory is the most accurate method on this chemical.'
      : `A baseline wins on this chemical. Across all chemicals at k = ${S.k} (this design), NeuroTrajectory beats log-linear interpolation on ${winsI} of ${kk.length} and analog kNN on ${winsN}.`;
    $('coverage').innerHTML = `${esc(verdict)} ${num(K.metrics.coverage90) ? `<strong>${pct(K.metrics.coverage90, 0)}</strong> of the held-out well measurements fall inside the 90% band.` : ''}`;
    const labs = K.held.map((i) => `${uM(d.levels[i])} µM`);
    const lm = K.level_mae;
    const bars = [['model', 'NeuroTrajectory', C.model], ['interp', 'Log-linear interpolation', C.interp], ['knn', 'Analog kNN', C.knn]]
      .map(([key2, name, color]) => ({ type: 'bar', name, x: labs, y: lm[key2], marker: { color },
        hovertemplate: `${name}<br>%{x}: %{y:.2f}<extra></extra>` }));
    plot('fig-levels', bars, {
      height: 260, margin: { l: 44, r: 8, t: 8, b: 40 }, barmode: 'group', bargap: 0.25, bargroupgap: 0.08,
      legend: { orientation: 'h', y: -0.28, x: 0, font: { size: 10 } },
      xaxis: axis({ type: 'category', tickfont: { size: 10 } }), yaxis: axis({ title: { text: 'curve MAE', font: { size: 10 } }, rangemode: 'tozero' }),
    });
  }

  function renderHeat() {
    const d = S.doc;
    if (!d) return;
    const K = kdata();
    const j = S.heatJ;
    makeSeg($('heat-seg'), K.held.map((_, i) => i), j, (v) => { S.heatJ = Number(v); renderHeat(); },
      (i) => `${uM(d.levels[K.held[i]])} µM`);
    const feats = S.meta.features;
    const ylab = feats.map(flabel);
    const xlab = S.meta.divs.map((v) => `DIV ${v}`);
    const zf = feats.map((_, f) => S.meta.divs.map((__, di) => dq(K.mu_held[f][di][j])));
    const li = K.held[j];
    const zo = feats.map((_, f) => S.meta.divs.map((__, di) => dq(d.level_mean[f][di][li])));
    const narrow = geometry() === 'narrow';
    const hm = (z, name, show) => ({
      type: 'heatmap', z, x: xlab, y: ylab, zmin: -6, zmax: 6, zmid: 0, colorscale: DIVERGING, xgap: 2, ygap: 2,
      showscale: show, colorbar: { thickness: 10, len: 0.8, title: { text: 'robust SD', side: 'right', font: { size: 10 } }, tickfont: { size: 10 } },
      hovertemplate: `${name}<br>%{y} · %{x}<br>%{z:.2f}<extra></extra>`, hoverongaps: false,
    });
    const lay = (extra) => Object.assign({
      height: 440, margin: { l: narrow ? 150 : 190, r: 8, t: 6, b: 30 },
      xaxis: axis({ side: 'bottom', showline: false, gridcolor: 'rgba(0,0,0,0)', automargin: false }),
      yaxis: axis({ autorange: 'reversed', showline: false, gridcolor: 'rgba(0,0,0,0)', tickfont: { size: narrow ? 9 : 10 }, automargin: false }),
    }, extra || {});
    const conc = `${uM(d.levels[li])} µM`;
    $('heat-f-cap').textContent = `Forecast at ${conc} (the model never saw this concentration)`;
    plot('fig-heat-f', [hm(zf, 'forecast', true)], lay());
    const eo = $('fig-heat-o');
    if (S.revealed) {
      $('heat-o-cap').textContent = `Observed at ${conc} (mean of ${d.level_n[li]} held-out wells)`;
      if (eo.classList.contains('placeholder')) { eo.classList.remove('placeholder'); eo.innerHTML = ''; }
      plot(eo, [hm(zo, 'observed', true)], lay());
    } else {
      $('heat-o-cap').textContent = 'Observed (hidden until you reveal the held-out wells)';
      if (window.Plotly) Plotly.purge(eo);
      eo.classList.add('placeholder');
      eo.innerHTML = '<span>Hidden until you press “Reveal the held-out wells”.</span>';
    }
  }

  function renderScatter() {
    const M = S.meta;
    const key = String(S.k);
    const bn = S.base;
    const rows = M.chemicals.filter((c) => num(c.k[key].m) && num(c.k[key][bn]));
    const mx = Math.max(...rows.map((c) => Math.max(c.k[key].m, c.k[key][bn]))) * 1.05;
    const trace = (sel, name, marker) => {
      const r = rows.filter(sel);
      return { type: 'scatter', mode: 'markers', name, x: r.map((c) => c.k[key][bn]), y: r.map((c) => c.k[key].m),
        customdata: r.map((c) => [c.id, c.chem]), marker,
        hovertemplate: `%{customdata[1]}<br>baseline %{x:.2f} · NeuroTrajectory %{y:.2f}<extra>${name}</extra>` };
    };
    const cur = rows.filter((c) => c.id === S.id);
    const bname = bn === 'i' ? 'log-linear interpolation' : 'analog kNN';
    const data = [
      { type: 'scatter', mode: 'lines', x: [0, mx], y: [0, mx], line: { color: C.zero, width: 1 }, hoverinfo: 'skip', showlegend: false },
      trace((c) => !c.k[key].a, 'answers', { size: 7, color: C.model, opacity: 0.75, line: { color: '#ffffff', width: 1 } }),
      trace((c) => c.k[key].a, 'abstains', { size: 8, color: C.ink2, symbol: 'circle-open', line: { width: 1.5 } }),
      { type: 'scatter', mode: 'markers+text', name: 'selected', x: cur.map((c) => c.k[key][bn]), y: cur.map((c) => c.k[key].m),
        text: cur.map((c) => c.chem.length > 28 ? c.chem.slice(0, 26) + '…' : c.chem), textposition: 'top center',
        textfont: { size: 11, color: C.ink }, marker: { size: 16, color: 'rgba(0,0,0,0)', line: { color: C.revealed, width: 2.5 } },
        hoverinfo: 'skip' },
    ];
    const n = rows.length;
    const winsI = rows.filter((c) => c.k[key].m < c.k[key].i).length;
    const winsN = rows.filter((c) => c.k[key].m < c.k[key].n).length;
    const nab = rows.filter((c) => c.k[key].a).length;
    $('overview-text').textContent = `With k = ${S.k} measured concentrations (evenly spread design), NeuroTrajectory is more accurate than log-linear interpolation on ${winsI} of ${n} chemicals and than analog kNN on ${winsN}; it abstains on ${nab}.`;
    const el = $('fig-scatter');
    plot(el, data, {
      height: 380, margin: { l: 52, r: 12, t: 10, b: 46 }, hovermode: 'closest', dragmode: false,
      legend: { orientation: 'h', x: 0, y: 1.08, font: { size: 11 } },
      xaxis: axis({ range: [0, mx], title: { text: `${bname} curve MAE`, font: { size: 11 } } }),
      yaxis: axis({ range: [0, mx], title: { text: 'NeuroTrajectory curve MAE', font: { size: 11 } } }),
    });
    if (window.Plotly && !S.scatterBound && el.on) {
      el.on('plotly_click', (ev) => {
        const p = ev && ev.points && ev.points[0];
        if (p && p.customdata && p.customdata[0]) {
          selectChem(p.customdata[0]);
          $('explorer').scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
      S.scatterBound = true;
    }
  }

  // ------------------------------------------------------------------ results section
  function renderResults() {
    const T = S.sum.trajectory;
    const ks = Object.keys(T.by_k).map(Number).sort((a, b) => a - b);
    $('results-lede').textContent = `${T.outer_folds.length}-fold cross-validation by chemical over ${T.n_chemicals} chemicals: every chemical is forecast only by the ${T.seeds.length}-model ensemble trained without it, from exactly the same measured wells given to every baseline (${T.designs_per_k} random designs per k, nested model selection).`;
    const order = ['neurotrajectory', 'analog_knn', 'loglinear_interp', 'hill_per_endpoint'];
    const data = order.filter((m) => ks.every((k) => num(T.by_k[k].curve_mae[m]))).map((m) => ({
      type: 'scatter', mode: 'lines+markers', name: METHOD[m].name, x: ks, y: ks.map((k) => T.by_k[k].curve_mae[m]),
      line: { color: METHOD[m].color, width: METHOD[m].width, dash: METHOD[m].dash },
      marker: { color: METHOD[m].color, symbol: METHOD[m].symbol, size: m === 'neurotrajectory' ? 9 : 7 },
      hovertemplate: `${METHOD[m].name}<br>k = %{x}: %{y:.3f}<extra></extra>`,
    }));
    const last = ks[ks.length - 1];
    plot('fig-cv', data, {
      height: 330, margin: { l: 50, r: 12, t: 10, b: 80 }, hovermode: 'closest',
      legend: { orientation: 'h', x: 0, y: -0.3, font: { size: 11 } },
      xaxis: axis({ tickvals: ks, title: { text: 'measured concentrations given (k)', font: { size: 11 } } }),
      yaxis: axis({ title: { text: 'curve MAE (robust SD)', font: { size: 11 } } }),
      annotations: [{ x: last, y: T.by_k[last].curve_mae.neurotrajectory, text: 'NeuroTrajectory', showarrow: false,
        xanchor: 'right', yanchor: 'top', yshift: -8, font: { size: 11, color: C.ink } }],
    });
    $('cv-cap').textContent = `Mean curve error over ${T.n_chemicals} held-out chemicals (lower is better). At k = 0 the baselines fall back to vehicle or population mean.`;

    const F4 = T.by_k_folds_1to4 || {};
    const pv = ks.map((k) => T.by_k[k].paired_vs_best);
    const p4 = ks.map((k) => F4[k] && F4[k].paired_vs_best);
    const errBar = (arr) => ({ type: 'data', symmetric: false, array: arr.map((p) => (p ? p.ci95[1] - p.mean_diff : null)),
      arrayminus: arr.map((p) => (p ? p.mean_diff - p.ci95[0] : null)), thickness: 1.5, width: 4 });
    const pdata = [
      { type: 'scatter', mode: 'markers', name: 'all folds', x: ks.map((k) => k - 0.08), y: pv.map((p) => p.mean_diff),
        error_y: Object.assign(errBar(pv), { color: C.model }), marker: { size: 9, color: C.model },
        customdata: ks.map((k, i) => [METHOD[T.by_k[k].best_baseline] ? METHOD[T.by_k[k].best_baseline].name : T.by_k[k].best_baseline, pv[i].rel_change_pct, pct(pv[i].frac_chem_improved, 0)]),
        hovertemplate: 'k = %{x:.0f} vs %{customdata[0]}<br>Δ %{y:.3f} (%{customdata[1]:.1f}%)<br>better on %{customdata[2]} of chemicals<extra>all folds</extra>' },
    ];
    if (p4.some(Boolean)) {
      pdata.push({ type: 'scatter', mode: 'markers', name: 'excluding development fold', x: ks.map((k) => k + 0.08), y: p4.map((p) => (p ? p.mean_diff : null)),
        error_y: Object.assign(errBar(p4), { color: C.knn }), marker: { size: 8, color: C.knn, symbol: 'square' },
        hovertemplate: 'k = %{x:.0f}<br>Δ %{y:.3f}<extra>excluding development fold</extra>' });
    }
    plot('fig-paired', pdata, {
      height: 330, margin: { l: 56, r: 12, t: 10, b: 80 }, hovermode: 'closest',
      legend: { orientation: 'h', x: 0, y: -0.3, font: { size: 11 } },
      xaxis: axis({ tickvals: ks, title: { text: 'measured concentrations given (k)', font: { size: 11 } } }),
      yaxis: axis({ zeroline: true, zerolinecolor: C.ink3, zerolinewidth: 1, title: { text: 'Δ curve MAE vs best baseline', font: { size: 11 } } }),
    });

    $('tbl-cv').innerHTML = tableCaption('tbl-cv', 'Curve MAE by number of measured concentrations; Δ = NeuroTrajectory − best baseline (paired bootstrap over chemicals). Scroll the table sideways on small screens.') +
      '<thead><tr><th scope="col">k</th><th scope="col" class="t">Best baseline</th><th scope="col">Baseline MAE</th><th scope="col">NeuroTrajectory MAE</th><th scope="col">Δ (95% CI)</th><th scope="col">Relative</th><th scope="col">Chemicals improved</th><th scope="col">90% well-band coverage</th></tr></thead><tbody>' +
      ks.map((k) => {
        const b = T.by_k[k];
        const p = b.paired_vs_best;
        const w = b.well_interval90 || {};
        return `<tr><td>${k}</td><td class="t">${esc(METHOD[b.best_baseline] ? METHOD[b.best_baseline].name : b.best_baseline)}</td><td>${fx(b.curve_mae[b.best_baseline], 3)}</td>` +
          `<td>${fx(b.curve_mae.neurotrajectory, 3)}</td><td>${signed(p.mean_diff)} (${ci(p.ci95)})</td><td>${num(p.rel_change_pct) ? signed(p.rel_change_pct, 1) + '%' : '–'}</td>` +
          `<td>${pct(p.frac_chem_improved, 0)}</td><td>${pct(w.coverage_mean)}</td></tr>`;
      }).join('') + '</tbody>';

    // conformal
    const CF = S.sum.conformal;
    const alphas = Object.keys(CF.results).sort((a, b) => CF.results[a].nominal - CF.results[b].nominal);
    const reg = CF.chemicals_per_regime || {};
    $('conf-text').textContent = `Cross-conformal calibration, out-of-fold, Mondrian by cytotoxicity regime (${Object.entries(reg).map(([k, v]) => `${v} ${k}`).join(', ')}) and k. Coverage is the share of held-out well measurements inside the band.`;
    $('tbl-conf').innerHTML = '<thead><tr><th scope="col">Nominal</th><th scope="col">k</th><th scope="col">Uncalibrated</th><th scope="col">Conformal (95% CI)</th><th scope="col">Mean width</th></tr></thead><tbody>' +
      alphas.map((a) => {
        const e = CF.results[a];
        return Object.keys(e).filter((x) => x.startsWith('k')).sort().map((kk) =>
          `<tr class="${e.nominal === 0.9 ? '' : ''}"><td>${pct(e.nominal, 0)}</td><td>${kk.slice(1)}</td><td>${pct(e[kk].parametric.coverage)}</td>` +
          `<td>${pct(e[kk].conformal.coverage)} (${pct(e[kk].conformal.ci95[0])}–${pct(e[kk].conformal.ci95[1])})</td><td>${fx(e[kk].conformal.width)}</td></tr>`).join('');
      }).join('') + '</tbody>';

    const ab = CF.abstention;
    const byk = ab.by_k || {};
    $('abst-text').textContent = `Rule: ${ab.rule}. Epistemic score = mean disagreement of the ensemble members divided by the mean predictive scale. ` +
      Object.entries(byk).map(([k, v]) => `k = ${k}: abstains on ${pct(v.rate)}, error ${fx(v.mae_retained)} when answering vs ${fx(v.mae_abstained)} when abstaining`).join('; ') + '.';
    $('abst-stats').innerHTML = [
      [pct(ab.abstention_rate), 'forecasts flagged'],
      [fx(ab.curve_mae_retained), 'curve MAE when it answers'],
      [fx(ab.curve_mae_abstained), 'curve MAE when it abstains'],
    ].map(([v, l]) => `<div class="stat"><span class="stat-v">${v}</span><span class="stat-l">${l}</span></div>`).join('');

    // EPA reproduction
    const E = S.sum.epa_baseline;
    const models = Object.entries(E.models);
    const maxDelta = Math.max(...models.flatMap(([, m]) => Object.values(m.published_delta_pp || {}).map((v) => Math.abs(v))));
    $('epa-text').textContent = `Before forecasting anything we reproduced the US EPA NFA bioactivity classifier on its ${E.n_reference} reference chemicals (${E.n_positive} positive, ${E.n_negative} negative): the largest difference from the published sensitivity, specificity or balanced accuracy is ${fx(maxDelta, 1)} percentage points. ${E.interpretation || ''}`;
    $('tbl-epa').innerHTML = '<thead><tr><th scope="col">EPA model</th><th scope="col">Hit threshold</th><th scope="col">Sensitivity</th><th scope="col">Specificity</th><th scope="col">Balanced accuracy</th><th scope="col">Δ vs published (pp)</th></tr></thead><tbody>' +
      models.map(([n, m]) => {
        const dl = m.published_delta_pp || {};
        const dmax = Math.max(...Object.values(dl).map((v) => Math.abs(v)));
        return `<tr><td>${esc(n)}</td><td>${m.threshold_hits}</td><td>${fx(m.sensitivity_pct, 1)}%</td><td>${fx(m.specificity_pct, 1)}%</td><td>${fx(m.balanced_accuracy_pct, 1)}%</td><td>${fx(dmax, 1)}</td></tr>`;
      }).join('') + '</tbody>';
  }

  // ------------------------------------------------------------------ chip layer
  function renderChip() {
    const X = S.sum.chip;
    const cond = Object.entries(X.conditions || {}).map(([k, v]) => `${v} ${k}`).join(', ');
    $('chip-lede').innerHTML = `The same question at chip scale: in a 4-compartment hippocampal MEA (entorhinal cortex → dentate gyrus → CA3 → CA1, joined by microtunnels), in which direction do axons carry spikes between compartments, and how fast? ` +
      `We use ${X.n_axons} spike-sorted axons from ${X.n_recordings} real recordings (${esc(cond)}) of the <a href="${esc(X.source)}" rel="noopener">Brewer dataset</a> (${esc(X.license)}), fit direction and conduction per compartment pair on the unstimulated recordings and test them leave-one-recording-out.`;

    // diagram
    const P = X.params;
    const edges = Object.fromEntries(P.pair_edges.map((e) => [e.pair, e]));
    const pos = { EC: [150, 70], DG: [330, 70], CA3: [330, 230], CA1: [150, 230] };
    const pairs = [['EC', 'DG'], ['DG', 'CA3'], ['CA3', 'CA1'], ['CA1', 'EC']];
    const W = 480;
    const H = 322;
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="chip-svg-t chip-svg-d" xmlns="http://www.w3.org/2000/svg">` +
      '<title id="chip-svg-t">Compartment graph with fitted feed-forward and feedback axon fractions</title>' +
      `<desc id="chip-svg-d">${esc(P.pair_edges.map((e) => `${e.pair}: feed-forward ${pct(e.ff_weight, 0)}, feedback ${pct(e.fb_weight, 0)}, ${e.conduction.n} axons, median conduction ${fx(e.conduction.median_ms, 2)} ms`).join('; '))}</desc>` +
      '<defs>' + [['ff', C.ff], ['fb', C.fb]].map(([id, col]) =>
        `<marker id="ah-${id}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="${col}"/></marker>`).join('') + '</defs>';
    const box = { w: 72, h: 40 };
    pairs.forEach(([a, b]) => {
      const e = edges[`${a}-${b}`];
      if (!e) return;
      const [ax, ay] = pos[a];
      const [bx, by] = pos[b];
      const dx = bx - ax;
      const dy = by - ay;
      const L = Math.hypot(dx, dy);
      const ux = dx / L;
      const uy = dy / L;
      const nx = -uy;
      const ny = ux;
      const shrinkA = Math.abs(ux) > 0.5 ? box.w / 2 + 8 : box.h / 2 + 8;
      const line = (from, to, off, col, w, id) => {
        const [fx0, fy0] = from;
        const [tx0, ty0] = to;
        const vx = (tx0 - fx0) / L;
        const vy = (ty0 - fy0) / L;
        const x1 = fx0 + vx * shrinkA + nx * off;
        const y1 = fy0 + vy * shrinkA + ny * off;
        const x2 = tx0 - vx * shrinkA + nx * off;
        const y2 = ty0 - vy * shrinkA + ny * off;
        return `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${col}" stroke-width="${w.toFixed(1)}" stroke-linecap="round" marker-end="url(#ah-${id})"/>`;
      };
      // the two directions run on opposite sides of the compartment axis so neither hides the other
      svg += line([ax, ay], [bx, by], -7, C.ff, 1.5 + 9 * e.ff_weight, 'ff');
      svg += line([bx, by], [ax, ay], 7, C.fb, 1.5 + 9 * e.fb_weight, 'fb');
      const mx = (ax + bx) / 2;
      const my = (ay + by) / 2;
      const horizontal = Math.abs(dy) < 1;
      const top = horizontal && ay < H / 2;
      const right = !horizontal && ax > W / 2;
      const tx = horizontal ? mx : right ? mx + 30 : mx - 30;
      const ty = horizontal ? (top ? my - 38 : my + 36) : my - 6;
      const anchor = horizontal ? 'middle' : right ? 'start' : 'end';
      svg += `<text x="${tx}" y="${ty}" text-anchor="${anchor}" font-size="12" fill="${C.ink}" font-family="${FONT}">` +
        `<tspan fill="${C.ink}" font-weight="600">FF ${pct(e.ff_weight, 0)}</tspan><tspan fill="${C.ink2}"> · FB ${pct(e.fb_weight, 0)}</tspan>` +
        `<tspan x="${tx}" dy="15" fill="${C.ink3}" font-size="11">${e.conduction.n} axons · ${fx(e.conduction.median_ms, 2)} ms</tspan></text>`;
    });
    Object.entries(pos).forEach(([n, [x, y]]) => {
      svg += `<rect x="${x - box.w / 2}" y="${y - box.h / 2}" width="${box.w}" height="${box.h}" rx="8" fill="#ffffff" stroke="${C.ink2}" stroke-width="1.2"/>` +
        `<text x="${x}" y="${y + 5}" text-anchor="middle" font-size="14" font-weight="700" fill="${C.ink}" font-family="${FONT}">${n}</text>`;
    });
    svg += `<g font-family="${FONT}" font-size="11" fill="${C.ink2}"><line x1="20" y1="${H - 12}" x2="46" y2="${H - 12}" stroke="${C.ff}" stroke-width="4"/><text x="52" y="${H - 8}">feed-forward (EC→DG→CA3→CA1→EC)</text>` +
      `<line x1="290" y1="${H - 12}" x2="316" y2="${H - 12}" stroke="${C.fb}" stroke-width="4"/><text x="322" y="${H - 8}">feedback (reverse)</text></g></svg>`;
    $('chip-diagram').innerHTML = svg;
    $('tbl-edges').innerHTML = '<caption class="sr-only">Fitted direction and conduction per compartment pair</caption>' +
      '<thead><tr><th scope="col">Pair</th><th scope="col">Feed-forward</th><th scope="col">Feedback</th><th scope="col">Axons</th><th scope="col">Median conduction</th></tr></thead><tbody>' +
      pairs.map(([a, b]) => edges[`${a}-${b}`]).filter(Boolean).map((e) =>
        `<tr><td class="nw">${esc(e.pair.replace('-', ' → '))}</td><td>${pct(e.ff_weight, 0)}</td><td>${pct(e.fb_weight, 0)}</td><td>${e.conduction.n}</td><td>${fx(e.conduction.median_ms, 2)} ms</td></tr>`).join('') +
      '</tbody>';
    $('chip-diagram-cap').textContent = `Arrow width = fitted share of axons in each direction (${P.source_cohort}); labels give the number of axons and the median conduction time across the tunnel. Scope: ${P.scope}.`;

    // directionality strip
    const names = pairs.map(([a, b]) => `${a}-${b}`);
    const obs = (X.observed || []).filter((o) => o.condition === 'NoStim');
    const sx = [];
    const sy = [];
    const st = [];
    obs.forEach((o, oi) => o.edges.forEach((e) => {
      const i = names.indexOf(e.pair);
      if (i < 0 || !num(e.directionality_d)) return;
      sx.push(i + (oi - (obs.length - 1) / 2) * 0.045);
      sy.push(e.directionality_d);
      st.push(`${e.pair} · recording ${o.fid}<br>FF ${e.n_axons_ff}, FB ${e.n_axons_fb} axons`);
    }));
    plot('fig-chip-dir', [
      { type: 'scatter', mode: 'markers', name: 'per recording', x: sx, y: sy, customdata: st,
        marker: { size: 8, color: C.model, opacity: 0.7, line: { color: '#ffffff', width: 1 } },
        hovertemplate: '%{customdata}<br>d = %{y:.2f}<extra></extra>' },
      { type: 'scatter', mode: 'markers', name: 'pooled', x: names.map((_, i) => i), y: names.map((n) => (edges[n] ? edges[n].observed_directionality_d : null)),
        marker: { symbol: 'line-ew-open', size: 34, color: C.ink, line: { width: 3, color: C.ink } },
        hovertemplate: 'pooled d = %{y:.2f}<extra></extra>' },
    ], {
      height: 320, margin: { l: 44, r: 10, t: 10, b: 36 }, showlegend: false, hovermode: 'closest',
      xaxis: axis({ tickvals: names.map((_, i) => i), ticktext: names, range: [-0.6, names.length - 0.4] }),
      yaxis: axis({ range: [-1.1, 1.1], zeroline: true, zerolinecolor: C.zero, title: { text: 'directionality d', font: { size: 11 } } }),
    });

    // leave-one-recording-out dumbbell
    const folds = (X.cv.folds || []).filter((f) => f.scores && f.scores.model && f.scores.tunnel_mean);
    const ylab = folds.map((f) => `rec. ${f.fid}`);
    const dd = [];
    folds.forEach((f, i) => dd.push({ type: 'scatter', mode: 'lines', x: [f.scores.tunnel_mean.ff_fraction_mae, f.scores.model.ff_fraction_mae],
      y: [ylab[i], ylab[i]], line: { color: C.zero, width: 2 }, hoverinfo: 'skip', showlegend: false }));
    dd.push({ type: 'scatter', mode: 'markers', name: 'pooled tunnel mean', x: folds.map((f) => f.scores.tunnel_mean.ff_fraction_mae), y: ylab,
      marker: { size: 10, color: C.knn, symbol: 'square' }, hovertemplate: '%{y}: %{x:.3f}<extra>tunnel mean</extra>' });
    dd.push({ type: 'scatter', mode: 'markers', name: 'pair/tunnel shrinkage model', x: folds.map((f) => f.scores.model.ff_fraction_mae), y: ylab,
      marker: { size: 10, color: C.model }, hovertemplate: '%{y}: %{x:.3f}<extra>model</extra>' });
    plot('fig-chip-cv', dd, {
      height: 340, margin: { l: 56, r: 12, t: 10, b: 80 }, hovermode: 'closest',
      legend: { orientation: 'h', x: 0, y: -0.28, font: { size: 11 } },
      xaxis: axis({ title: { text: 'feed-forward fraction MAE (held-out recording)', font: { size: 11 } } }),
      yaxis: axis({ type: 'category', autorange: 'reversed' }),
    });
    const cvs = X.cv.summary;
    $('chip-cv-cap').textContent = `${X.cv.protocol}; model: ${X.cv.model}. Mean over recordings: model ${fx(cvs.model.ff_fraction_mae, 3)} vs tunnel mean ${fx(cvs.tunnel_mean.ff_fraction_mae, 3)}.`;

    const metricName = { ff_fraction_mae: 'Feed-forward fraction MAE', conduction_mae_ms: 'Conduction time MAE (ms)', conduction_nll_nats: 'Conduction NLL (nats)' };
    const baseName = { tunnel_mean: 'Pooled tunnel mean', permuted_direction: 'Permuted direction labels' };
    const rows = [];
    Object.entries(X.cv.comparisons || {}).forEach(([bn, mets]) => Object.entries(mets).forEach(([mn, c]) => {
      if (bn === 'permuted_direction' && mn !== 'ff_fraction_mae') return;   // delay is identical by construction
      rows.push(`<tr><td>${esc(metricName[mn] || mn)}</td><td class="t">${esc(baseName[bn] || bn)}</td><td>${fx(cvs.model[mn], 3)}</td><td>${fx(cvs[bn] && cvs[bn][mn], 3)}</td>` +
        `<td>${signed(c.difference_model_minus_baseline)} (${ci(c.ci95)})</td><td>${fx(c.p_bootstrap_no_improvement, 3)}</td></tr>`);
    }));
    $('tbl-chip').innerHTML = tableCaption('tbl-chip', 'Model − baseline over held-out recordings (lower is better); bootstrap by recording.') +
      '<thead><tr><th scope="col">Metric</th><th scope="col" class="t">Baseline</th><th scope="col">Model</th><th scope="col">Baseline value</th><th scope="col">Δ (95% CI)</th><th scope="col">p (no gain)</th></tr></thead><tbody>' +
      rows.join('') + '</tbody>';
    $('chip-note').textContent = `Burst analysis: ${X.burst_interpretation}. These are descriptive parameters of one platform; compartment "what-if" scenarios built on them are simulations, not measurements.`;
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
