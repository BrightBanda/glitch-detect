/* GLITCH//DETECT — front-end controller */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const PROVIDERS = ["openrouter", "groq"];

  const els = {
    apiKey: $("apiKey"),
    toggleKey: $("toggleKey"),
    rememberKey: $("rememberKey"),
    keyBadge: $("keyBadge"),
    keyLabel: $("keyLabel"),
    keysLink: $("keysLink"),
    providerNote: $("providerNote"),
    footerTransport: $("footerTransport"),
    providerButtons: Array.from(document.querySelectorAll(".provider-btn")),
    model: $("model"),
    modelHint: $("modelHint"),
    refreshModels: $("refreshModels"),
    chunkWords: $("chunkWords"),
    engineLabel: $("engineLabel"),
    nodeStatus: $("nodeStatus"),
    nodeLabel: $("nodeLabel"),
    text: $("text"),
    textStats: $("textStats"),
    analyzeBtn: $("analyzeBtn"),
    sampleBtn: $("sampleBtn"),
    clearBtn: $("clearBtn"),
    elapsed: $("elapsed"),
    editorPanel: $("editorPanel"),
    alert: $("alert"),
    alertTitle: $("alertTitle"),
    alertMessage: $("alertMessage"),
    alertDetailWrap: $("alertDetailWrap"),
    alertDetail: $("alertDetail"),
    alertClose: $("alertClose"),
    results: $("results"),
    emptyState: $("emptyState"),
    scoreboard: $("scoreboard"),
    gaugeArc: $("gaugeArc"),
    scoreValue: $("scoreValue"),
    verdictTag: $("verdictTag"),
    verdict: $("verdict"),
    verdictNote: $("verdictNote"),
    scanMeta: $("scanMeta"),
    countAI: $("countAI"),
    countRefined: $("countRefined"),
    countHuman: $("countHuman"),
    countTotal: $("countTotal"),
    warnings: $("warnings"),
    warningList: $("warningList"),
    doc: $("doc"),
    showTags: $("showTags"),
    copyBtn: $("copyBtn"),
    downloadBtn: $("downloadBtn"),
    rawJson: $("rawJson"),
    dots: $("dots"),
  };

  const state = {
    provider: "openrouter",
    keys: { openrouter: "", groq: "" },
    models: { openrouter: [], groq: [] },
    lastModel: { openrouter: "", groq: "" },
    analysis: null,
    busy: false,
    timer: null,
    startedAt: 0,
    dotTimer: null,
    keyTimer: null,
  };

  const SAMPLE = `In today's rapidly evolving digital landscape, artificial intelligence plays a crucial role in shaping the future of human communication. By leveraging advanced algorithms, organizations can foster innovation, streamline workflows, and unlock unprecedented levels of efficiency. Moreover, these systems demonstrate a strong commitment to adaptability in an increasingly complex world.

I tried the beta on a Tuesday and it ate four hours of my afternoon. The export button silently dropped every table with a merged cell, which I only noticed because a colleague asked why the revenue column was empty. We fixed it by pasting into a spreadsheet and re-importing, a workaround so stupid that I assumed it would be temporary. It has been two years.

Ultimately, the integration of intelligent systems serves as a testament to the transformative power of technology, ensuring that stakeholders remain well-positioned to navigate the challenges of tomorrow.`;

  /* ─────────────────────────── utilities ─────────────────────────── */
  const countWords = (value) => {
    const trimmed = value.trim();
    return trimmed ? trimmed.split(/\s+/).length : 0;
  };

  function fmtElapsed(ms) {
    const s = ms / 1000;
    return s < 60 ? `${s.toFixed(1)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  }

  function setNode(stateName, label) {
    els.nodeStatus.dataset.state = stateName;
    els.nodeLabel.textContent = label;
  }

  function providerMeta(provider) {
    const btn = els.providerButtons.find((b) => b.dataset.provider === provider);
    return btn ? btn.dataset : null;
  }

  function activeKey() {
    return els.apiKey.value.trim();
  }

  function syncEngineLabel() {
    const meta = providerMeta(state.provider);
    const label = meta ? meta.providerLabel : "Node";
    const model = els.model.value.trim() || "—";
    els.engineLabel.textContent = `${label} · ${model}`;
    els.engineLabel.title = `${label} · ${model}`;
  }

  /* ─────────────────────────── health ─────────────────────────── */
  async function checkHealth() {
    try {
      const res = await fetch("/api/health", { cache: "no-store" });
      if (!res.ok) throw new Error("bad status");
      await res.json();
      setNode("online", "LOCAL NODE ONLINE");
    } catch {
      setNode("offline", "LOCAL NODE OFFLINE");
    }
  }

  /* ─────────────────────────── provider + keys ─────────────────────────── */
  function refreshKeyBadge() {
    const has = activeKey().length >= 12;
    els.keyBadge.textContent = has ? "KEY SET" : "NO KEY";
    els.keyBadge.dataset.state = has ? "set" : "empty";
  }

  function persistProvider() {
    try {
      sessionStorage.setItem("glitch_provider", state.provider);
    } catch {
      /* ignore */
    }
  }

  function persistKey() {
    state.keys[state.provider] = activeKey();
    try {
      if (els.rememberKey.checked && activeKey()) {
        sessionStorage.setItem(`glitch_key_${state.provider}`, activeKey());
      } else {
        sessionStorage.removeItem(`glitch_key_${state.provider}`);
      }
    } catch {
      /* storage disabled */
    }
  }

  function rememberModel(provider, model) {
    state.lastModel[provider] = model;
    try {
      sessionStorage.setItem(`glitch_model_${provider}`, model);
    } catch {
      /* ignore */
    }
  }

  function restoreState() {
    try {
      const provider = sessionStorage.getItem("glitch_provider");
      if (PROVIDERS.includes(provider)) state.provider = provider;

      for (const p of PROVIDERS) {
        const key = sessionStorage.getItem(`glitch_key_${p}`);
        if (key) state.keys[p] = key;
        const model = sessionStorage.getItem(`glitch_model_${p}`);
        if (model) state.lastModel[p] = model;
      }
      els.rememberKey.checked = Boolean(
        sessionStorage.getItem(`glitch_key_${state.provider}`)
      );
    } catch {
      /* ignore */
    }
  }

  function setProvider(provider, { load = true, autoSelect = false } = {}) {
    const meta = providerMeta(provider);
    if (!meta) return;

    if (state.provider !== provider) {
      state.keys[state.provider] = activeKey();
      state.lastModel[state.provider] = els.model.value.trim();
    }

    state.provider = provider;
    els.providerButtons.forEach((btn) => {
      const active = btn.dataset.provider === provider;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-pressed", active ? "true" : "false");
    });

    els.keyLabel.textContent = meta.keyLabel;
    els.apiKey.placeholder = meta.keyPlaceholder;
    els.apiKey.value = state.keys[provider] || "";
    els.keysLink.href = meta.keysUrl;
    els.keysLink.textContent = "Get a key ↗";
    els.providerNote.textContent = `${meta.providerLabel.toUpperCase()} TRANSPORT · FREE MODELS ONLY`;
    els.footerTransport.textContent = `LOCAL NODE / ${meta.providerLabel.toUpperCase()} TRANSPORT`;
    els.model.value = state.lastModel[provider] || meta.defaultModel;
    els.modelHint.textContent =
      provider === "groq"
        ? "Only Groq Free Plan text models · 8K TPM."
        : "Only free models · ranked for this task.";

    refreshKeyBadge();
    syncEngineLabel();
    persistProvider();

    if (els.rememberKey.checked && activeKey()) {
      try {
        sessionStorage.setItem(`glitch_key_${provider}`, activeKey());
      } catch {
        /* ignore */
      }
    }

    if (load) loadModels(provider, { autoSelect });
  }

  /* ─────────────────────────── model catalogue ─────────────────────────── */
  function renderModelOptions(models) {
    const list = $("modelList");
    list.innerHTML = "";

    const groups = new Map();
    models.forEach((m) => {
      const label = m.group_label || "FREE MODELS";
      if (!groups.has(label)) groups.set(label, []);
      groups.get(label).push(m);
    });

    for (const [label, items] of groups) {
      const group = document.createElement("optgroup");
      group.label = label;
      items.forEach((m) => {
        const option = document.createElement("option");
        option.value = m.id;
        const ctx = m.context ? `${Math.round(m.context / 1024)}K ctx` : "";
        const descriptor = [ctx, m.note].filter(Boolean).join(" · ");
        option.label = descriptor ? `${m.id} — ${descriptor}` : m.id;
        group.appendChild(option);
      });
      list.appendChild(group);
    }
  }

  async function loadModels(provider, { force = false, autoSelect = false } = {}) {
    if (!providerMeta(provider)) return;

    const isActive = state.provider === provider;
    const key = isActive ? activeKey() : state.keys[provider] || "";
    if (isActive) els.modelHint.textContent = "LOADING FREE MODELS…";
    els.refreshModels.disabled = true;

    try {
      const res = await fetch("/api/models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider, api_key: key, force }),
      });
      const data = await res.json().catch(() => null);

      if (!res.ok || !data || !data.ok) {
        const err = (data && data.error) || {};
        throw new Error(err.message || `Could not load models (HTTP ${res.status}).`);
      }

      const models = Array.isArray(data.models) ? data.models : [];
      state.models[provider] = models;
      if (state.provider !== provider) return;

      renderModelOptions(models);

      const ids = models.map((m) => m.id);
      const current = els.model.value.trim();
      if (autoSelect || !current || !ids.includes(current)) {
        els.model.value = data.default || models[0]?.id || providerMeta(provider).defaultModel;
        rememberModel(provider, els.model.value);
      }

      const count = models.length;
      const source = data.source === "live" ? "live" : "documented";
      els.modelHint.textContent = `${count} free model${count === 1 ? "" : "s"} · ${source}${
        data.note ? ` · ${data.note}` : ""
      }`;
      syncEngineLabel();
    } catch (err) {
      if (state.provider === provider) {
        els.modelHint.textContent = err.message;
      }
    } finally {
      els.refreshModels.disabled = false;
    }
  }

  /* ─────────────────────────── alert ─────────────────────────── */
  function showAlert(title, message, detail = "") {
    els.alertTitle.textContent = title;
    els.alertMessage.textContent = message;
    if (detail) {
      els.alertDetail.textContent = detail;
      els.alertDetailWrap.hidden = false;
    } else {
      els.alertDetail.textContent = "";
      els.alertDetailWrap.hidden = true;
    }
    els.alert.hidden = false;
  }

  function hideAlert() {
    els.alert.hidden = true;
  }

  /* ─────────────────────────── loading ─────────────────────────── */
  function startLoading() {
    state.busy = true;
    state.startedAt = performance.now();
    els.analyzeBtn.disabled = true;
    els.analyzeBtn.classList.add("loading");
    els.editorPanel.classList.add("scanning");
    hideAlert();
    els.elapsed.hidden = false;
    els.elapsed.textContent = "0.0s";

    state.timer = setInterval(() => {
      els.elapsed.textContent = fmtElapsed(performance.now() - state.startedAt);
    }, 100);

    let n = 0;
    state.dotTimer = setInterval(() => {
      n = (n + 1) % 4;
      els.dots.textContent = ".".repeat(n);
    }, 420);
  }

  function stopLoading() {
    state.busy = false;
    els.analyzeBtn.disabled = false;
    els.analyzeBtn.classList.remove("loading");
    els.editorPanel.classList.remove("scanning");
    clearInterval(state.timer);
    clearInterval(state.dotTimer);
    state.timer = null;
    state.dotTimer = null;
  }

  /* ─────────────────────────── rendering ─────────────────────────── */
  function levelFor(score) {
    if (score >= 67) return "high";
    if (score >= 34) return "mid";
    return "low";
  }

  function verdictFor(score) {
    if (score >= 85) {
      return ["HIGH MACHINE PROBABILITY", "The passage's rhythm and phrasing are strongly machine-typical."];
    }
    if (score >= 67) {
      return ["MACHINE-LEANING", "Most sentences read as generated or heavily optimised."];
    }
    if (score >= 45) {
      return ["HYBRID DOCUMENT", "Machine and human signals are both present. Inspect the sentence map."];
    }
    if (score >= 22) {
      return ["MOSTLY HUMAN", "A minority of sentences carry machine or polished signatures."];
    }
    return ["HUMAN SIGNAL", "Variance, idiom and rhythm read as organic."];
  }

  function animateNumber(target) {
    const duration = 750;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (document.hidden || reduce) {
      els.scoreValue.textContent = target;
      return;
    }
    const start = performance.now();
    const from = Number(els.scoreValue.textContent) || 0;
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      els.scoreValue.textContent = target;
    };
    // rAF can be throttled in background tabs; never leave the gauge at zero.
    const guard = setTimeout(finish, duration + 250);
    const step = (now) => {
      if (done) return;
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      els.scoreValue.textContent = Math.round(from + (target - from) * eased);
      if (t < 1) requestAnimationFrame(step);
      else {
        clearTimeout(guard);
        finish();
      }
    };
    requestAnimationFrame(step);
  }

  function render(result) {
    state.analysis = result;

    const score = Number(result.score) || 0;
    const level = levelFor(score);
    els.scoreboard.dataset.level = level;
    els.gaugeArc.style.strokeDashoffset = String(100 - score);
    animateNumber(score);

    const [verdict, note] = verdictFor(score);
    els.verdict.textContent = verdict;
    els.verdictNote.textContent = note;
    els.verdictTag.textContent = score >= 45 ? "MACHINE SIGNAL" : "HUMAN SIGNAL";

    els.countAI.textContent = result.counts.AI;
    els.countRefined.textContent = result.counts.REFINED;
    els.countHuman.textContent = result.counts.HUMAN;
    els.countTotal.textContent = result.total;

    const seconds = (result.elapsed_ms / 1000).toFixed(1);
    const provider = (result.provider_label || "").toUpperCase();
    els.scanMeta.textContent =
      `${result.total} SENTENCES · ${result.chunks} CALL${result.chunks === 1 ? "" : "S"} · ${seconds}s${
        provider ? ` · ${provider}` : ""
      }`;

    if (result.warnings && result.warnings.length) {
      els.warningList.innerHTML = "";
      result.warnings.forEach((w) => {
        const li = document.createElement("li");
        li.textContent = w;
        els.warningList.appendChild(li);
      });
      els.warnings.hidden = false;
    } else {
      els.warnings.hidden = true;
    }

    els.doc.innerHTML = result.html;
    els.doc.classList.toggle("show-tags", els.showTags.checked);

    const payload = {
      score: result.score,
      counts: result.counts,
      total: result.total,
      chunks: result.chunks,
      provider: result.provider,
      provider_label: result.provider_label,
      model: result.model,
      words: result.words,
      chars: result.chars,
      elapsed_ms: result.elapsed_ms,
      warnings: result.warnings,
      sentences: result.sentences,
    };
    els.rawJson.textContent = JSON.stringify(payload, null, 2);

    els.results.hidden = false;
    els.emptyState.hidden = true;
    els.results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /* ─────────────────────────── analyze ─────────────────────────── */
  async function analyze() {
    if (state.busy) return;

    const text = els.text.value.trim();
    const apiKey = activeKey();
    const model = els.model.value.trim();
    const provider = state.provider;

    hideAlert();

    if (!apiKey || apiKey.length < 12) {
      showAlert(
        "KEY REQUIRED",
        `Paste your ${provider === "groq" ? "Groq" : "OpenRouter"} API key in the config rail before scanning.`
      );
      els.apiKey.focus();
      return;
    }
    if (text.length < 80) {
      showAlert("INPUT TOO SHORT", "Paste at least a couple of sentences (80+ characters) so the model can classify them.");
      els.text.focus();
      return;
    }
    if (!model.includes("/")) {
      showAlert("MODEL INVALID", "Pick a free model from the list, or enter a valid provider model ID.");
      els.model.focus();
      return;
    }

    persistKey();
    rememberModel(provider, model);
    startLoading();

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 240000);

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          api_key: apiKey,
          model,
          provider,
          chunk_words: Number(els.chunkWords.value) || 900,
        }),
        signal: controller.signal,
      });

      const data = await res.json().catch(() => null);

      if (!res.ok || !data || !data.ok) {
        const err = (data && data.error) || {};
        let title = "SCAN FAILED";
        if (res.status === 429) title = "RATE LIMITED";
        else if (res.status === 401) title = "KEY REJECTED";
        else if (res.status === 400) title = "REQUEST BLOCKED";
        showAlert(title, err.message || `The local node returned HTTP ${res.status}.`, err.detail || "");
        if (res.status === 401 && provider === "groq") {
          els.modelHint.textContent = "Groq key rejected — model list unavailable.";
        }
        return;
      }

      render(data.result);
    } catch (err) {
      if (err.name === "AbortError") {
        showAlert("REQUEST TIMED OUT", "The free model took longer than four minutes. Try a shorter passage or another model.");
      } else {
        showAlert("CONNECTION ERROR", "Could not reach the local server. Is it still running?");
      }
    } finally {
      clearTimeout(timeout);
      stopLoading();
    }
  }

  /* ─────────────────────────── actions ─────────────────────────── */
  function resetView() {
    els.results.hidden = true;
    els.emptyState.hidden = false;
    hideAlert();
    state.analysis = null;
  }

  async function copyJson() {
    if (!state.analysis) return;
    const text = els.rawJson.textContent;
    try {
      await navigator.clipboard.writeText(text);
      const original = els.copyBtn.textContent;
      els.copyBtn.textContent = "COPIED ✓";
      setTimeout(() => (els.copyBtn.textContent = original), 1500);
    } catch {
      showAlert("CLIPBOARD BLOCKED", "Your browser refused clipboard access. Open the raw JSON panel and copy manually.");
    }
  }

  function downloadJson() {
    if (!state.analysis) return;
    const blob = new Blob([els.rawJson.textContent], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    a.href = url;
    a.download = `glitch-analysis-${stamp}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  /* ─────────────────────────── wire up ─────────────────────────── */
  function updateStats() {
    const value = els.text.value;
    els.textStats.textContent = `${countWords(value).toLocaleString()} WORDS · ${value.length.toLocaleString()} CHARS`;
  }

  els.text.addEventListener("input", updateStats);

  els.text.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      analyze();
    }
  });

  els.analyzeBtn.addEventListener("click", analyze);
  els.keyBadge.addEventListener("click", () => els.apiKey.focus());

  els.providerButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      if (state.busy || btn.dataset.provider === state.provider) return;
      setProvider(btn.dataset.provider, {
        load: true,
        autoSelect: !state.lastModel[btn.dataset.provider],
      });
    });
  });

  els.apiKey.addEventListener("input", () => {
    state.keys[state.provider] = els.apiKey.value;
    refreshKeyBadge();
    if (els.rememberKey.checked) persistKey();

    if (state.provider === "groq") {
      clearTimeout(state.keyTimer);
      if (activeKey().length >= 20) {
        state.keyTimer = setTimeout(() => loadModels("groq"), 800);
      }
    }
  });

  els.apiKey.addEventListener("change", () => {
    if (els.rememberKey.checked) persistKey();
  });

  els.rememberKey.addEventListener("change", () => {
    if (els.rememberKey.checked) {
      persistKey();
    } else {
      try {
        for (const p of PROVIDERS) sessionStorage.removeItem(`glitch_key_${p}`);
      } catch {
        /* ignore */
      }
    }
  });

  els.toggleKey.addEventListener("click", () => {
    const showing = els.apiKey.type === "text";
    els.apiKey.type = showing ? "password" : "text";
    els.toggleKey.textContent = showing ? "SHOW" : "HIDE";
    els.toggleKey.setAttribute("aria-label", showing ? "Show API key" : "Hide API key");
  });

  els.model.addEventListener("input", () => {
    rememberModel(state.provider, els.model.value.trim());
    syncEngineLabel();
  });

  els.refreshModels.addEventListener("click", () => {
    loadModels(state.provider, { force: true });
  });

  els.sampleBtn.addEventListener("click", () => {
    els.text.value = SAMPLE;
    updateStats();
    els.text.focus();
    els.text.setSelectionRange(0, 0);
    els.text.scrollTop = 0;
  });

  els.clearBtn.addEventListener("click", () => {
    els.text.value = "";
    updateStats();
    resetView();
    els.text.focus();
  });

  els.alertClose.addEventListener("click", hideAlert);

  els.showTags.addEventListener("change", () => {
    els.doc.classList.toggle("show-tags", els.showTags.checked);
  });

  els.copyBtn.addEventListener("click", copyJson);
  els.downloadBtn.addEventListener("click", downloadJson);

  /* init */
  restoreState();
  setProvider(state.provider, { load: false });
  updateStats();
  checkHealth();
  loadModels(state.provider, { autoSelect: !state.lastModel[state.provider] });
})();
