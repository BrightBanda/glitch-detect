/* GLITCH//DETECT — front-end controller */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const MIN_KEY = 12;
  const UI_TIMEOUT_MS = 240000;
  const MIN_CHARS = Number($("text").dataset.minChars) || 80;
  const VIEWS = ["check", "humanize", "settings", "account"];

  /* Provider metadata rendered by the server (labels, defaults, key readiness). */
  const PROVIDER_META = {};
  document.querySelectorAll("#providerData [data-provider]").forEach((el) => {
    PROVIDER_META[el.dataset.provider] = { ...el.dataset, ready: Boolean(el.dataset.ready) };
  });
  const PROVIDERS = Object.keys(PROVIDER_META);

  const els = {
    views: Array.from(document.querySelectorAll(".view")),
    navLinks: Array.from(document.querySelectorAll(".sb-link")),
    setupDot: $("setupDot"),
    nodeStatus: $("nodeStatus"),
    nodeLabel: $("nodeLabel"),
    greeting: $("greeting"),
    // check
    text: $("text"),
    textStats: $("textStats"),
    analyzeBtn: $("analyzeBtn"),
    sampleBtn: $("sampleBtn"),
    clearBtn: $("clearBtn"),
    progressNote: $("progressNote"),
    editorPanel: $("editorPanel"),
    dots: $("dots"),
    alert: $("alert"),
    alertTitle: $("alertTitle"),
    alertMessage: $("alertMessage"),
    alertAction: $("alertAction"),
    alertDetailWrap: $("alertDetailWrap"),
    alertDetail: $("alertDetail"),
    alertClose: $("alertClose"),
    results: $("results"),
    emptyState: $("emptyState"),
    scoreboard: $("scoreboard"),
    gaugeArc: $("gaugeArc"),
    scoreValue: $("scoreValue"),
    verdict: $("verdict"),
    verdictNote: $("verdictNote"),
    countAI: $("countAI"),
    countRefined: $("countRefined"),
    countHuman: $("countHuman"),
    notice: $("notice"),
    doc: $("doc"),
    showTags: $("showTags"),
    downloadBtn: $("downloadBtn"),
    // settings
    providerChoices: Array.from(document.querySelectorAll("[data-provider-choice]")),
    model: $("model"),
    modelHint: $("modelHint"),
    refreshModels: $("refreshModels"),
    optFallback: $("optFallback"),
    optCrossProvider: $("optCrossProvider"),
    optCompact: $("optCompact"),
    optBalance: $("optBalance"),
    rememberKeys: $("rememberKeys"),
    chunkWords: $("chunkWords"),
    concurrency: $("concurrency"),
    // account
    avatar: $("avatar"),
    displayName: $("displayName"),
    statScans: $("statScans"),
    statWords: $("statWords"),
    statLast: $("statLast"),
    clearData: $("clearData"),
  };

  const DEFAULT_SETTINGS = {
    provider: PROVIDERS[0] || "openrouter",
    models: {},
    fallback: true,
    crossProvider: true,
    chunkWords: 900,
    concurrency: 0,
    compact: true,
    balance: true,
    rememberKeys: false,
  };

  const state = {
    settings: { ...DEFAULT_SETTINGS },
    keys: Object.fromEntries(PROVIDERS.map((p) => [p, ""])),
    profile: { name: "", scans: 0, words: 0, last: 0 },
    analysis: null,
    busy: false,
    dotTimer: null,
    noteTimer: null,
    keyTimer: null,
    modelsRequest: 0,
  };

  const SAMPLE = `In today's rapidly evolving digital landscape, artificial intelligence plays a crucial role in shaping the future of human communication. By leveraging advanced algorithms, organizations can foster innovation, streamline workflows, and unlock unprecedented levels of efficiency. Moreover, these systems demonstrate a strong commitment to adaptability in an increasingly complex world.

I tried the beta on a Tuesday and it ate four hours of my afternoon. The export button silently dropped every table with a merged cell, which I only noticed because a colleague asked why the revenue column was empty. We fixed it by pasting into a spreadsheet and re-importing, a workaround so stupid that I assumed it would be temporary. It has been two years.

Ultimately, the integration of intelligent systems serves as a testament to the transformative power of technology, ensuring that stakeholders remain well-positioned to navigate the challenges of tomorrow.`;

  /* ─────────────────────────── storage ─────────────────────────── */
  const store = {
    get(area, key) {
      try {
        return window[area].getItem(key);
      } catch {
        return null;
      }
    },
    set(area, key, value) {
      try {
        window[area].setItem(key, value);
      } catch {
        /* storage disabled */
      }
    },
    remove(area, key) {
      try {
        window[area].removeItem(key);
      } catch {
        /* storage disabled */
      }
    },
  };

  function readJson(key, fallback) {
    try {
      return { ...fallback, ...JSON.parse(store.get("localStorage", key) || "{}") };
    } catch {
      return { ...fallback };
    }
  }

  function load() {
    state.settings = readJson("glitch_settings", DEFAULT_SETTINGS);
    if (!PROVIDERS.includes(state.settings.provider)) state.settings.provider = DEFAULT_SETTINGS.provider;
    state.profile = readJson("glitch_profile", state.profile);
    if (state.settings.rememberKeys) {
      for (const p of PROVIDERS) state.keys[p] = store.get("sessionStorage", `glitch_key_${p}`) || "";
    }
  }

  const saveSettings = () => store.set("localStorage", "glitch_settings", JSON.stringify(state.settings));
  const saveProfile = () => store.set("localStorage", "glitch_profile", JSON.stringify(state.profile));

  function persistKeys() {
    for (const p of PROVIDERS) {
      const key = state.keys[p].trim();
      if (state.settings.rememberKeys && key) store.set("sessionStorage", `glitch_key_${p}`, key);
      else store.remove("sessionStorage", `glitch_key_${p}`);
    }
  }

  /* ─────────────────────────── keys ─────────────────────────── */
  const userKey = (p) => (state.keys[p] || "").trim();

  /** "custom" (user's key) | "ready" (set up on this computer) | "missing" */
  function keySource(p) {
    if (userKey(p).length >= MIN_KEY) return "custom";
    if (PROVIDER_META[p]?.ready) return "ready";
    return "missing";
  }

  /** Only keys the user typed are sent; built-in keys never leave the server. */
  function customKeys() {
    const out = {};
    for (const p of PROVIDERS) if (keySource(p) === "custom") out[p] = userKey(p);
    return out;
  }

  const currentModel = () => state.settings.models[state.settings.provider] || PROVIDER_META[state.settings.provider].defaultModel;

  /* ─────────────────────────── navigation ─────────────────────────── */
  function showView(name) {
    const view = VIEWS.includes(name) ? name : "check";
    els.views.forEach((v) => (v.hidden = v.dataset.view !== view));
    els.navLinks.forEach((link) => {
      const active = link.dataset.view === view;
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    if (view === "account") renderProfile();
    window.scrollTo(0, 0);
  }

  /* ─────────────────────────── settings view ─────────────────────────── */
  function refreshSetupState() {
    const s = state.settings;
    els.providerChoices.forEach((btn) => {
      const p = btn.dataset.providerChoice;
      btn.setAttribute("aria-checked", p === s.provider ? "true" : "false");
      const status = btn.querySelector("[data-status]");
      const source = keySource(p);
      status.dataset.state = source === "missing" ? "needs-key" : "ready";
      status.textContent =
        source === "custom" ? "✓ Using your key" : source === "ready" ? "✓ Ready to use" : "Needs a key — add one below";
    });
    els.setupDot.hidden = keySource(s.provider) !== "missing";
  }

  function fillSettings() {
    const s = state.settings;
    els.optFallback.checked = s.fallback;
    els.optCrossProvider.checked = s.crossProvider;
    els.optCrossProvider.disabled = !s.fallback;
    els.rememberKeys.checked = s.rememberKeys;
    els.optCompact.checked = s.compact;
    els.optBalance.checked = s.balance;
    els.chunkWords.value = s.chunkWords;
    els.concurrency.value = String([0, 1, 4].includes(s.concurrency) ? s.concurrency : 0);
    document.querySelectorAll("[data-key-input]").forEach((input) => {
      input.value = state.keys[input.dataset.keyInput] || "";
    });
    refreshSetupState();
  }

  function setProvider(provider) {
    if (!PROVIDER_META[provider] || state.busy) return;
    state.settings.provider = provider;
    saveSettings();
    refreshSetupState();
    loadModels();
  }

  const GROUP_NAMES = { recommended: "Recommended", specialized: "Experimental", general: "Other free models" };

  function renderModelOptions(models, selected) {
    const fragment = document.createDocumentFragment();
    const groups = new Map();
    for (const m of models) {
      const name = GROUP_NAMES[m.group] || "Free models";
      if (!groups.has(name)) groups.set(name, []);
      groups.get(name).push(m);
    }
    for (const [name, items] of groups) {
      const group = document.createElement("optgroup");
      group.label = name;
      for (const m of items) {
        const option = document.createElement("option");
        option.value = m.id;
        option.textContent = (m.name || m.id).replace(/\s*\(free\)\s*$/i, "");
        group.appendChild(option);
      }
      fragment.appendChild(group);
    }
    els.model.replaceChildren(fragment);
    els.model.value = selected;
  }

  async function loadModels({ force = false } = {}) {
    const provider = state.settings.provider;
    const request = ++state.modelsRequest; // ignore stale responses
    els.refreshModels.disabled = true;
    els.modelHint.textContent = "Loading models…";

    try {
      const res = await fetch("/api/models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider, api_keys: customKeys(), force }),
      });
      const data = await res.json().catch(() => null);
      if (!res.ok || !data || !data.ok) throw new Error(res.status === 401 ? "key" : "load");
      if (request !== state.modelsRequest) return;

      const models = Array.isArray(data.models) ? data.models : [];
      let selected = state.settings.models[provider];
      if (!models.some((m) => m.id === selected)) selected = data.default || models[0]?.id;
      if (selected) {
        state.settings.models[provider] = selected;
        saveSettings();
      }
      renderModelOptions(models, selected);
      els.modelHint.textContent = "All models listed here are free to use. The first one is our recommendation.";
    } catch (err) {
      if (request !== state.modelsRequest) return;
      els.modelHint.textContent =
        err.message === "key"
          ? "That API key didn't work, so we couldn't load the list. Check it below."
          : "Couldn't load the list right now. The recommended model will be used.";
    } finally {
      if (request === state.modelsRequest) els.refreshModels.disabled = false;
    }
  }

  function readOptions() {
    const words = Number(els.chunkWords.value);
    Object.assign(state.settings, {
      fallback: els.optFallback.checked,
      crossProvider: els.optCrossProvider.checked,
      rememberKeys: els.rememberKeys.checked,
      compact: els.optCompact.checked,
      balance: els.optBalance.checked,
      chunkWords: Number.isFinite(words) ? Math.min(1800, Math.max(200, words)) : 900,
      concurrency: Number(els.concurrency.value) || 0,
    });
    els.chunkWords.value = state.settings.chunkWords;
    els.optCrossProvider.disabled = !state.settings.fallback;
    saveSettings();
    persistKeys();
  }

  /* ─────────────────────────── account view ─────────────────────────── */
  function renderProfile() {
    const { name, scans, words, last } = state.profile;
    els.displayName.value = name;
    els.avatar.textContent = name ? name.trim()[0].toUpperCase() : "?";
    els.statScans.textContent = scans.toLocaleString();
    els.statWords.textContent = words.toLocaleString();
    els.statLast.textContent = last ? new Date(last).toLocaleDateString(undefined, { month: "short", day: "numeric" }) : "—";
    els.greeting.textContent = name ? `Welcome back, ${name}` : "AI writing checker";
  }

  /* ─────────────────────────── alerts ─────────────────────────── */
  function showAlert(title, message, { detail = "", action = false, tone = "error" } = {}) {
    els.alert.dataset.tone = tone;
    els.alertTitle.textContent = title;
    els.alertMessage.textContent = message;
    els.alertAction.hidden = !action;
    els.alertDetail.textContent = detail;
    els.alertDetailWrap.hidden = !detail;
    els.alert.hidden = false;
  }
  const hideAlert = () => (els.alert.hidden = true);

  /** Plain-language explanation for a failed check. */
  function explainError(status, error) {
    const detail = [error.message, error.detail].filter(Boolean).join("\n");
    const friendly = {
      401: ["That API key didn't work", "Check the key in Settings, or remove it to use the built-in one.", true],
      402: ["The AI service declined the request", "The account behind this key may be out of credits.", true],
      404: ["That model isn't available anymore", "Pick a different model in Settings.", true],
      429: ["The service is busy right now", "Free AI services limit how often they can be used. Wait a minute and try again.", false],
      504: ["This is taking too long", "Try again, or check a shorter piece of text.", false],
    };
    if (friendly[status]) {
      const [title, message, action] = friendly[status];
      return { title, message, action, detail };
    }
    if (status === 400 && /key/i.test(error.message || "")) {
      return { title: "One quick setup step", message: "Add an API key in Settings to start checking.", action: true, detail };
    }
    if (status === 400 || status === 413) {
      return { title: "We couldn't check this text", message: error.message || "Please try again.", action: false, detail: "" };
    }
    return { title: "Something went wrong", message: "The AI service had a problem. Please try again in a moment.", action: false, detail };
  }

  /* ─────────────────────────── loading ─────────────────────────── */
  function startLoading() {
    state.busy = true;
    els.analyzeBtn.disabled = true;
    els.analyzeBtn.classList.add("loading");
    els.editorPanel.classList.add("scanning");
    hideAlert();
    let n = 0;
    state.dotTimer = setInterval(() => {
      n = (n + 1) % 4;
      els.dots.textContent = ".".repeat(n);
    }, 420);
    state.noteTimer = setTimeout(() => (els.progressNote.hidden = false), 4000);
  }

  function stopLoading() {
    state.busy = false;
    els.analyzeBtn.disabled = false;
    els.analyzeBtn.classList.remove("loading");
    els.editorPanel.classList.remove("scanning");
    els.progressNote.hidden = true;
    clearInterval(state.dotTimer);
    clearTimeout(state.noteTimer);
  }

  /* ─────────────────────────── results ─────────────────────────── */
  const levelFor = (score) => (score >= 67 ? "high" : score >= 34 ? "mid" : "low");

  function verdictFor(score) {
    if (score >= 85) return "Very likely AI-written";
    if (score >= 67) return "Mostly AI-written";
    if (score >= 45) return "A mix of AI and human writing";
    if (score >= 22) return "Mostly human-written";
    return "Likely human-written";
  }

  function animateNumber(target) {
    const duration = 700;
    if (document.hidden || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      els.scoreValue.textContent = target;
      return;
    }
    const start = performance.now();
    const from = Number(els.scoreValue.textContent) || 0;
    // rAF is throttled in background tabs; never leave the gauge mid-count.
    const guard = setTimeout(() => (els.scoreValue.textContent = target), duration + 250);
    const step = (now) => {
      const t = Math.min(1, (now - start) / duration);
      els.scoreValue.textContent = Math.round(from + (target - from) * (1 - Math.pow(1 - t, 3)));
      if (t < 1) requestAnimationFrame(step);
      else clearTimeout(guard);
    };
    requestAnimationFrame(step);
  }

  function render(result) {
    state.analysis = result;
    const score = Number(result.score) || 0;
    const { AI, REFINED, HUMAN } = result.counts;

    els.scoreboard.dataset.level = levelFor(score);
    els.gaugeArc.style.strokeDashoffset = String(100 - score);
    animateNumber(score);

    els.verdict.textContent = verdictFor(score);
    const flagged = AI + REFINED;
    els.verdictNote.textContent =
      flagged === 0
        ? `None of the ${result.total} sentences look AI-written.`
        : `${flagged} of ${result.total} sentence${result.total === 1 ? "" : "s"} look AI-written or AI-polished.`;

    els.countAI.textContent = AI;
    els.countRefined.textContent = REFINED;
    els.countHuman.textContent = HUMAN;

    const notes = [];
    if (result.fallbacks && result.fallbacks.length) {
      notes.push("The AI service was busy, so a backup model finished your check.");
    }
    if (result.cached_chunks && result.cached_chunks === result.chunks) {
      notes.push("You'd already checked this text, so the earlier result was re-used instantly.");
    } else if (result.cached_chunks) {
      notes.push("Unchanged parts were re-used from your last check.");
    }
    if ((result.warnings || []).some((w) => w.includes("Sentence boundaries"))) {
      notes.push("Some sentences may be grouped a little differently from your original.");
    }
    els.notice.textContent = notes.join(" ");
    els.notice.hidden = notes.length === 0;

    els.doc.innerHTML = result.html; // escaped by the server
    els.doc.classList.toggle("show-tags", els.showTags.checked);

    els.results.hidden = false;
    els.emptyState.hidden = true;
    els.results.scrollIntoView({ behavior: "smooth", block: "start" });

    state.profile.scans += 1;
    state.profile.words += result.words || 0;
    state.profile.last = Date.now();
    saveProfile();
  }

  /* ─────────────────────────── check ─────────────────────────── */
  async function analyze() {
    if (state.busy) return;
    const text = els.text.value.trim();
    const provider = state.settings.provider;
    hideAlert();

    if (keySource(provider) === "missing") {
      showAlert("One quick setup step", "Add an API key in Settings to start checking.", { action: true, tone: "info" });
      return;
    }
    if (text.length < MIN_CHARS) {
      showAlert("A little more text, please", "Paste at least a couple of full sentences so there's enough to go on.", { tone: "info" });
      els.text.focus();
      return;
    }

    startLoading();
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), UI_TIMEOUT_MS);
    const s = state.settings;

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          provider,
          model: currentModel(),
          api_keys: customKeys(),
          chunk_words: s.chunkWords,
          concurrency: s.concurrency,
          fallback: s.fallback,
          cross_provider: s.crossProvider,
          compact: s.compact,
          balance: s.balance,
        }),
        signal: controller.signal,
      });
      const data = await res.json().catch(() => null);

      if (!res.ok || !data || !data.ok) {
        const { title, message, action, detail } = explainError(res.status, (data && data.error) || {});
        showAlert(title, message, { action, detail });
        return;
      }
      render(data.result);
    } catch (err) {
      if (err.name === "AbortError") {
        showAlert("This is taking too long", "Try again, or check a shorter piece of text.");
      } else {
        showAlert("Can't reach the app", "Glitch//Detect seems to have stopped running on this computer. Restart it and try again.");
      }
    } finally {
      clearTimeout(timeout);
      stopLoading();
    }
  }

  function downloadReport() {
    const r = state.analysis;
    if (!r) return;
    const names = { AI: "Likely AI", REFINED: "AI-polished", HUMAN: "Human" };
    const lines = [
      "GLITCH//DETECT REPORT",
      new Date().toLocaleString(),
      "",
      `Score: ${r.score}% likely AI — ${verdictFor(r.score)}`,
      `Likely AI: ${r.counts.AI}   AI-polished: ${r.counts.REFINED}   Human: ${r.counts.HUMAN}   (of ${r.total} sentences)`,
      "",
      ...r.sentences.map((s) => `[${names[s.classification]}] ${s.text}`),
      "",
      "A score is a hint, not proof.",
    ];
    const url = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/plain" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `glitch-report-${new Date().toISOString().slice(0, 10)}.txt`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  /* ─────────────────────────── wire up ─────────────────────────── */
  let statsFrame = 0;
  function updateStats() {
    cancelAnimationFrame(statsFrame);
    statsFrame = requestAnimationFrame(() => {
      const words = els.text.value.trim() ? els.text.value.trim().split(/\s+/).length : 0;
      els.textStats.textContent = `${words.toLocaleString()} word${words === 1 ? "" : "s"}`;
    });
  }

  window.addEventListener("hashchange", () => showView(location.hash.slice(1)));

  els.text.addEventListener("input", updateStats);
  els.text.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      analyze();
    }
  });
  els.analyzeBtn.addEventListener("click", analyze);
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
    els.results.hidden = true;
    els.emptyState.hidden = false;
    hideAlert();
    state.analysis = null;
    els.text.focus();
  });
  els.alertClose.addEventListener("click", hideAlert);
  els.showTags.addEventListener("change", () => els.doc.classList.toggle("show-tags", els.showTags.checked));
  els.downloadBtn.addEventListener("click", downloadReport);

  els.providerChoices.forEach((btn) => btn.addEventListener("click", () => setProvider(btn.dataset.providerChoice)));
  els.model.addEventListener("change", () => {
    state.settings.models[state.settings.provider] = els.model.value;
    saveSettings();
  });
  els.refreshModels.addEventListener("click", () => loadModels({ force: true }));
  [els.optFallback, els.optCrossProvider, els.optCompact, els.optBalance, els.rememberKeys, els.concurrency, els.chunkWords].forEach((el) =>
    el.addEventListener("change", readOptions)
  );

  document.querySelectorAll("[data-key-input]").forEach((input) => {
    input.addEventListener("input", () => {
      const p = input.dataset.keyInput;
      state.keys[p] = input.value.trim();
      persistKeys();
      refreshSetupState();
      if (p === state.settings.provider) {
        clearTimeout(state.keyTimer);
        state.keyTimer = setTimeout(loadModels, 800);
      }
    });
  });
  document.querySelectorAll("[data-key-toggle]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const input = document.querySelector(`[data-key-input="${btn.dataset.keyToggle}"]`);
      const showing = input.type === "text";
      input.type = showing ? "password" : "text";
      btn.textContent = showing ? "Show" : "Hide";
    });
  });
  document.querySelectorAll("[data-key-clear]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const input = document.querySelector(`[data-key-input="${btn.dataset.keyClear}"]`);
      input.value = "";
      input.dispatchEvent(new Event("input"));
    });
  });

  els.displayName.addEventListener("input", () => {
    state.profile.name = els.displayName.value.trim();
    saveProfile();
    renderProfile();
  });
  els.clearData.addEventListener("click", () => {
    if (!window.confirm("Clear your name, activity, keys and settings from this device?")) return;
    for (const area of ["localStorage", "sessionStorage"]) {
      try {
        Object.keys(window[area])
          .filter((k) => k.startsWith("glitch_"))
          .forEach((k) => window[area].removeItem(k));
      } catch {
        /* storage disabled */
      }
    }
    location.reload();
  });

  /* ─────────────────────────── init ─────────────────────────── */
  async function checkHealth() {
    try {
      const res = await fetch("/api/health", { cache: "no-store" });
      els.nodeStatus.dataset.state = res.ok ? "online" : "offline";
      els.nodeLabel.textContent = res.ok ? "Ready" : "Offline";
    } catch {
      els.nodeStatus.dataset.state = "offline";
      els.nodeLabel.textContent = "Offline";
    }
  }

  load();
  fillSettings();
  renderProfile();
  updateStats();
  showView(location.hash.slice(1));
  checkHealth();
  loadModels();
})();
