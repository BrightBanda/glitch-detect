/* GLITCH//DETECT — front-end controller */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const MIN_KEY = 12;
  const UI_TIMEOUT_MS = 240000;
  // Defaults until /api/config answers (see loadConfig).
  let MIN_CHARS = 80;
  let MAX_WORDS = 2500;
  let MAX_REWRITE_WORDS = 1500;
  let PREMIUM_REWRITE_WORDS = 1500;
  let DEFAULT_MODEL = "";
  let TONE_LABELS = {};
  const VIEWS = ["check", "humanize", "settings", "account"];

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
    limitNote: $("limitNote"),
    analyzeBtn: $("analyzeBtn"),
    sampleBtn: $("sampleBtn"),
    clearBtn: $("clearBtn"),
    progressNote: $("progressNote"),
    editorPanel: $("editorPanel"),
    alert: $("alert"),
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
    humanizeFlagged: $("humanizeFlagged"),
    downloadBtn: $("downloadBtn"),
    // humanize
    humanizePanel: $("humanizePanel"),
    humanizeInput: $("humanizeInput"),
    humanizeStats: $("humanizeStats"),
    askQuestions: $("askQuestions"),
    questionList: $("questionList"),
    questionsEmpty: $("questionsEmpty"),
    downloadTrail: $("downloadTrail"),
    humanizeLimitNote: $("humanizeLimitNote"),
    useLastCheck: $("useLastCheck"),
    humanizeTone: $("humanizeTone"),
    humanizeVoice: $("humanizeVoice"),
    tryAgain: $("tryAgain"),
    styleShift: $("styleShift"),
    rhythmBefore: $("rhythmBefore"),
    rhythmAfter: $("rhythmAfter"),
    phrasesBefore: $("phrasesBefore"),
    phrasesAfter: $("phrasesAfter"),
    groundingTips: $("groundingTips"),
    tipList: $("tipList"),
    onlyFlagged: $("onlyFlagged"),
    secondPass: $("secondPass"),
    humanizeBtn: $("humanizeBtn"),
    hAlert: $("hAlert"),
    humanizeResult: $("humanizeResult"),
    scoreShift: $("scoreShift"),
    shiftFrom: $("shiftFrom"),
    shiftTo: $("shiftTo"),
    rewriteNote: $("rewriteNote"),
    humanizeOutput: $("humanizeOutput"),
    copyRewrite: $("copyRewrite"),
    checkRewrite: $("checkRewrite"),
    // settings
    providerChoices: $("providerChoices"),
    noProviders: $("noProviders"),
    model: $("model"),
    modelHint: $("modelHint"),
    refreshModels: $("refreshModels"),
    optFallback: $("optFallback"),
    optCrossProvider: $("optCrossProvider"),
    optCompact: $("optCompact"),
    optBalance: $("optBalance"),
    newKey: $("newKey"),
    toggleNewKey: $("toggleNewKey"),
    addKey: $("addKey"),
    keyFeedback: $("keyFeedback"),
    keyList: $("keyList"),
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
    nameHint: $("nameHint"),
    accountSub: $("accountSub"),
    authCard: $("authCard"),
    authTitle: $("authTitle"),
    authTabs: Array.from(document.querySelectorAll("[data-auth]")),
    authEmail: $("authEmail"),
    authNameRow: $("authNameRow"),
    authName: $("authName"),
    authPassword: $("authPassword"),
    authHint: $("authHint"),
    authSubmit: $("authSubmit"),
    authFeedback: $("authFeedback"),
    planCard: $("planCard"),
    planBadge: $("planBadge"),
    planLine: $("planLine"),
    usageChecks: $("usageChecks"),
    usageChecksLabel: $("usageChecksLabel"),
    usageRewrites: $("usageRewrites"),
    usageRewritesLabel: $("usageRewritesLabel"),
    usageWords: $("usageWords"),
    usageRewriteWords: $("usageRewriteWords"),
    signOutBtn: $("signOutBtn"),
    currentPassword: $("currentPassword"),
    newPassword: $("newPassword"),
    changePasswordBtn: $("changePasswordBtn"),
    passwordFeedback: $("passwordFeedback"),
    deletePassword: $("deletePassword"),
    deleteAccountBtn: $("deleteAccountBtn"),
    deleteFeedback: $("deleteFeedback"),
    rewriteNudge: $("rewriteNudge"),
    rewritesLeft: $("rewritesLeft"),
  };

  const DEFAULT_SETTINGS = {
    provider: "openrouter",
    models: {},
    fallback: true,
    crossProvider: true,
    chunkWords: 900,
    concurrency: 0,
    compact: true,
    balance: true,
    rememberKeys: false,
    tone: "natural",
    voice: "personal",
  };

  const state = {
    providers: {}, // id -> {id, label, billing, keys_url, default_model, ready}
    settings: { ...DEFAULT_SETTINGS },
    keys: {}, // provider id -> the user's own key
    profile: { name: "", scans: 0, words: 0, last: 0 },
    analysis: null,
    lastCheck: null, // {text, sentences} from the most recent check
    rewrite: null,
    busy: false,
    keyTimer: null,
    modelsRequest: 0,
    me: null, // /api/me: { user | null, plan, usage }
    authMode: "signup",
    nameTimer: null,
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

  function readJson(area, key, fallback) {
    try {
      return { ...fallback, ...JSON.parse(store.get(area, key) || "{}") };
    } catch {
      return { ...fallback };
    }
  }

  function load() {
    state.settings = readJson("localStorage", "glitch_settings", DEFAULT_SETTINGS);
    state.profile = readJson("localStorage", "glitch_profile", state.profile);
    if (state.settings.rememberKeys) state.keys = readJson("sessionStorage", "glitch_keys", {});
  }

  const saveSettings = () => store.set("localStorage", "glitch_settings", JSON.stringify(state.settings));
  const saveProfile = () => store.set("localStorage", "glitch_profile", JSON.stringify(state.profile));

  function persistKeys() {
    if (state.settings.rememberKeys && Object.keys(state.keys).length) {
      store.set("sessionStorage", "glitch_keys", JSON.stringify(state.keys));
    } else {
      store.remove("sessionStorage", "glitch_keys");
    }
  }

  /* ─────────────────────────── utilities ─────────────────────────── */
  const countWords = (value) => (value.trim() ? value.trim().split(/\s+/).length : 0);
  const plural = (n, word) => `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`;

  async function postJson(url, payload, { timeout = UI_TIMEOUT_MS } = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      const data = await res.json().catch(() => null);
      return { res, data };
    } finally {
      clearTimeout(timer);
    }
  }

  /* ─────────────────────────── keys + providers ─────────────────────────── */
  const userKey = (p) => (state.keys[p] || "").trim();

  /** "custom" (user's key) | "ready" (set up on the server) | "missing" */
  function keySource(p) {
    if (userKey(p).length >= MIN_KEY) return "custom";
    if (state.providers[p]?.ready) return "ready";
    return "missing";
  }

  const availableProviders = () => Object.keys(state.providers).filter((p) => keySource(p) !== "missing");

  /** Only keys the user added travel with requests; server keys never leave it. */
  function customKeys() {
    const out = {};
    for (const p of Object.keys(state.keys)) if (keySource(p) === "custom") out[p] = userKey(p);
    return out;
  }

  function routingOptions() {
    const s = state.settings;
    return {
      provider: s.provider,
      model: currentModel(),
      api_keys: customKeys(),
      chunk_words: s.chunkWords,
      concurrency: s.concurrency,
      fallback: s.fallback,
      cross_provider: s.crossProvider,
      compact: s.compact,
      balance: s.balance,
    };
  }

  function currentModel() {
    const p = state.settings.provider;
    return state.settings.models[p] || state.providers[p]?.default_model || DEFAULT_MODEL || "";
  }

  async function loadProviders() {
    try {
      const res = await fetch("/api/providers", { cache: "no-store" });
      const data = await res.json();
      state.providers = Object.fromEntries((data.providers || []).map((p) => [p.id, p]));
      els.nodeStatus.dataset.state = "online";
      els.nodeLabel.textContent = "Ready";
    } catch {
      els.nodeStatus.dataset.state = "offline";
      els.nodeLabel.textContent = "Offline";
      return;
    }
    ensureProvider();
    renderProviders();
    renderKeys();
    loadModels();
  }

  /** Keep the selected service usable: fall back to the first available one. */
  function ensureProvider() {
    const available = availableProviders();
    if (!available.includes(state.settings.provider) && available.length) {
      state.settings.provider = available[0];
      saveSettings();
    }
  }

  const BILLING_NOTE = {
    free: "Free models only.",
    "free-tier": "Uses this key's free allowance.",
    paid: "Billed to the key's account.",
  };

  function renderProviders() {
    const available = availableProviders();
    els.providerChoices.replaceChildren(
      ...available.map((id) => {
        const p = state.providers[id];
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "choice";
        btn.setAttribute("role", "radio");
        btn.setAttribute("aria-checked", id === state.settings.provider ? "true" : "false");
        const title = document.createElement("span");
        title.className = "choice-title";
        title.textContent = p.label;
        const desc = document.createElement("span");
        desc.className = "choice-desc";
        desc.textContent = BILLING_NOTE[p.billing] || "";
        const status = document.createElement("span");
        status.className = "choice-status";
        status.dataset.state = "ready";
        status.textContent = keySource(id) === "custom" ? "✓ Using your key" : "✓ Ready to use";
        btn.append(title, desc, status);
        btn.addEventListener("click", () => setProvider(id));
        return btn;
      })
    );
    els.noProviders.hidden = available.length > 0;
    els.setupDot.hidden = available.length > 0;
  }

  function renderKeys() {
    const items = Object.keys(state.keys)
      .filter((p) => userKey(p) && state.providers[p])
      .map((p) => {
        const li = document.createElement("li");
        const name = document.createElement("span");
        name.className = "key-provider";
        name.textContent = state.providers[p].label;
        const masked = document.createElement("span");
        masked.className = "key-mask";
        const key = userKey(p);
        masked.textContent = `${key.slice(0, 4)}…${key.slice(-4)}`;
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "mini-btn";
        remove.textContent = "Remove";
        remove.addEventListener("click", () => {
          delete state.keys[p];
          persistKeys();
          afterKeysChanged();
        });
        li.append(name, masked, remove);
        return li;
      });
    els.keyList.replaceChildren(...items);
  }

  function afterKeysChanged() {
    ensureProvider();
    renderProviders();
    renderKeys();
    loadModels();
  }

  function keyFeedback(message, tone) {
    els.keyFeedback.textContent = message;
    els.keyFeedback.dataset.state = tone;
    els.keyFeedback.hidden = !message;
  }

  async function addKey() {
    const key = els.newKey.value.trim();
    if (!key) return;
    els.addKey.disabled = true;
    keyFeedback("Checking which service this key is for…", "info");
    try {
      const { res, data } = await postJson("/api/keys/identify", { key }, { timeout: 30000 });
      if (!res.ok || !data || !data.ok) {
        keyFeedback((data && data.error && data.error.message) || "That key couldn't be added.", "error");
        return;
      }
      state.keys[data.provider] = key;
      state.settings.provider = data.provider;
      saveSettings();
      persistKeys();
      els.newKey.value = "";
      keyFeedback(
        data.verified ? `✓ ${data.label} key added and working.` : `✓ Recognised as a ${data.label} key.`,
        "ok"
      );
      afterKeysChanged();
    } catch {
      keyFeedback("Couldn't reach the app to check the key.", "error");
    } finally {
      els.addKey.disabled = false;
    }
  }

  /* ─────────────────────────── settings view ─────────────────────────── */
  function fillSettings() {
    const s = state.settings;
    els.optFallback.checked = s.fallback;
    els.optCrossProvider.checked = s.crossProvider;
    els.optCrossProvider.disabled = !s.fallback;
    els.optCompact.checked = s.compact;
    els.optBalance.checked = s.balance;
    els.rememberKeys.checked = s.rememberKeys;
    els.chunkWords.value = s.chunkWords;
    els.concurrency.value = String([0, 1, 4].includes(s.concurrency) ? s.concurrency : 0);
    els.humanizeTone.value = s.tone;
    els.humanizeVoice.value = s.voice;
  }

  function readOptions() {
    const words = Number(els.chunkWords.value);
    Object.assign(state.settings, {
      fallback: els.optFallback.checked,
      crossProvider: els.optCrossProvider.checked,
      compact: els.optCompact.checked,
      balance: els.optBalance.checked,
      rememberKeys: els.rememberKeys.checked,
      chunkWords: Number.isFinite(words) ? Math.min(1800, Math.max(200, words)) : 900,
      concurrency: Number(els.concurrency.value) || 0,
    });
    els.chunkWords.value = state.settings.chunkWords;
    els.optCrossProvider.disabled = !state.settings.fallback;
    saveSettings();
    persistKeys();
  }

  function setProvider(provider) {
    if (!state.providers[provider] || state.busy) return;
    state.settings.provider = provider;
    saveSettings();
    renderProviders();
    loadModels();
  }

  const GROUP_NAMES = { recommended: "Recommended", specialized: "Experimental", general: "Other models" };

  function renderModelOptions(models, selected) {
    const fragment = document.createDocumentFragment();
    const groups = new Map();
    for (const m of models) {
      const name = GROUP_NAMES[m.group] || "Models";
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
    if (keySource(provider) === "missing") return;
    const request = ++state.modelsRequest; // ignore stale responses
    els.refreshModels.disabled = true;
    els.modelHint.textContent = "Loading models…";
    try {
      const { res, data } = await postJson(
        "/api/models",
        { provider, api_keys: customKeys(), force },
        { timeout: 30000 }
      );
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
      els.modelHint.textContent = `${BILLING_NOTE[data.billing] || ""} The first model is our recommendation.`;
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
    if (view === "account") {
      renderProfile();
      loadMe(); // plan or usage may have changed elsewhere (e.g. an admin grant)
    }
    if (view === "humanize") els.useLastCheck.hidden = !state.lastCheck;
    window.scrollTo(0, 0);
  }

  /* ─────────────────────────── account view ─────────────────────────── */
  function renderProfile() {
    const { scans, words, last } = state.profile;
    const account = state.me && state.me.user;
    const name = account ? account.name || "" : state.profile.name;
    els.displayName.value = name;
    els.avatar.textContent = name ? name.trim()[0].toUpperCase() : "?";
    els.statScans.textContent = scans.toLocaleString();
    els.statWords.textContent = words.toLocaleString();
    els.statLast.textContent = last ? new Date(last).toLocaleDateString(undefined, { month: "short", day: "numeric" }) : "—";
    els.greeting.textContent = name ? `Welcome back, ${name}` : "AI writing checker";
  }

  /* ─────────────────────────── alerts ─────────────────────────── */
  function showAlert(box, title, message, { detail = "", action = false, tone = "error" } = {}) {
    box.dataset.tone = tone;
    box.querySelector(".alert-title").textContent = title;
    box.querySelector(".alert-message").textContent = message;
    // `action`: true → Settings link, or { href, text } for another page.
    const link = box.querySelector(".alert-action");
    link.hidden = !action;
    link.href = (action && action.href) || "#settings";
    link.textContent = (action && action.text) || "Open settings →";
    box.querySelector(".alert-detail").textContent = detail;
    box.querySelector(".alert-detail-wrap").hidden = !detail;
    box.hidden = false;
  }
  const hideAlert = (box) => (box.hidden = true);

  /** Plain-language explanation for a failed request. */
  const ACCOUNT_LINK = { href: "#account", text: "Go to your account →" };

  function explainError(status, error, verb = "check") {
    const detail = [error.message, error.detail].filter(Boolean).join("\n");
    const byCode = {
      auth_required: ["Rewriting needs a free account", ACCOUNT_LINK],
      quota: ["You've reached today's limit", ACCOUNT_LINK],
      plan_words: ["Too long for your plan", ACCOUNT_LINK],
      plan_tone: ["That tone is part of Premium", ACCOUNT_LINK],
      suspended: ["This account is suspended", false],
    };
    if (byCode[error.code]) {
      const [title, action] = byCode[error.code];
      if (error.code === "auth_required" || error.code === "suspended") loadMe();
      return { title, message: error.message, action, detail: "", tone: "info" };
    }
    const friendly = {
      401: ["That API key didn't work", "Check the key in Settings, or remove it to use the built-in one.", true],
      402: ["The AI service declined the request", "The account behind this key may be out of credits.", true],
      404: ["That model isn't available anymore", "Pick a different model in Settings.", true],
      429: ["The service is busy right now", "Free AI services limit how often they can be used. Wait a minute and try again.", false],
      504: ["This is taking too long", `Try again, or ${verb} a shorter piece of text.`, false],
    };
    if (friendly[status]) {
      const [title, message, action] = friendly[status];
      return { title, message, action, detail };
    }
    if (status === 400 && /key/i.test(error.message || "")) {
      return { title: "One quick setup step", message: "Add an API key in Settings to get started.", action: true, detail };
    }
    if (status === 400 || status === 413) {
      return { title: `We couldn't ${verb} this text`, message: error.message || "Please try again.", action: false, detail: "" };
    }
    return { title: "Something went wrong", message: "The AI service had a problem. Please try again in a moment.", action: false, detail };
  }

  function needsSetup(box) {
    if (availableProviders().length) return false;
    showAlert(box, "One quick setup step", "Add an API key in Settings to get started.", { action: true, tone: "info" });
    return true;
  }

  /* ─────────────────────────── word limits ─────────────────────────── */
  /** Checks and rewrites have separate caps (a rewrite costs far more). */
  const limitFor = (kind) => (kind === "rewrite" ? MAX_REWRITE_WORDS : MAX_WORDS);

  function showWordCount(words, stats, note, kind = "check") {
    const limit = limitFor(kind);
    stats.textContent = `${words.toLocaleString()} / ${limit.toLocaleString()} words`;
    stats.dataset.state = words > limit ? "over" : words > limit * 0.9 ? "near" : "ok";
    const over = words - limit;
    note.hidden = over <= 0;
    if (over > 0) {
      note.textContent =
        `That's ${words.toLocaleString()} words, which is over the ${limit.toLocaleString()}-word ${kind} limit${planSuffix()}. ` +
        `Please remove about ${plural(over, "word")}, or work on it in parts.${upgradeHint(kind)}`;
    }
  }

  function overLimit(text, box, kind = "check") {
    const words = countWords(text);
    const limit = limitFor(kind);
    if (words <= limit) return false;
    const hint = upgradeHint(kind);
    showAlert(
      box,
      "This text is too long",
      `You can ${kind} up to ${limit.toLocaleString()} words at a time${planSuffix()}. This one has ${words.toLocaleString()}. Please shorten it or work on it in parts.${hint}`,
      { tone: "info", action: hint ? ACCOUNT_LINK : false }
    );
    return true;
  }

  /* ─────────────────────────── account & plan ─────────────────────────── */
  const signedIn = () => Boolean(state.me && state.me.user);
  const planOf = () => (state.me && state.me.plan) || null;

  function planSuffix() {
    const plan = planOf();
    if (!plan) return "";
    return plan.id === "guest" ? " for guests" : ` on the ${plan.label} plan`;
  }

  function upgradeHint(kind = "check") {
    const plan = planOf();
    const premium = kind === "rewrite" ? PREMIUM_REWRITE_WORDS : 2500;
    if (!plan || limitFor(kind) >= premium) return "";
    const amount = premium.toLocaleString();
    return plan.id === "guest" ? ` A free account doesn't raise this; Premium allows ${amount}.` : ` Premium allows ${amount}.`;
  }

  async function loadMe() {
    try {
      const res = await fetch("/api/me", { cache: "no-store" });
      const data = await res.json();
      if (data && data.ok) state.me = data;
    } catch {
      /* offline: keep what we had */
    }
    applyPlan();
    renderAccount();
  }

  /** Limits from the plan: word cap, allowed tones, rewrite nudges. */
  function applyPlan() {
    const plan = planOf();
    if (plan && plan.max_words) MAX_WORDS = plan.max_words;
    if (plan && plan.rewrite_words) MAX_REWRITE_WORDS = plan.rewrite_words;
    els.humanizeInput.dispatchEvent(new Event("input"));
    els.text.placeholder = `Paste an essay, article, email… anything with at least a few sentences (up to ${MAX_WORDS.toLocaleString()} words).`;
    const allowed = new Set((plan && plan.tones) || []);
    Array.from(els.humanizeTone.options).forEach((option) => {
      const label = TONE_LABELS[option.value] || option.value;
      const locked = plan && plan.id !== "guest" && !allowed.has(option.value);
      option.disabled = Boolean(locked);
      option.textContent = locked ? `${label} · Premium` : label;
    });
    if (els.humanizeTone.selectedOptions[0] && els.humanizeTone.selectedOptions[0].disabled) {
      els.humanizeTone.value = (plan && plan.tones && plan.tones[0]) || "natural";
    }
    els.rewriteNudge.hidden = signedIn();
    if (signedIn()) {
      const used = state.me.usage.rewrites || 0;
      const left = Math.max(0, plan.rewrites_per_day - used);
      els.rewritesLeft.hidden = false;
      els.rewritesLeft.textContent = `${left} of ${plan.rewrites_per_day} rewrites left today`;
      els.rewritesLeft.dataset.state = left ? "ok" : "out";
    } else {
      els.rewritesLeft.hidden = true;
    }
    updateStats();
    updateHumanizeStats();
  }

  function renderAccount() {
    const me = state.me;
    const user = me && me.user;
    els.authCard.hidden = Boolean(user);
    els.planCard.hidden = !user;
    els.accountSub.textContent = user
      ? `Signed in as ${user.email}.`
      : "Checking works without an account. A free account adds rewrites.";
    els.nameHint.textContent = user ? "Saved to your account." : "Only used to greet you. Saved on this device.";
    if (user) {
      const plan = me.plan;
      const usage = me.usage || {};
      els.planBadge.textContent = plan.label;
      els.planBadge.dataset.plan = plan.id;
      const ends = user.period_end
        ? new Date(user.period_end * 1000).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" })
        : "";
      const lines = {
        free: "Free plan. Upgrade for longer texts, more rewrites and every tone.",
        grace: `Your plan ended on ${ends}. You keep it for 48 hours while you renew.`,
        cancelled: `Cancelled. Your plan stays active until ${ends}.`,
        expired: "Your paid plan has ended, so you're on Free limits.",
        pending_payment: "Payment pending. You'll get your plan as soon as it's confirmed.",
      };
      els.planLine.textContent =
        user.status === "active" ? (ends ? `Active until ${ends}.` : "Active.") : lines[user.status] || "";
      els.usageChecks.textContent = `${usage.checks || 0}/${plan.checks_per_day}`;
      els.usageRewrites.textContent = `${usage.rewrites || 0}/${plan.rewrites_per_day}`;
      els.usageWords.textContent = plan.max_words.toLocaleString();
      els.usageRewriteWords.textContent = plan.rewrite_words.toLocaleString();
    }
    renderProfile();
  }

  function setAuthMode(mode) {
    state.authMode = mode;
    els.authTabs.forEach((tab) => {
      const active = tab.dataset.auth === mode;
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-selected", active ? "true" : "false");
    });
    const signup = mode === "signup";
    els.authTitle.textContent = signup ? "Create a free account" : "Sign in";
    els.authNameRow.hidden = !signup;
    els.authPassword.autocomplete = signup ? "new-password" : "current-password";
    els.authHint.hidden = !signup;
    els.authSubmit.querySelector(".btn-label").textContent = signup ? "CREATE ACCOUNT" : "SIGN IN";
    setFeedback(els.authFeedback, "");
  }

  function setFeedback(element, message, tone = "info") {
    element.textContent = message || "";
    element.dataset.state = tone;
    element.hidden = !message;
  }

  async function submitAuth() {
    if (state.busy) return;
    const signup = state.authMode === "signup";
    const body = { email: els.authEmail.value.trim(), password: els.authPassword.value };
    if (signup) body.name = els.authName.value.trim() || state.profile.name || "";
    if (!body.email || !body.password) {
      setFeedback(els.authFeedback, "Enter your email and password.", "error");
      return;
    }
    const stop = startBusy(els.authSubmit, els.authCard);
    try {
      const { res, data } = await postJson(signup ? "/api/auth/signup" : "/api/auth/login", body, { timeout: 30000 });
      if (!res.ok || !data || !data.ok) {
        setFeedback(els.authFeedback, (data && data.error && data.error.message) || "That didn't work. Try again.", "error");
        return;
      }
      state.me = data;
      els.authPassword.value = "";
      applyPlan();
      renderAccount();
      setFeedback(els.authFeedback, "");
    } catch {
      setFeedback(els.authFeedback, "Couldn't reach the app.", "error");
    } finally {
      stop();
    }
  }

  async function accountPost(url, body, feedback) {
    try {
      const { res, data } = await postJson(url, body, { timeout: 30000 });
      if (!res.ok || !data || !data.ok) {
        setFeedback(feedback, (data && data.error && data.error.message) || "That didn't work. Try again.", "error");
        return null;
      }
      return data;
    } catch {
      setFeedback(feedback, "Couldn't reach the app.", "error");
      return null;
    }
  }

  async function signOutAccount() {
    const data = await accountPost("/api/auth/logout", {}, els.authFeedback);
    if (data) state.me = data;
    applyPlan();
    renderAccount();
    setAuthMode("login");
  }

  async function changePassword() {
    const data = await accountPost(
      "/api/me/password",
      { current: els.currentPassword.value, new: els.newPassword.value },
      els.passwordFeedback
    );
    if (!data) return;
    els.currentPassword.value = "";
    els.newPassword.value = "";
    setFeedback(els.passwordFeedback, "✓ Password changed. Other devices were signed out.", "ok");
  }

  async function deleteAccount() {
    if (!window.confirm("Delete your account and its usage history? This can't be undone.")) return;
    const data = await accountPost("/api/me/delete", { password: els.deletePassword.value }, els.deleteFeedback);
    if (!data) return;
    els.deletePassword.value = "";
    state.me = data;
    applyPlan();
    renderAccount();
    setAuthMode("signup");
    setFeedback(els.authFeedback, "Your account was deleted.", "ok");
  }

  function saveAccountName() {
    clearTimeout(state.nameTimer);
    state.nameTimer = setTimeout(async () => {
      try {
        const res = await fetch("/api/me", {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: els.displayName.value.trim() }),
        });
        const data = await res.json();
        if (data && data.ok) {
          state.me = data;
          renderProfile();
        }
      } catch {
        /* name stays local until the next successful save */
      }
    }, 600);
  }

  /* ─────────────────────────── busy state ─────────────────────────── */
  function startBusy(button, panel) {
    state.busy = true;
    button.disabled = true;
    button.classList.add("loading");
    panel.classList.add("scanning");
    let n = 0;
    const dots = button.querySelector(".dots");
    const timer = setInterval(() => {
      n = (n + 1) % 4;
      dots.textContent = ".".repeat(n);
    }, 420);
    return () => {
      state.busy = false;
      button.disabled = false;
      button.classList.remove("loading");
      panel.classList.remove("scanning");
      clearInterval(timer);
    };
  }

  /* ─────────────────────────── check results ─────────────────────────── */
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

  function render(result, text) {
    state.analysis = result;
    state.lastCheck = { text, sentences: result.sentences };
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
        : `${flagged} of ${plural(result.total, "sentence")} look AI-written or AI-polished.`;
    els.humanizeFlagged.hidden = flagged === 0;

    els.countAI.textContent = AI;
    els.countRefined.textContent = REFINED;
    els.countHuman.textContent = HUMAN;

    const notes = [];
    if (result.fallbacks && result.fallbacks.length) {
      notes.push("The AI service was busy, so a backup model finished your check.");
    }
    const reused = result.reused_sentences || 0;
    const alreadyChecked = reused ? result.chunks === 0 : result.cached_chunks && result.cached_chunks === result.chunks;
    if (alreadyChecked) {
      notes.push("You'd already checked this text, so the result was instant.");
    } else if (reused) {
      const fresh = result.total - reused;
      notes.push(`Only the ${fresh} new or edited sentence${fresh === 1 ? " was" : "s were"} checked. The other ${reused} came from your earlier check.`);
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

  async function analyze() {
    if (state.busy) return;
    const text = els.text.value.trim();
    hideAlert(els.alert);
    if (needsSetup(els.alert) || overLimit(text, els.alert)) return;
    if (text.length < MIN_CHARS) {
      showAlert(els.alert, "A little more text, please", "Paste at least a couple of full sentences so there's enough to go on.", { tone: "info" });
      els.text.focus();
      return;
    }

    const stop = startBusy(els.analyzeBtn, els.editorPanel);
    const note = setTimeout(() => (els.progressNote.hidden = false), 4000);
    try {
      const { res, data } = await postJson("/api/analyze", { text, ...routingOptions() });
      if (!res.ok || !data || !data.ok) {
        const e = explainError(res.status, (data && data.error) || {});
        showAlert(els.alert, e.title, e.message, { action: e.action, detail: e.detail, tone: e.tone });
        return;
      }
      render(data.result, text);
      loadMe();
    } catch (err) {
      if (err.name === "AbortError") showAlert(els.alert, "This is taking too long", "Try again, or check a shorter piece of text.");
      else showAlert(els.alert, "Can't reach the app", "Glitch//Detect seems to have stopped running on this computer. Restart it and try again.");
    } finally {
      clearTimeout(note);
      els.progressNote.hidden = true;
      stop();
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

  /* ─────────────────────────── humanize ─────────────────────────── */
  function updateHumanizeStats() {
    showWordCount(countWords(els.humanizeInput.value), els.humanizeStats, els.humanizeLimitNote, "rewrite");
  }

  function useLastCheck() {
    if (!state.lastCheck) return;
    els.humanizeInput.value = state.lastCheck.text;
    updateHumanizeStats();
  }

  function renderRewrite(result) {
    state.rewrite = result;
    els.humanizeOutput.innerHTML = result.html; // escaped by the server

    const hasScores = result.score_before !== null && result.score_after !== null;
    els.scoreShift.hidden = !hasScores || result.rewritten === 0;
    if (hasScores) {
      els.shiftFrom.textContent = `${result.score_before}%`;
      els.shiftTo.textContent = `${result.score_after}%`;
    }

    let note;
    if (result.targets === 0) {
      note = "Nothing needed rewriting: none of the sentences look AI-written.";
    } else {
      note = `Rewrote ${plural(result.rewritten, "flagged sentence")} in ${plural(result.passages, "passage")}.`;
      if (result.reused) note += " Some came from an earlier rewrite, so they were instant.";
      if (result.rewritten < result.targets) note += " The rest were left as they were.";
    }
    if (result.fallbacks && result.fallbacks.length) note += " The AI service was busy, so a backup model helped.";
    const facts = (result.changes || []).filter((c) => c.facts).length;
    if (facts) note += ` Used your details in ${plural(facts, "passage")}.`;
    (result.notes || []).forEach((extra) => (note += ` ${extra}`));
    els.rewriteNote.textContent = note;

    const st = result.style || {};
    els.styleShift.hidden = !(st.before && st.after) || result.rewritten === 0;
    if (!els.styleShift.hidden) {
      const fmt = (v) => (v === null || v === undefined ? "—" : Number(v).toFixed(2));
      els.rhythmBefore.textContent = fmt(st.before.rhythm);
      els.rhythmAfter.textContent = fmt(st.after.rhythm);
      els.rhythmAfter.dataset.good = String((st.after.rhythm || 0) >= 0.5);
      els.phrasesBefore.textContent = st.before.ai_phrases;
      els.phrasesAfter.textContent = st.after.ai_phrases;
      els.phrasesAfter.dataset.good = String(st.after.ai_phrases === 0);
    }

    const tips = result.tips || [];
    els.tipList.replaceChildren(
      ...tips.map((tip) => {
        const li = document.createElement("li");
        li.textContent = `Paragraph ${tip.paragraph}: “${tip.opening}…”`;
        return li;
      })
    );
    els.groundingTips.hidden = tips.length === 0;

    els.humanizeResult.hidden = false;
    els.humanizeResult.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /* Optional details from the writer (the only new facts a rewrite may use). */
  async function askQuestions() {
    const text = els.humanizeInput.value.trim();
    hideAlert(els.hAlert);
    if (text.length < MIN_CHARS) {
      showAlert(els.hAlert, "A little more text, please", "Paste your draft first, then we'll suggest questions.", { tone: "info" });
      return;
    }
    const labels = state.lastCheck && state.lastCheck.text === text ? state.lastCheck.sentences : undefined;
    try {
      const { res, data } = await postJson("/api/humanize/questions", { text, labels });
      if (!res.ok || !data || !data.ok) {
        const e = explainError(res.status, (data && data.error) || {}, "rewrite");
        showAlert(els.hAlert, e.title, e.message, { tone: e.tone });
        return;
      }
      renderQuestions(data.paragraphs || []);
    } catch {
      showAlert(els.hAlert, "Can't reach the app", "Glitch//Detect seems to have stopped running on this computer. Restart it and try again.");
    }
  }

  function renderQuestions(paragraphs) {
    const kept = collectFacts();
    els.questionsEmpty.hidden = paragraphs.length > 0;
    els.questionList.replaceChildren(
      ...paragraphs.map((para, index) => {
        const block = document.createElement("div");
        block.className = "question-block";
        const opening = document.createElement("p");
        opening.className = "question-opening";
        opening.textContent = `“${para.opening}…”`;
        const list = document.createElement("ul");
        para.questions.forEach((q) => {
          const li = document.createElement("li");
          li.textContent = q;
          list.appendChild(li);
        });
        const answer = document.createElement("textarea");
        answer.className = "question-answer";
        answer.rows = 2;
        answer.maxLength = 600;
        answer.dataset.key = para.key;
        answer.id = `fact-${index}`;
        answer.placeholder = "Your answer (leave blank to skip)";
        answer.setAttribute("aria-label", `Details for the paragraph starting ${para.opening}`);
        answer.value = kept[para.key] || "";
        block.append(opening, list, answer);
        return block;
      })
    );
  }

  function collectFacts() {
    const facts = {};
    els.questionList.querySelectorAll("textarea[data-key]").forEach((box) => {
      if (box.value.trim()) facts[box.dataset.key] = box.value.trim();
    });
    return facts;
  }

  /** A plain-text record of authorship: original, rewritten, and author-supplied details. */
  function downloadTrail() {
    const r = state.rewrite;
    if (!r) return;
    const models = (r.models_used || []).map((m) => m.model).join(", ") || "unknown";
    const lines = [
      "GLITCH//DETECT AUTHORSHIP RECORD",
      new Date().toLocaleString(),
      `Tone: ${r.tone}   Voice: ${r.voice}   Model(s): ${models}`,
      "",
      `${plural(r.passages, "passage")} rewritten by an AI model. Everything else is the writer's own text as submitted.`,
      "",
    ];
    (r.changes || []).forEach((change, i) => {
      lines.push(`── Passage ${i + 1} ──`, "WRITTEN BY THE AUTHOR (before):", change.before, "");
      if (change.facts) lines.push("DETAILS SUPPLIED BY THE AUTHOR:", change.facts, "");
      lines.push("REWRITTEN BY AI (after):", change.after, "");
    });
    lines.push("── Final text ──", r.text);
    const url = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/plain" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `glitch-authorship-${new Date().toISOString().slice(0, 10)}.txt`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  async function runHumanize({ fresh = false } = {}) {
    if (state.busy) return;
    const text = els.humanizeInput.value.trim();
    hideAlert(els.hAlert);
    if (needsSetup(els.hAlert) || overLimit(text, els.hAlert, "rewrite")) return;
    if (!signedIn()) {
      showAlert(els.hAlert, "Rewriting needs a free account", "Create a free account or sign in. Free accounts get 5 rewrites a day.", {
        tone: "info",
        action: { href: "#account", text: "Create an account →" },
      });
      return;
    }
    if (text.length < MIN_CHARS) {
      showAlert(els.hAlert, "A little more text, please", "Paste at least a couple of full sentences to rewrite.", { tone: "info" });
      els.humanizeInput.focus();
      return;
    }

    // Labels from the last check save the detector call when the text matches.
    const labels = state.lastCheck && state.lastCheck.text === text ? state.lastCheck.sentences : undefined;
    const stop = startBusy(els.humanizeBtn, els.humanizePanel);
    try {
      const { res, data } = await postJson("/api/humanize", {
        text,
        tone: els.humanizeTone.value,
        voice: els.humanizeVoice.value,
        fresh,
        only_flagged: els.onlyFlagged.checked,
        second_pass: els.secondPass.checked,
        labels,
        facts: collectFacts(),
        ...routingOptions(),
      });
      if (!res.ok || !data || !data.ok) {
        const e = explainError(res.status, (data && data.error) || {}, "rewrite");
        showAlert(els.hAlert, e.title, e.message, { action: e.action, detail: e.detail, tone: e.tone });
        return;
      }
      renderRewrite(data.result);
      loadMe();
    } catch (err) {
      if (err.name === "AbortError") showAlert(els.hAlert, "This is taking too long", "Try again, or rewrite a shorter piece of text.");
      else showAlert(els.hAlert, "Can't reach the app", "Glitch//Detect seems to have stopped running on this computer. Restart it and try again.");
    } finally {
      stop();
    }
  }

  async function copyRewrite() {
    if (!state.rewrite) return;
    try {
      await navigator.clipboard.writeText(state.rewrite.text);
      els.copyRewrite.textContent = "Copied ✓";
      setTimeout(() => (els.copyRewrite.textContent = "Copy"), 1500);
    } catch {
      showAlert(els.hAlert, "Couldn't copy", "Your browser blocked the clipboard. Select the text and copy it manually.", { tone: "info" });
    }
  }

  /* ─────────────────────────── wire up ─────────────────────────── */
  let statsFrame = 0;
  function updateStats() {
    cancelAnimationFrame(statsFrame);
    statsFrame = requestAnimationFrame(() => showWordCount(countWords(els.text.value), els.textStats, els.limitNote));
  }

  window.addEventListener("hashchange", () => showView(location.hash.slice(1)));
  document.querySelectorAll(".alert-close").forEach((btn) =>
    btn.addEventListener("click", () => hideAlert(btn.closest(".alert")))
  );

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
    hideAlert(els.alert);
    state.analysis = null;
    els.text.focus();
  });
  els.showTags.addEventListener("change", () => els.doc.classList.toggle("show-tags", els.showTags.checked));
  els.downloadBtn.addEventListener("click", downloadReport);
  els.humanizeFlagged.addEventListener("click", () => {
    useLastCheck();
    els.onlyFlagged.checked = true;
    location.hash = "#humanize";
    runHumanize();
  });

  els.humanizeInput.addEventListener("input", updateHumanizeStats);
  els.useLastCheck.addEventListener("click", useLastCheck);
  els.humanizeBtn.addEventListener("click", () => runHumanize());
  els.tryAgain.addEventListener("click", () => runHumanize({ fresh: true }));
  els.humanizeTone.addEventListener("change", () => {
    state.settings.tone = els.humanizeTone.value;
    saveSettings();
  });
  els.humanizeVoice.addEventListener("change", () => {
    state.settings.voice = els.humanizeVoice.value;
    saveSettings();
  });
  els.copyRewrite.addEventListener("click", copyRewrite);
  els.downloadTrail.addEventListener("click", downloadTrail);
  els.askQuestions.addEventListener("click", askQuestions);
  els.checkRewrite.addEventListener("click", () => {
    if (!state.rewrite) return;
    els.text.value = state.rewrite.text;
    updateStats();
    location.hash = "#check";
    analyze();
  });

  els.model.addEventListener("change", () => {
    state.settings.models[state.settings.provider] = els.model.value;
    saveSettings();
  });
  els.refreshModels.addEventListener("click", () => loadModels({ force: true }));
  [els.optFallback, els.optCrossProvider, els.optCompact, els.optBalance, els.rememberKeys, els.concurrency, els.chunkWords].forEach(
    (el) => el.addEventListener("change", readOptions)
  );
  els.addKey.addEventListener("click", addKey);
  els.newKey.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      addKey();
    }
  });
  els.toggleNewKey.addEventListener("click", () => {
    const showing = els.newKey.type === "text";
    els.newKey.type = showing ? "password" : "text";
    els.toggleNewKey.textContent = showing ? "Show" : "Hide";
  });

  els.displayName.addEventListener("input", () => {
    if (signedIn()) {
      saveAccountName();
      return;
    }
    state.profile.name = els.displayName.value.trim();
    saveProfile();
    renderProfile();
  });
  els.authTabs.forEach((tab) => tab.addEventListener("click", () => setAuthMode(tab.dataset.auth)));
  els.authSubmit.addEventListener("click", submitAuth);
  [els.authEmail, els.authPassword, els.authName].forEach((input) =>
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        submitAuth();
      }
    })
  );
  els.signOutBtn.addEventListener("click", signOutAccount);
  els.changePasswordBtn.addEventListener("click", changePassword);
  els.deleteAccountBtn.addEventListener("click", deleteAccount);
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
  /** Limits, defaults and tone list from the API (the frontend is static). */
  async function loadConfig() {
    try {
      const res = await fetch("/api/config", { cache: "no-store" });
      const cfg = await res.json();
      MIN_CHARS = cfg.min_chars || MIN_CHARS;
      MAX_WORDS = cfg.max_words || MAX_WORDS;
      PREMIUM_REWRITE_WORDS = cfg.max_rewrite_words || PREMIUM_REWRITE_WORDS;
      DEFAULT_MODEL = cfg.default_model || "";
      TONE_LABELS = cfg.tone_labels || {};
      if (!store.get("localStorage", "glitch_settings") && cfg.default_provider) {
        state.settings.provider = cfg.default_provider;
      }
      if (cfg.max_chars) els.text.maxLength = cfg.max_chars;
      els.text.placeholder = `Paste an essay, article, email… anything with at least a few sentences (up to ${MAX_WORDS.toLocaleString()} words).`;
      if (Array.isArray(cfg.tones) && cfg.tones.length) {
        els.humanizeTone.replaceChildren(
          ...cfg.tones.map((tone) => {
            const option = document.createElement("option");
            option.value = tone;
            option.textContent = TONE_LABELS[tone] || tone.charAt(0).toUpperCase() + tone.slice(1);
            return option;
          })
        );
      }
    } catch {
      /* keep defaults; loadProviders reports the node as offline */
    }
  }

  (async () => {
    load();
    renderProfile();
    showView(location.hash.slice(1));
    await loadConfig();
    fillSettings();
    await loadMe();
    loadProviders();
  })();
})();
