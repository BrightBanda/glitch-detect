/* GLITCH//DETECT — admin dashboard controller */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const TOKEN_KEY = "glitch_admin_token";
  const VIEWS = ["keys", "users", "analytics", "providers"];
  const USERS_PAGE = 50;
  const PLAN_LABELS = { free: "Free", premium: "Premium", family: "Family", guest: "Guest" };
  const STATUS = {
    free: { label: "Free", tone: "count" },
    pending_payment: { label: "Pending payment", tone: "warn" },
    active: { label: "Active", tone: "ok" },
    grace: { label: "Grace period", tone: "warn" },
    expired: { label: "Expired", tone: "count" },
    cancelled: { label: "Cancelled", tone: "count" },
  };
  const REQUEST_TIMEOUT_MS = 30000;
  const LOAD_TIMEOUT_MS = 15000;
  const ADMIN_DISABLED =
    "Admin access isn't enabled on this server. Set ADMIN_TOKEN in .env and restart.";

  const BILLING = {
    free: { label: "Free", note: "Free models only." },
    "free-tier": { label: "Free tier", note: "Uses the key's free allowance." },
    paid: { label: "Paid", note: "Billed to the key's account." },
  };

  const els = {
    gate: $("gate"),
    gateAlert: $("gateAlert"),
    adminToken: $("adminToken"),
    toggleToken: $("toggleToken"),
    signInBtn: $("signInBtn"),
    app: $("adminApp"),
    views: Array.from(document.querySelectorAll(".view")),
    navLinks: Array.from(document.querySelectorAll(".sb-link[data-view]")),
    adminStatusLabel: $("adminStatusLabel"),
    signOutBtn: $("signOutBtn"),
    // keys
    newAdminKey: $("newAdminKey"),
    toggleNewKey: $("toggleNewKey"),
    newKeyLabel: $("newKeyLabel"),
    addKeyBtn: $("addKeyBtn"),
    addProgress: $("addProgress"),
    addFeedback: $("addFeedback"),
    addAlert: $("addAlert"),
    keySummary: $("keySummary"),
    refreshKeysBtn: $("refreshKeysBtn"),
    keysAlert: $("keysAlert"),
    keysFeedback: $("keysFeedback"),
    keyGroups: $("keyGroups"),
    keysEmpty: $("keysEmpty"),
    keysUsageHint: $("keysUsageHint"),
    // providers
    providerSummary: $("providerSummary"),
    refreshProvidersBtn: $("refreshProvidersBtn"),
    providersAlert: $("providersAlert"),
    providerList: $("providerList"),
    providersEmpty: $("providersEmpty"),
    // analytics
    analyticsRefreshBtn: $("analyticsRefreshBtn"),
    analyticsAlert: $("analyticsAlert"),
    analyticsUpdated: $("analyticsUpdated"),
    analyticsStats: $("analyticsStats"),
    analyticsEmpty: $("analyticsEmpty"),
    statApi: $("statApi"),
    statChecks: $("statChecks"),
    statRewrites: $("statRewrites"),
    statTokens: $("statTokens"),
    activeDay: $("activeDay"),
    activeWeek: $("activeWeek"),
    activeMonth: $("activeMonth"),
    activeYear: $("activeYear"),
    activeNote: $("activeNote"),
    usageChart: $("usageChart"),
    chartStart: $("chartStart"),
    chartMax: $("chartMax"),
    chartEnd: $("chartEnd"),
    chartEmpty: $("chartEmpty"),
    endpointRows: $("endpointRows"),
    providerRows: $("providerRows"),
    acctTotal: $("acctTotal"),
    acctPaid: $("acctPaid"),
    acctFree: $("acctFree"),
    acctNew: $("acctNew"),
    acctNote: $("acctNote"),
    paymentsBadge: $("paymentsBadge"),
    // users
    usersSummary: $("usersSummary"),
    refreshUsersBtn: $("refreshUsersBtn"),
    userSearch: $("userSearch"),
    userPlanFilter: $("userPlanFilter"),
    userStatusFilter: $("userStatusFilter"),
    usersAlert: $("usersAlert"),
    usersFeedback: $("usersFeedback"),
    userRows: $("userRows"),
    userPager: $("userPager"),
    usersPrev: $("usersPrev"),
    usersNext: $("usersNext"),
    usersRange: $("usersRange"),
    userDetail: $("userDetail"),
    detailTitle: $("detailTitle"),
    detailSub: $("detailSub"),
    detailFacts: $("detailFacts"),
    detailAlert: $("detailAlert"),
    detailFeedback: $("detailFeedback"),
    closeDetail: $("closeDetail"),
    grantPlan: $("grantPlan"),
    grantDays: $("grantDays"),
    grantBtn: $("grantBtn"),
    editPlan: $("editPlan"),
    editStatus: $("editStatus"),
    editOpenEnded: $("editOpenEnded"),
    editNotes: $("editNotes"),
    saveUserBtn: $("saveUserBtn"),
    resetUsageBtn: $("resetUsageBtn"),
    signOutUserBtn: $("signOutUserBtn"),
    suspendBtn: $("suspendBtn"),
    deleteUserBtn: $("deleteUserBtn"),
    newUserEmail: $("newUserEmail"),
    newUserName: $("newUserName"),
    newUserPassword: $("newUserPassword"),
    createUserBtn: $("createUserBtn"),
    createUserFeedback: $("createUserFeedback"),
  };

  const state = {
    token: "",
    keys: [],
    providers: [],
    providersById: {},
    analytics: null,
    analyticsLoading: false,
    analyticsLoaded: false,
    users: [],
    usersTotal: 0,
    usersOffset: 0,
    usersLoading: false,
    usersLoaded: false,
    selected: null, // the account open in the detail panel
    searchTimer: null,
    busy: false,
    refreshing: false,
    session: 0,
  };

  /* ─────────────────────────── storage ─────────────────────────── */
  const store = {
    get() {
      try {
        return window.sessionStorage.getItem(TOKEN_KEY);
      } catch {
        return null;
      }
    },
    set(value) {
      try {
        window.sessionStorage.setItem(TOKEN_KEY, value);
      } catch {
        /* storage disabled */
      }
    },
    remove() {
      try {
        window.sessionStorage.removeItem(TOKEN_KEY);
      } catch {
        /* storage disabled */
      }
    },
  };

  /* ─────────────────────────── helpers ─────────────────────────── */
  function make(name, className = "", text) {
    const element = document.createElement(name);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function badge(tone, text) {
    const element = make("span", "badge", text);
    element.dataset.tone = tone;
    return element;
  }

  const plural = (count, word) => `${count.toLocaleString()} ${word}${count === 1 ? "" : "s"}`;
  const numberFormat = new Intl.NumberFormat();
  const formatCount = (value) => numberFormat.format(Number(value) || 0);
  const formatDay = (iso) => {
    const parsed = new Date(`${iso}T12:00:00`);
    return Number.isNaN(parsed.getTime())
      ? String(iso || "—")
      : parsed.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  };
  const formatTime = (unix) => {
    const parsed = new Date(Number(unix) * 1000);
    return Number.isNaN(parsed.getTime())
      ? "just now"
      : parsed.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  };
  const formatAvg = (ms) => (Number(ms) > 0 ? `${formatCount(ms)} ms` : "—");

  function formatAdded(unix) {
    const seconds = Number(unix);
    if (!Number.isFinite(seconds) || seconds <= 0) return "";
    return new Date(seconds * 1000).toLocaleDateString(undefined, {
      day: "numeric",
      month: "short",
      year: "numeric",
    });
  }

  function errorOf(data) {
    if (data && typeof data === "object" && data.error && typeof data.error === "object") {
      return data.error;
    }
    return {};
  }

  function adminDisabled(res, data) {
    if (res.status !== 404) return false;
    const message = (errorOf(data).message || "").trim();
    return message === "" || message === "Not found.";
  }

  function adminAuthLost(res, data) {
    if (res.status !== 401) return false;
    return (errorOf(data).message || "").toLowerCase().includes("admin token");
  }

  function safeHref(value) {
    const raw = String(value || "").trim();
    if (!raw) return "";
    try {
      const url = new URL(raw, window.location.origin);
      return url.protocol === "http:" || url.protocol === "https:" ? url.href : "";
    } catch {
      return "";
    }
  }

  function maskedTail(masked) {
    const text = String(masked || "");
    const index = text.lastIndexOf("…");
    return index >= 0 ? text.slice(index + 1) : "";
  }

  function providerLabelOf(entry) {
    const provider = state.providersById[entry.provider];
    return (provider && provider.label) || entry.provider_label || "service";
  }

  const billingFor = (id) => BILLING[id] || { label: "Unknown", note: "" };
  const currentView = () => {
    const name = location.hash.slice(1);
    return VIEWS.includes(name) ? name : "keys";
  };

  /* ─────────────────────────── alerts ─────────────────────────── */
  function showAlert(box, title, message, { detail = "", tone = "error" } = {}) {
    box.dataset.tone = tone;
    box.querySelector(".alert-title").textContent = title;
    box.querySelector(".alert-message").textContent = message;
    box.querySelector(".alert-detail").textContent = detail;
    box.querySelector(".alert-detail-wrap").hidden = !detail;
    box.hidden = false;
  }

  const hideAlert = (box) => {
    if (box) box.hidden = true;
  };

  function setFeedback(element, message, tone = "info") {
    element.textContent = message || "";
    element.dataset.state = tone;
    element.hidden = !message;
  }

  /* ─────────────────────────── requests ─────────────────────────── */
  async function adminFetch(url, { method = "GET", body, timeout = REQUEST_TIMEOUT_MS } = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    const headers = { Authorization: `Bearer ${state.token}` };
    if (body !== undefined) headers["Content-Type"] = "application/json";
    try {
      const res = await fetch(url, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        cache: "no-store",
        signal: controller.signal,
      });
      const data = await res.json().catch(() => null);
      return { res, data };
    } finally {
      clearTimeout(timer);
    }
  }

  function setGateBusy(busy) {
    els.signInBtn.disabled = busy;
    els.signInBtn.classList.toggle("loading", busy);
    els.adminToken.disabled = busy;
    els.toggleToken.disabled = busy;
  }

  function setRefreshBusy(busy) {
    [els.refreshKeysBtn, els.refreshProvidersBtn].forEach((button) => {
      button.disabled = busy;
      button.textContent = busy ? "Refreshing…" : "Refresh";
    });
    els.addKeyBtn.disabled = busy;
  }

  function setAnalyticsBusy(busy) {
    els.analyticsRefreshBtn.disabled = busy;
    els.analyticsRefreshBtn.textContent = busy ? "Refreshing…" : "Refresh";
  }

  function resetTokenInput() {
    els.adminToken.value = "";
    els.adminToken.type = "password";
    els.toggleToken.textContent = "Show";
    els.toggleToken.setAttribute("aria-label", "Show token");
  }

  function resetKeyInput() {
    els.newAdminKey.value = "";
    els.newKeyLabel.value = "";
    els.newAdminKey.type = "password";
    els.toggleNewKey.textContent = "Show";
    els.toggleNewKey.setAttribute("aria-label", "Show key");
  }

  /* ─────────────────────────── rendering ─────────────────────────── */
  function recomputeReady() {
    const ready = new Set(state.keys.map((entry) => entry.provider));
    state.providers.forEach((provider) => {
      provider.ready = ready.has(provider.id);
    });
  }

  function updateStatus() {
    const total = state.providers.length;
    const ready = state.providers.filter((provider) => provider.ready).length;
    if (!total) {
      els.adminStatusLabel.textContent = "Admin console";
      return;
    }
    if (!ready) {
      els.adminStatusLabel.textContent = "No keys yet";
      return;
    }
    const noun = total === 1 ? "service" : "services";
    els.adminStatusLabel.textContent = `${ready} of ${total} ${noun} ready`;
  }

  function activeKeyIds() {
    const active = {};
    for (const entry of state.keys) {
      if (!active[entry.provider]) active[entry.provider] = entry.id;
    }
    return active;
  }

  function renderKey(entry, inUse) {
    const item = make("li", "admin-key");
    const main = make("div", "key-main");
    const line = make("div", "key-line");
    line.append(make("span", "key-mask", entry.key || "…"));

    if (inUse) line.append(badge("in-use", "In use"));
    if (entry.source === "env") {
      line.append(badge("env", "Read-only"));
    } else {
      line.append(entry.verified ? badge("ok", "Working") : badge("warn", "Not yet verified"));
    }

    const meta = make("div", "key-meta");
    if (entry.source === "env") {
      meta.append(make("span", "key-label", "Set in the server's .env file"));
    } else if (entry.label) {
      meta.append(make("span", "key-label", entry.label));
    }
    meta.append(
      badge(
        entry.source === "admin" ? "count" : "env",
        entry.source === "admin" ? "Added here" : "Server environment"
      )
    );
    const added = formatAdded(entry.added_at);
    if (added) meta.append(make("span", "key-date", `Added ${added}`));

    main.append(line, meta);
    item.append(main);

    if (entry.source === "admin") {
      const remove = make("button", "mini-btn danger", "Remove");
      remove.type = "button";
      const tail = maskedTail(entry.key);
      const label = providerLabelOf(entry);
      remove.setAttribute(
        "aria-label",
        tail ? `Remove the ${label} key ending ${tail}` : `Remove the ${label} key`
      );
      remove.addEventListener("click", () => removeKey(entry, remove));
      item.append(remove);
    }
    return item;
  }

  function renderKeyGroup(provider, label, entries, active) {
    const group = make("div", "key-group");
    const title = make("h3", "key-group-title");
    title.append(make("span", "", label), badge("count", plural(entries.length, "key")));
    const list = make("ul", "key-list");
    entries.forEach((entry) => list.append(renderKey(entry, active[provider] === entry.id)));
    group.append(title, list);
    return group;
  }

  function renderKeyGroups() {
    const active = activeKeyIds();
    const groups = new Map();
    for (const entry of state.keys) {
      if (!groups.has(entry.provider)) groups.set(entry.provider, []);
      groups.get(entry.provider).push(entry);
    }

    const nodes = [];
    for (const provider of state.providers) {
      const entries = groups.get(provider.id);
      if (!entries || !entries.length) continue;
      const label = provider.label || (entries[0] && entries[0].provider_label) || "Unknown service";
      nodes.push(renderKeyGroup(provider.id, label, entries, active));
      groups.delete(provider.id);
    }
    for (const [provider, entries] of groups) {
      const label = (entries[0] && entries[0].provider_label) || "Unknown service";
      nodes.push(renderKeyGroup(provider, label, entries, active));
    }

    els.keyGroups.replaceChildren(...nodes);
    const serviceCount = new Set(state.keys.map((entry) => entry.provider)).size;
    els.keySummary.textContent = state.keys.length
      ? `${plural(state.keys.length, "key")} · ${plural(serviceCount, "service")}`
      : "No keys yet";
    els.keysEmpty.hidden = state.keys.length > 0;
    els.keysUsageHint.hidden = state.keys.length === 0;
  }

  function renderProvider(provider) {
    const row = make("li", "provider-row");
    const main = make("div", "provider-main");
    const title = make("div", "provider-title");
    const billing = billingFor(provider.billing);
    title.append(
      make("span", "provider-name", provider.label || "Unknown service"),
      badge(BILLING[provider.billing] ? provider.billing : "count", billing.label)
    );
    main.append(title, make("p", "provider-note", billing.note || "Billing details unavailable."));

    const actions = make("div", "provider-actions");
    const status = make("span", "choice-status", provider.ready ? "✓ Ready" : "Needs a key");
    status.dataset.state = provider.ready ? "ready" : "needs-key";
    actions.append(status);

    const href = safeHref(provider.keys_url);
    if (href) {
      const link = make("a", "mini-btn", "Get a key ↗");
      link.href = href;
      link.target = "_blank";
      link.setAttribute("rel", "noopener noreferrer");
      link.setAttribute(
        "aria-label",
        `Get a ${provider.label || "provider"} key (opens in a new tab)`
      );
      actions.append(link);
    }

    row.append(main, actions);
    return row;
  }

  function renderProviders() {
    const total = state.providers.length;
    const ready = state.providers.filter((provider) => provider.ready).length;
    els.providerSummary.textContent = total ? `${ready} of ${total} ready` : "No services";
    els.providersEmpty.hidden = total > 0;
    els.providerList.replaceChildren(...state.providers.map(renderProvider));
  }

  function renderAll() {
    renderKeyGroups();
    renderProviders();
    updateStatus();
  }

  /* ─────────────────────────── analytics ─────────────────────────── */
  function emptyRow(columns, text) {
    const row = make("tr");
    const cell = make("td", "table-empty", text);
    cell.colSpan = columns;
    row.append(cell);
    return row;
  }

  function renderUsageChart(series) {
    const days = Array.isArray(series) ? series : [];
    const max = Math.max(1, ...days.map((day) => Math.max(Number(day.requests) || 0, Number(day.active) || 0)));
    const columns = [];

    for (const day of days) {
      const column = make("span", "chart-col");
      column.title = `${formatDay(day.date)}: ${formatCount(day.requests)} API calls, ${formatCount(day.active)} active browsers`;
      column.setAttribute("aria-hidden", "true");

      const calls = make("span", "chart-bar calls");
      const callHeight = Number(day.requests) > 0
        ? Math.max(2, Math.round(((Number(day.requests) || 0) / max) * 100))
        : 0;
      calls.style.height = `${callHeight}%`;
      const active = make("span", "chart-bar active");
      const activeHeight = Number(day.active) > 0
        ? Math.max(2, Math.round(((Number(day.active) || 0) / max) * 100))
        : 0;
      active.style.height = `${activeHeight}%`;

      column.append(calls, active);
      columns.push(column);
    }

    els.usageChart.replaceChildren(...columns);
    els.chartStart.textContent = days.length ? formatDay(days[0].date) : "—";
    els.chartEnd.textContent = days.length ? formatDay(days[days.length - 1].date) : "—";
    els.chartMax.textContent = `Busiest ${formatCount(max)}`;
    els.usageChart.setAttribute(
      "aria-label",
      days.length
        ? `Daily API calls and active browsers for ${days.length} days. Busiest day: ${formatCount(max)}.`
        : "No daily usage data yet."
    );
    const total = days.reduce(
      (sum, day) =>
        sum + (Number(day.requests) || 0) + (Number(day.visits) || 0) + (Number(day.active) || 0),
      0
    );
    els.chartEmpty.hidden = total > 0;
  }

  function renderEndpointTable(endpoints) {
    const items = Array.isArray(endpoints) ? [...endpoints] : [];
    const order = ["analyze", "humanize", "models", "identify"];
    const labels = {
      analyze: "Checks",
      humanize: "Rewrites",
      models: "Model lists",
      identify: "Key identification",
    };
    items.sort((a, b) => {
      const ai = order.indexOf(a.id);
      const bi = order.indexOf(b.id);
      return (ai < 0 ? 99 : ai) - (bi < 0 ? 99 : bi);
    });

    if (!items.length) {
      els.endpointRows.replaceChildren(emptyRow(4, "No API calls in this period."));
      return;
    }

    els.endpointRows.replaceChildren(
      ...items.map((item) => {
        const row = make("tr");
        row.append(
          make("td", "", labels[item.id] || item.id),
          make("td", "", formatCount(item.requests)),
          make("td", "", formatCount(item.errors)),
          make("td", "", formatAvg(item.avg_ms))
        );
        return row;
      })
    );
  }

  function renderProviderTable(providers) {
    const items = Array.isArray(providers) ? [...providers] : [];
    items.sort((a, b) => (Number(b.requests) || 0) - (Number(a.requests) || 0));

    if (!items.length) {
      els.providerRows.replaceChildren(emptyRow(4, "No service calls in this period."));
      return;
    }

    els.providerRows.replaceChildren(
      ...items.map((item) => {
        const provider = state.providersById[item.id];
        const label = (provider && provider.label) || item.id || "Unknown service";
        const row = make("tr");
        row.append(
          make("td", "", label),
          make("td", "", formatCount(item.requests)),
          make("td", "", formatCount(item.tokens)),
          make("td", "", formatCount(item.errors))
        );
        return row;
      })
    );
  }

  function renderAnalytics() {
    const data = state.analytics;
    if (!data) return;
    const month = (data.windows && data.windows.month) || {};
    const active = data.active || {};

    els.statApi.textContent = formatCount(month.requests);
    els.statChecks.textContent = formatCount(month.checks);
    els.statRewrites.textContent = formatCount(month.rewrites);
    els.statTokens.textContent = formatCount(month.tokens);

    els.activeDay.textContent = formatCount(active.day);
    els.activeWeek.textContent = formatCount(active.week);
    els.activeMonth.textContent = formatCount(active.month);
    els.activeYear.textContent = formatCount(active.year);
    els.activeNote.textContent = `New this month: ${formatCount(active.new_month)} · Seen all time: ${formatCount(active.total)}`;

    els.analyticsUpdated.textContent = data.generated_at
      ? `Updated ${formatTime(data.generated_at)}`
      : "Updated just now";
    els.analyticsEmpty.hidden = !data.empty;

    renderUsageChart(data.series);
    renderEndpointTable(month.endpoints);
    renderProviderTable(month.providers);
    renderAccounts(data.plans || {});
  }

  function renderAccounts(plans) {
    const connected = plans.connected === true;
    els.acctTotal.textContent = connected ? formatCount(plans.total) : "—";
    els.acctPaid.textContent = connected ? formatCount(plans.paid) : "—";
    els.acctFree.textContent = connected ? formatCount(plans.free) : "—";
    els.acctNew.textContent = connected ? formatCount(plans.new_month) : "—";
    const statuses = Object.entries(plans.by_status || {})
      .filter(([, count]) => count > 0)
      .map(([status, count]) => `${(STATUS[status] || { label: status }).label} ${formatCount(count)}`)
      .join(" · ");
    els.acctNote.textContent = connected
      ? `Active today: ${formatCount(plans.active_today)} · Suspended: ${formatCount(plans.suspended)}${statuses ? ` · ${statuses}` : ""}`
      : "Accounts aren't available on this server.";
    els.paymentsBadge.textContent = plans.payments ? "Payments connected" : "Payments not connected";
    els.paymentsBadge.dataset.tone = plans.payments ? "ok" : "warn";
  }

  function clearAnalytics() {
    state.analytics = null;
    state.analyticsLoaded = false;
    state.analyticsLoading = false;
    els.statApi.textContent = "—";
    els.statChecks.textContent = "—";
    els.statRewrites.textContent = "—";
    els.statTokens.textContent = "—";
    els.activeDay.textContent = "—";
    els.activeWeek.textContent = "—";
    els.activeMonth.textContent = "—";
    els.activeYear.textContent = "—";
    els.activeNote.textContent = "New this month: — · Seen all time: —";
    els.analyticsUpdated.textContent = "Not loaded";
    els.analyticsEmpty.hidden = true;
    els.usageChart.replaceChildren();
    els.usageChart.setAttribute("aria-label", "Daily API usage and active browsers for the last 30 days");
    els.chartStart.textContent = "—";
    els.chartEnd.textContent = "—";
    els.chartMax.textContent = "—";
    els.chartEmpty.hidden = true;
    els.endpointRows.replaceChildren(emptyRow(4, "No data yet."));
    els.providerRows.replaceChildren(emptyRow(4, "No data yet."));
    renderAccounts({});
    hideAlert(els.analyticsAlert);
    setAnalyticsBusy(false);
  }

  /* ─────────────────────────── navigation ─────────────────────────── */
  function showView(name) {
    const view = VIEWS.includes(name) ? name : "keys";
    els.views.forEach((section) => {
      section.hidden = section.dataset.view !== view;
    });
    els.navLinks.forEach((link) => {
      const active = link.dataset.view === view;
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    const titles = { keys: "API keys", users: "Users", analytics: "Analytics", providers: "Providers" };
    document.title = `${titles[view]} — Glitch//Detect Admin`;
    window.scrollTo(0, 0);
    if (view === "analytics" && state.token && !state.analyticsLoaded && !state.analyticsLoading) {
      loadAnalytics();
    }
    if (view === "users" && state.token && !state.usersLoaded && !state.usersLoading) {
      loadUsers();
    }
  }

  function focusCurrentHeading() {
    const section = els.views.find((view) => !view.hidden);
    const heading = section && section.querySelector("h1");
    if (heading) heading.focus();
  }

  function focusAfterRemove() {
    const next = els.keyGroups.querySelector(".admin-key .mini-btn.danger");
    if (next) next.focus();
    else els.newAdminKey.focus();
  }

  /* ─────────────────────────── session ─────────────────────────── */
  function signOut({ title = "", message = "", tone = "info" } = {}) {
    state.session += 1;
    state.token = "";
    store.remove();
    state.keys = [];
    state.providers = [];
    state.providersById = {};
    state.busy = false;
    state.refreshing = false;
    clearAnalytics();
    clearUsers();

    setRefreshBusy(false);
    setGateBusy(false);
    els.addKeyBtn.disabled = false;
    els.addKeyBtn.classList.remove("loading");
    els.addProgress.hidden = true;

    els.keyGroups.replaceChildren();
    els.providerList.replaceChildren();
    els.keySummary.textContent = "—";
    els.providerSummary.textContent = "—";
    els.keysEmpty.hidden = true;
    els.providersEmpty.hidden = true;
    els.adminStatusLabel.textContent = "Admin console";
    setFeedback(els.addFeedback, "");
    setFeedback(els.keysFeedback, "");
    hideAlert(els.addAlert);
    hideAlert(els.keysAlert);
    hideAlert(els.providersAlert);

    resetKeyInput();
    resetTokenInput();
    els.app.hidden = true;
    els.gate.hidden = false;
    document.title = "Admin — Glitch//Detect";
    if (title) showAlert(els.gateAlert, title, message, { tone });
    else hideAlert(els.gateAlert);
    els.adminToken.focus();
  }

  async function attemptSignIn() {
    if (state.busy) return;
    const typed = els.adminToken.value.trim();
    const token = typed || state.token;
    hideAlert(els.gateAlert);
    if (!token) {
      showAlert(els.gateAlert, "Enter the admin token", "Paste the token from the server, then sign in.", {
        tone: "info",
      });
      els.adminToken.focus();
      return;
    }

    state.busy = true;
    const session = state.session + 1;
    state.session = session;
    state.token = token;
    setGateBusy(true);

    try {
      const { res, data } = await adminFetch("/api/admin/keys", { timeout: LOAD_TIMEOUT_MS });
      if (session !== state.session) return;

      if (res.status === 404) {
        state.token = "";
        store.remove();
        showAlert(els.gateAlert, "Admin access is off", ADMIN_DISABLED, { tone: "info" });
        els.adminToken.focus();
        return;
      }
      if (res.status === 401) {
        state.token = "";
        store.remove();
        showAlert(els.gateAlert, "That token didn't work", "Check the token and try again.");
        els.adminToken.focus();
        return;
      }
      if (!res.ok || !data || data.ok !== true) {
        const error = errorOf(data);
        showAlert(
          els.gateAlert,
          "Something went wrong",
          error.message || "The server couldn't load the keys. Try again.",
          { detail: [`HTTP ${res.status}`, error.detail].filter(Boolean).join("\n") }
        );
        els.adminToken.focus();
        return;
      }

      store.set(token);
      state.keys = Array.isArray(data.keys) ? data.keys : [];
      state.providers = Array.isArray(data.providers) ? data.providers : [];
      state.providersById = Object.fromEntries(
        state.providers.map((provider) => [provider.id, provider])
      );
      recomputeReady();
      resetTokenInput();
      hideAlert(els.gateAlert);
      els.gate.hidden = true;
      els.app.hidden = false;
      renderAll();
      showView(location.hash.slice(1));
      focusCurrentHeading();
    } catch (error) {
      if (session !== state.session) return;
      const aborted = error && error.name === "AbortError";
      showAlert(
        els.gateAlert,
        aborted ? "That took too long" : "Can't reach the app",
        aborted
          ? "The server didn't answer in time. Try again."
          : "Glitch//Detect seems to have stopped running on this computer. Restart it and try again.",
        { tone: "info" }
      );
      els.adminToken.focus();
    } finally {
      if (session === state.session) {
        state.busy = false;
        setGateBusy(false);
      }
    }
  }

  /* ─────────────────────────── refresh ─────────────────────────── */
  async function loadAnalytics({ force = false } = {}) {
    if (state.analyticsLoading) return false;
    if (state.analyticsLoaded && !force) return true;
    state.analyticsLoading = true;
    const session = state.session;
    setAnalyticsBusy(true);
    hideAlert(els.analyticsAlert);
    if (!state.analytics) els.analyticsUpdated.textContent = "Loading…";

    try {
      const { res, data } = await adminFetch("/api/admin/analytics", {
        timeout: LOAD_TIMEOUT_MS,
      });
      if (session !== state.session) return false;

      if (adminDisabled(res, data)) {
        signOut({ title: "Admin access is off", message: ADMIN_DISABLED, tone: "info" });
        return false;
      }
      if (adminAuthLost(res, data)) {
        signOut({
          title: "Your session ended",
          message: "Sign in again to keep viewing analytics.",
          tone: "info",
        });
        return false;
      }
      if (!res.ok || !data || data.ok !== true) {
        const error = errorOf(data);
        showAlert(
          els.analyticsAlert,
          "Couldn't load analytics",
          error.message || "The server didn't return usage data. Try refreshing.",
          { detail: [`HTTP ${res.status}`, error.detail].filter(Boolean).join("\n") }
        );
        return false;
      }

      state.analytics = data;
      state.analyticsLoaded = true;
      renderAnalytics();
      return true;
    } catch (error) {
      if (session !== state.session) return false;
      const aborted = error && error.name === "AbortError";
      showAlert(
        els.analyticsAlert,
        aborted ? "That took too long" : "Can't reach the app",
        aborted
          ? "The server didn't answer in time. Try again."
          : "Glitch//Detect seems to have stopped running on this computer. Restart it and try again.",
        { tone: "info" }
      );
      return false;
    } finally {
      if (session === state.session) {
        state.analyticsLoading = false;
        setAnalyticsBusy(false);
      }
    }
  }

  async function loadData({ quiet = false, force = false } = {}) {
    if (state.refreshing || (state.busy && !force)) return false;
    state.refreshing = true;
    const session = state.session;
    setRefreshBusy(true);
    if (!quiet) {
      hideAlert(els.keysAlert);
      hideAlert(els.providersAlert);
    }

    try {
      const { res, data } = await adminFetch("/api/admin/keys", { timeout: LOAD_TIMEOUT_MS });
      if (session !== state.session) return false;

      if (adminDisabled(res, data)) {
        signOut({ title: "Admin access is off", message: ADMIN_DISABLED, tone: "info" });
        return false;
      }
      if (res.status === 401) {
        signOut({
          title: "Your session ended",
          message: "Sign in again to keep managing keys.",
          tone: "info",
        });
        return false;
      }
      if (!res.ok || !data || data.ok !== true) {
        const error = errorOf(data);
        const box = currentView() === "providers" ? els.providersAlert : els.keysAlert;
        showAlert(box, "Couldn't load the keys", error.message || "The server didn't return the key list. Try refreshing.", {
          detail: [`HTTP ${res.status}`, error.detail].filter(Boolean).join("\n"),
        });
        return false;
      }

      state.keys = Array.isArray(data.keys) ? data.keys : [];
      state.providers = Array.isArray(data.providers) ? data.providers : [];
      state.providersById = Object.fromEntries(
        state.providers.map((provider) => [provider.id, provider])
      );
      recomputeReady();
      renderAll();
      return true;
    } catch (error) {
      if (session !== state.session) return false;
      const aborted = error && error.name === "AbortError";
      const box = currentView() === "providers" ? els.providersAlert : els.keysAlert;
      showAlert(
        box,
        aborted ? "That took too long" : "Can't reach the app",
        aborted
          ? "The server didn't answer in time. Try again."
          : "Glitch//Detect seems to have stopped running on this computer. Restart it and try again.",
        { tone: "info" }
      );
      return false;
    } finally {
      if (session === state.session) {
        state.refreshing = false;
        setRefreshBusy(false);
      }
    }
  }

  /* ─────────────────────────── add key ─────────────────────────── */
  function showAddFailure(status, error) {
    const message = (error && error.message) || "";
    const detail = [`HTTP ${status}`, error && error.detail].filter(Boolean).join("\n");
    if (status === 400) {
      showAlert(
        els.addAlert,
        "That key wasn't recognised",
        message || "We don't recognise this kind of key yet. Check it and try again.",
        { detail }
      );
    } else if (status === 401) {
      showAlert(
        els.addAlert,
        "That key didn't work",
        message || "The service rejected this key. Check it and try again.",
        { detail }
      );
    } else if (status === 503) {
      showAlert(
        els.addAlert,
        "Couldn't confirm the key right now",
        message || "Try again in a moment.",
        { detail, tone: "info" }
      );
    } else if (status === 413) {
      showAlert(
        els.addAlert,
        "That request was too large",
        message || "Try a shorter label and try again.",
        { detail }
      );
    } else {
      showAlert(
        els.addAlert,
        "The key couldn't be added",
        message || "Please try again in a moment.",
        { detail }
      );
    }
    els.newAdminKey.focus();
  }

  function upsertKey(entry) {
    const index = state.keys.findIndex((key) => key.id === entry.id);
    if (index >= 0) state.keys[index] = entry;
    else state.keys.unshift(entry);
  }

  async function addKey() {
    if (state.busy) return;
    if (state.refreshing) {
      setFeedback(els.addFeedback, "The list is refreshing. Try again in a moment.", "info");
      return;
    }
    const key = els.newAdminKey.value.trim();
    const label = els.newKeyLabel.value.trim();
    hideAlert(els.addAlert);
    setFeedback(els.addFeedback, "");
    setFeedback(els.keysFeedback, "");

    if (!key) {
      showAlert(els.addAlert, "Paste a key first", "Paste the API key you want to add, then press Add key.", {
        tone: "info",
      });
      els.newAdminKey.focus();
      return;
    }

    state.busy = true;
    const session = state.session;
    els.addKeyBtn.disabled = true;
    els.addKeyBtn.classList.add("loading");
    els.addProgress.hidden = false;

    try {
      const { res, data } = await adminFetch("/api/admin/keys", {
        method: "POST",
        body: { key, label },
        timeout: REQUEST_TIMEOUT_MS,
      });
      if (session !== state.session) return;

      if (adminDisabled(res, data)) {
        signOut({ title: "Admin access is off", message: ADMIN_DISABLED, tone: "info" });
        return;
      }
      if (adminAuthLost(res, data)) {
        signOut({
          title: "Your session ended",
          message: "Sign in again to keep managing keys.",
          tone: "info",
        });
        return;
      }
      if (!res.ok || !data || data.ok !== true) {
        showAddFailure(res.status, errorOf(data));
        return;
      }

      const entry = data.key;
      if (!entry || !entry.id) {
        const refreshed = await loadData({ quiet: true, force: true });
        if (session !== state.session) return;
        setFeedback(
          els.addFeedback,
          refreshed ? "✓ Key added." : "✓ Key added. Refresh the list to see it.",
          "ok"
        );
        return;
      }

      const already = state.keys.some((item) => item.id === entry.id);
      upsertKey(entry);
      recomputeReady();
      renderAll();

      const provider = entry.provider_label || providerLabelOf(entry);
      if (already) {
        setFeedback(els.addFeedback, `That ${provider} key is already added.`, "ok");
      } else if (entry.verified) {
        setFeedback(els.addFeedback, `✓ ${provider} key added and working.`, "ok");
      } else {
        setFeedback(
          els.addFeedback,
          `✓ ${provider} key added. We couldn't confirm it yet.`,
          "info"
        );
      }

      resetKeyInput();
      els.newAdminKey.focus();
    } catch (error) {
      if (session !== state.session) return;
      const aborted = error && error.name === "AbortError";
      showAlert(
        els.addAlert,
        aborted ? "That took too long" : "Can't reach the app",
        aborted
          ? "The service didn't answer in time. Try again in a moment."
          : "Glitch//Detect seems to have stopped running on this computer. Restart it and try again.",
        { tone: "info" }
      );
    } finally {
      if (session === state.session) {
        state.busy = false;
        els.addKeyBtn.disabled = false;
        els.addKeyBtn.classList.remove("loading");
        els.addProgress.hidden = true;
      }
    }
  }

  /* ─────────────────────────── remove key ─────────────────────────── */
  async function removeKey(entry, button) {
    if (state.refreshing) {
      setFeedback(els.keysFeedback, "The list is refreshing. Try again in a moment.", "info");
      return;
    }
    const label = providerLabelOf(entry);
    const tail = maskedTail(entry.key);
    const question = tail
      ? `Remove the ${label} key ending ${tail}?`
      : `Remove this ${label} key?`;
    if (!window.confirm(question)) return;

    const session = state.session;
    button.disabled = true;
    button.textContent = "Removing…";
    hideAlert(els.keysAlert);
    setFeedback(els.keysFeedback, "");
    setFeedback(els.addFeedback, "");

    try {
      const { res, data } = await adminFetch(
        `/api/admin/keys/${encodeURIComponent(entry.id)}`,
        { method: "DELETE", timeout: LOAD_TIMEOUT_MS }
      );
      if (session !== state.session) return;

      if (adminDisabled(res, data)) {
        signOut({ title: "Admin access is off", message: ADMIN_DISABLED, tone: "info" });
        return;
      }
      if (adminAuthLost(res, data)) {
        signOut({
          title: "Your session ended",
          message: "Sign in again to keep managing keys.",
          tone: "info",
        });
        return;
      }
      if (!res.ok || !data || data.ok !== true) {
        if (res.status === 404) {
          state.keys = state.keys.filter((item) => item.id !== entry.id);
          recomputeReady();
          renderAll();
          setFeedback(
            els.keysFeedback,
            "That key was already removed. The list is up to date.",
            "info"
          );
          focusAfterRemove();
          return;
        }
        const error = errorOf(data);
        showAlert(
          els.keysAlert,
          "The key couldn't be removed",
          error.message || "Please try again.",
          { detail: [`HTTP ${res.status}`, error.detail].filter(Boolean).join("\n") }
        );
        return;
      }

      state.keys = state.keys.filter((item) => item.id !== entry.id);
      recomputeReady();
      renderAll();
      setFeedback(els.keysFeedback, `Removed the ${label} key.`, "ok");
      focusAfterRemove();
    } catch (error) {
      if (session !== state.session) return;
      const aborted = error && error.name === "AbortError";
      showAlert(
        els.keysAlert,
        aborted ? "That took too long" : "Can't reach the app",
        aborted
          ? "The key wasn't removed. Try again."
          : "The key wasn't removed because the server couldn't be reached.",
        { tone: "info" }
      );
    } finally {
      if (button.isConnected) {
        button.disabled = false;
        button.textContent = "Remove";
      }
    }
  }

  /* ─────────────────────────── users ─────────────────────────── */
  /**
   * One admin request with the shared session handling: signs out when admin
   * access is off or the token stopped working, shows other failures in `box`.
   * Resolves to the response data, or null when the caller should stop.
   */
  async function adminCall(url, { method = "GET", body, box, failTitle, timeout = REQUEST_TIMEOUT_MS } = {}) {
    const session = state.session;
    try {
      const { res, data } = await adminFetch(url, { method, body, timeout });
      if (session !== state.session) return null;
      // "Not found." means admin access is off; "No such user." is a normal 404.
      if (adminDisabled(res, data)) {
        signOut({ title: "Admin access is off", message: ADMIN_DISABLED, tone: "info" });
        return null;
      }
      if (adminAuthLost(res, data)) {
        signOut({ title: "Your session ended", message: "Sign in again to keep managing users.", tone: "info" });
        return null;
      }
      if (!res.ok || !data || data.ok !== true) {
        const error = errorOf(data);
        if (box) {
          showAlert(box, failTitle || "That didn't work", error.message || "Please try again.", {
            detail: [`HTTP ${res.status}`, error.detail].filter(Boolean).join("\n"),
          });
        }
        return res.status === 404 ? { notFound: true } : null;
      }
      return data;
    } catch (error) {
      if (session !== state.session) return null;
      const aborted = error && error.name === "AbortError";
      if (box) {
        showAlert(
          box,
          aborted ? "That took too long" : "Can't reach the app",
          aborted
            ? "The server didn't answer in time. Try again."
            : "Glitch//Detect seems to have stopped running on this computer. Restart it and try again.",
          { tone: "info" }
        );
      }
      return null;
    }
  }

  const planLabel = (id) => PLAN_LABELS[id] || id || "—";
  const statusOf = (id) => STATUS[id] || { label: id || "—", tone: "count" };

  function formatWhen(unix) {
    const seconds = Number(unix);
    if (!Number.isFinite(seconds) || seconds <= 0) return "Never";
    const diff = Date.now() / 1000 - seconds;
    if (diff < 120) return "Just now";
    if (diff < 3600) return `${Math.round(diff / 60)} min ago`;
    if (diff < 86400) return `${Math.round(diff / 3600)} h ago`;
    return formatAdded(seconds);
  }

  function planBadge(user) {
    const plan = user.effective_plan || user.plan;
    return badge(plan === "free" ? "count" : "premium", planLabel(plan));
  }

  function statusBadges(user) {
    const status = statusOf(user.status);
    const nodes = [badge(status.tone, status.label)];
    if (user.suspended) nodes.push(badge("danger", "Suspended"));
    return nodes;
  }

  function usageText(user) {
    const used = user.usage_today || {};
    const limits = user.limits || {};
    return `${formatCount(used.checks)}/${formatCount(limits.checks_per_day)} checks · ${formatCount(used.rewrites)}/${formatCount(limits.rewrites_per_day)} rewrites`;
  }

  function setUsersBusy(busy) {
    els.refreshUsersBtn.disabled = busy;
    els.refreshUsersBtn.textContent = busy ? "Refreshing…" : "Refresh";
  }

  function renderUsers() {
    if (!state.users.length) {
      const filtered = els.userSearch.value.trim() || els.userPlanFilter.value || els.userStatusFilter.value;
      els.userRows.replaceChildren(
        emptyRow(6, filtered ? "No accounts match these filters." : "No accounts yet. People can sign up from the app's Account page.")
      );
    } else {
      els.userRows.replaceChildren(
        ...state.users.map((user) => {
          const row = make("tr");
          row.dataset.selected = String(Boolean(state.selected && state.selected.id === user.id));
          const who = make("td");
          const cell = make("div", "user-cell");
          cell.append(make("span", "user-email", user.email));
          if (user.name) cell.append(make("span", "user-name", user.name));
          who.append(cell);
          const plan = make("td");
          plan.append(planBadge(user));
          const status = make("td");
          status.append(...statusBadges(user));
          const usage = make("td", "usage-cell", usageText(user));
          const seen = make("td", "", formatWhen(user.last_seen));
          const actions = make("td");
          const manage = make("button", "mini-btn", "Manage");
          manage.type = "button";
          manage.setAttribute("aria-label", `Manage ${user.email}`);
          manage.addEventListener("click", () => openUser(user.id));
          actions.append(manage);
          row.append(who, plan, status, usage, seen, actions);
          return row;
        })
      );
    }
    els.usersSummary.textContent = plural(state.usersTotal, "account");
    const start = state.usersTotal ? state.usersOffset + 1 : 0;
    const end = state.usersOffset + state.users.length;
    els.userPager.hidden = state.usersTotal <= USERS_PAGE;
    els.usersRange.textContent = `${formatCount(start)}–${formatCount(end)} of ${formatCount(state.usersTotal)}`;
    els.usersPrev.disabled = state.usersOffset === 0;
    els.usersNext.disabled = end >= state.usersTotal;
  }

  async function loadUsers({ offset = state.usersOffset } = {}) {
    if (state.usersLoading) return;
    state.usersLoading = true;
    setUsersBusy(true);
    hideAlert(els.usersAlert);
    const params = new URLSearchParams({ limit: String(USERS_PAGE), offset: String(Math.max(0, offset)) });
    if (els.userSearch.value.trim()) params.set("q", els.userSearch.value.trim());
    if (els.userPlanFilter.value) params.set("plan", els.userPlanFilter.value);
    if (els.userStatusFilter.value) params.set("status", els.userStatusFilter.value);
    try {
      const data = await adminCall(`/api/admin/users?${params}`, {
        box: els.usersAlert,
        failTitle: "Couldn't load accounts",
        timeout: LOAD_TIMEOUT_MS,
      });
      if (!data || data.notFound) return;
      state.users = Array.isArray(data.users) ? data.users : [];
      state.usersTotal = Number(data.total) || 0;
      state.usersOffset = Math.max(0, offset);
      state.usersLoaded = true;
      renderUsers();
    } finally {
      state.usersLoading = false;
      setUsersBusy(false);
    }
  }

  function clearUsers() {
    state.users = [];
    state.usersTotal = 0;
    state.usersOffset = 0;
    state.usersLoaded = false;
    state.usersLoading = false;
    state.selected = null;
    clearTimeout(state.searchTimer);
    els.userRows.replaceChildren(emptyRow(6, "No data yet."));
    els.usersSummary.textContent = "—";
    els.userPager.hidden = true;
    els.userDetail.hidden = true;
    setFeedback(els.usersFeedback, "");
    hideAlert(els.usersAlert);
    hideAlert(els.detailAlert);
    setUsersBusy(false);
  }

  function fact(label, value) {
    const box = make("div", "detail-fact");
    box.append(make("span", "", label), make("strong", "", value));
    return box;
  }

  function renderDetail() {
    const user = state.selected;
    if (!user) {
      els.userDetail.hidden = true;
      return;
    }
    const used = user.usage_today || {};
    const limits = user.limits || {};
    els.detailTitle.textContent = user.email;
    els.detailSub.textContent = user.name ? `${user.name} · joined ${formatAdded(user.created_at)}` : `Joined ${formatAdded(user.created_at)}`;
    const end = user.period_end ? formatAdded(user.period_end) : user.plan === "free" ? "—" : "No end date";
    els.detailFacts.replaceChildren(
      fact("Gets", `${planLabel(user.effective_plan)} limits`),
      fact("Plan · status", `${planLabel(user.plan)} · ${statusOf(user.status).label}${user.suspended ? " · Suspended" : ""}`),
      fact("Plan ends", end),
      fact("Checks today", `${formatCount(used.checks)} of ${formatCount(limits.checks_per_day)}`),
      fact("Rewrites today", `${formatCount(used.rewrites)} of ${formatCount(limits.rewrites_per_day)}`),
      fact("Last seen", formatWhen(user.last_seen))
    );
    if (state.selectedHistory) {
      const totals = state.selectedHistory.reduce(
        (sum, day) => ({ checks: sum.checks + (Number(day.checks) || 0), rewrites: sum.rewrites + (Number(day.rewrites) || 0) }),
        { checks: 0, rewrites: 0 }
      );
      els.detailFacts.append(fact("Last 30 days", `${formatCount(totals.checks)} checks · ${formatCount(totals.rewrites)} rewrites`));
    }
    els.editPlan.value = user.plan;
    els.editStatus.value = user.status;
    els.editOpenEnded.checked = user.plan !== "free" && !user.period_end;
    els.editNotes.value = user.notes || "";
    els.suspendBtn.textContent = user.suspended ? "Lift suspension" : "Suspend";
    els.userDetail.hidden = false;
  }

  async function openUser(id) {
    hideAlert(els.detailAlert);
    setFeedback(els.detailFeedback, "");
    const data = await adminCall(`/api/admin/users/${encodeURIComponent(id)}`, {
      box: els.usersAlert,
      failTitle: "Couldn't open that account",
      timeout: LOAD_TIMEOUT_MS,
    });
    if (!data) return;
    if (data.notFound) {
      setFeedback(els.usersFeedback, "That account no longer exists. The list is refreshed.", "info");
      loadUsers();
      return;
    }
    state.selected = data.user;
    state.selectedHistory = Array.isArray(data.history) ? data.history : [];
    renderDetail();
    renderUsers();
    els.userDetail.scrollIntoView({ behavior: "smooth", block: "start" });
    els.detailTitle.focus({ preventScroll: true });
  }

  /** Run an action on the open account, then refresh it and the list. */
  async function userAction(button, url, { method = "POST", body = {}, done, confirmText } = {}) {
    const user = state.selected;
    if (!user) return;
    if (confirmText && !window.confirm(confirmText)) return;
    const original = button.textContent;
    button.disabled = true;
    button.textContent = "Working…";
    hideAlert(els.detailAlert);
    setFeedback(els.detailFeedback, "");
    try {
      const data = await adminCall(url, {
        method,
        body: method === "DELETE" ? undefined : body,
        box: els.detailAlert,
        failTitle: "That change wasn't saved",
      });
      if (!data) return;
      if (data.notFound) {
        state.selected = null;
        renderDetail();
        setFeedback(els.usersFeedback, "That account no longer exists.", "info");
      } else if (data.user) {
        state.selected = data.user;
        renderDetail();
      }
      if (done) done(data);
      loadUsers();
      state.analyticsLoaded = false; // account counts changed
    } finally {
      if (button.isConnected) {
        button.disabled = false;
        button.textContent = original;
      }
    }
  }

  const userUrl = (suffix = "") =>
    `/api/admin/users/${encodeURIComponent(state.selected ? state.selected.id : "")}${suffix}`;

  function grantPlan() {
    const days = Number(els.grantDays.value);
    if (!Number.isInteger(days) || days < 1 || days > 3660) {
      showAlert(els.detailAlert, "Check the number of days", "Grant between 1 and 3,660 days.", { tone: "info" });
      els.grantDays.focus();
      return;
    }
    const plan = els.grantPlan.value;
    userAction(els.grantBtn, userUrl("/grant"), {
      body: { plan, days },
      done: (data) =>
        setFeedback(
          els.detailFeedback,
          `✓ ${planLabel(plan)} until ${formatAdded(data.user && data.user.period_end)}.`,
          "ok"
        ),
    });
  }

  function saveUser() {
    const body = {
      plan: els.editPlan.value,
      status: els.editPlan.value === "free" ? "free" : els.editStatus.value,
      notes: els.editNotes.value,
      clear_period: els.editOpenEnded.checked,
    };
    userAction(els.saveUserBtn, userUrl(), {
      method: "PATCH",
      body,
      done: () => setFeedback(els.detailFeedback, "✓ Changes saved.", "ok"),
    });
  }

  async function createUser() {
    const email = els.newUserEmail.value.trim();
    const password = els.newUserPassword.value;
    setFeedback(els.createUserFeedback, "");
    if (!email || password.length < 8) {
      setFeedback(els.createUserFeedback, "Enter an email and a password of at least 8 characters.", "error");
      return;
    }
    els.createUserBtn.disabled = true;
    try {
      const { res, data } = await adminFetch("/api/admin/users", {
        method: "POST",
        body: { email, password, name: els.newUserName.value.trim() },
      });
      if (adminAuthLost(res, data)) {
        signOut({ title: "Your session ended", message: "Sign in again to keep managing users.", tone: "info" });
        return;
      }
      if (!res.ok || !data || data.ok !== true) {
        setFeedback(els.createUserFeedback, errorOf(data).message || "The account couldn't be created.", "error");
        return;
      }
      els.newUserEmail.value = "";
      els.newUserName.value = "";
      els.newUserPassword.value = "";
      setFeedback(els.createUserFeedback, `✓ Created ${data.user.email}.`, "ok");
      state.analyticsLoaded = false;
      await loadUsers({ offset: 0 });
    } catch {
      setFeedback(els.createUserFeedback, "Couldn't reach the app.", "error");
    } finally {
      els.createUserBtn.disabled = false;
    }
  }

  function wireUsers() {
    els.refreshUsersBtn.addEventListener("click", () => loadUsers());
    els.userSearch.addEventListener("input", () => {
      clearTimeout(state.searchTimer);
      state.searchTimer = setTimeout(() => loadUsers({ offset: 0 }), 300);
    });
    els.userPlanFilter.addEventListener("change", () => loadUsers({ offset: 0 }));
    els.userStatusFilter.addEventListener("change", () => loadUsers({ offset: 0 }));
    els.usersPrev.addEventListener("click", () => loadUsers({ offset: state.usersOffset - USERS_PAGE }));
    els.usersNext.addEventListener("click", () => loadUsers({ offset: state.usersOffset + USERS_PAGE }));
    els.closeDetail.addEventListener("click", () => {
      state.selected = null;
      renderDetail();
      renderUsers();
    });
    els.grantBtn.addEventListener("click", grantPlan);
    els.saveUserBtn.addEventListener("click", saveUser);
    els.editPlan.addEventListener("change", () => {
      if (els.editPlan.value === "free") els.editStatus.value = "free";
      else if (els.editStatus.value === "free") els.editStatus.value = "active";
    });
    els.resetUsageBtn.addEventListener("click", () =>
      userAction(els.resetUsageBtn, userUrl("/reset-usage"), {
        done: () => setFeedback(els.detailFeedback, "✓ Today's usage was reset.", "ok"),
      })
    );
    els.signOutUserBtn.addEventListener("click", () =>
      userAction(els.signOutUserBtn, userUrl("/sign-out"), {
        done: (data) => setFeedback(els.detailFeedback, `✓ Signed out of ${plural(Number(data.sessions_ended) || 0, "session")}.`, "ok"),
      })
    );
    els.suspendBtn.addEventListener("click", () => {
      const user = state.selected;
      if (!user) return;
      const suspend = !user.suspended;
      userAction(els.suspendBtn, userUrl(), {
        method: "PATCH",
        body: { suspended: suspend },
        confirmText: suspend ? `Suspend ${user.email}? They'll be signed out and can't sign in until you lift it.` : "",
        done: () => setFeedback(els.detailFeedback, suspend ? "✓ Account suspended." : "✓ Suspension lifted.", "ok"),
      });
    });
    els.deleteUserBtn.addEventListener("click", () => {
      const user = state.selected;
      if (!user) return;
      userAction(els.deleteUserBtn, userUrl(), {
        method: "DELETE",
        confirmText: `Delete ${user.email}? This removes the account and its usage history. It can't be undone.`,
        done: () => {
          state.selected = null;
          renderDetail();
          setFeedback(els.usersFeedback, `Deleted ${user.email}.`, "ok");
        },
      });
    });
    els.createUserBtn.addEventListener("click", createUser);
  }

  /* ─────────────────────────── wire up ─────────────────────────── */
  function wireToggle(input, button, noun) {
    button.addEventListener("click", () => {
      const showing = input.type === "text";
      input.type = showing ? "password" : "text";
      button.textContent = showing ? "Show" : "Hide";
      button.setAttribute("aria-label", showing ? `Show ${noun}` : `Hide ${noun}`);
    });
  }

  function wire() {
    document.querySelectorAll(".alert-close").forEach((button) => {
      button.addEventListener("click", () => {
        const box = button.closest(".alert");
        if (box) box.hidden = true;
      });
    });

    els.signInBtn.addEventListener("click", attemptSignIn);
    els.adminToken.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        attemptSignIn();
      }
    });
    els.adminToken.addEventListener("input", () => hideAlert(els.gateAlert));
    wireToggle(els.adminToken, els.toggleToken, "token");
    wireToggle(els.newAdminKey, els.toggleNewKey, "key");

    const addOnEnter = (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        addKey();
      }
    };
    els.addKeyBtn.addEventListener("click", addKey);
    els.newAdminKey.addEventListener("keydown", addOnEnter);
    els.newKeyLabel.addEventListener("keydown", addOnEnter);
    els.newAdminKey.addEventListener("input", () => {
      hideAlert(els.addAlert);
      setFeedback(els.addFeedback, "");
    });

    els.refreshKeysBtn.addEventListener("click", () => loadData());
    els.refreshProvidersBtn.addEventListener("click", () => loadData());
    els.analyticsRefreshBtn.addEventListener("click", () => loadAnalytics({ force: true }));
    wireUsers();
    els.signOutBtn.addEventListener("click", () => signOut());
    window.addEventListener("hashchange", () => {
      if (state.token && !els.app.hidden) showView(location.hash.slice(1));
    });
  }

  /* ─────────────────────────── init ─────────────────────────── */
  function init() {
    wire();
    const saved = store.get();
    if (saved) {
      state.token = saved;
      attemptSignIn();
    } else {
      els.adminToken.focus();
    }
  }

  init();
})();
