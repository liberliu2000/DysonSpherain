"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Archive,
  BrainCircuit,
  Calculator,
  CheckCircle2,
  Clock3,
  DatabaseZap,
  FilePenLine,
  Layers3,
  Lock,
  Search,
  Send,
  Settings,
  ShieldCheck,
  Sparkles,
  TrendingUp,
  WalletCards
} from "lucide-react";
import { MemoryEditor } from "@/components/memory-editor";
import { MetricCard } from "@/components/metric-card";
import { RouteCard } from "@/components/route-card";
import { SoftButton } from "@/components/soft-button";
import { SoftPanel } from "@/components/soft-panel";
import { TokenTrendChart } from "@/components/token-trend-chart";
import { TopNav } from "@/components/top-nav";
import { getDashboardData, type Locale } from "@/lib/dashboard-data";

type ApiTokenEvent = { estimated_saved_tokens?: number; updated_at?: string };
type ApiTokenSummary = {
  status?: string;
  windows?: Record<string, { estimated_saved_tokens?: number; saving_ratio?: number; saved_ratio?: number; event_count?: number }>;
  events?: ApiTokenEvent[];
  llm_prompt_token_economy?: { estimated_saved_tokens?: number; saving_ratio?: number; saved_ratio?: number; event_count?: number };
};
type LifecycleRecord = {
  id: string;
  title: string;
  content?: string;
  scope?: string;
  state?: string;
  updated_at?: string;
  raw?: Record<string, unknown>;
  tags?: string[];
  why?: string;
  source_ids?: string[];
};
type LifecycleSummary = {
  status?: string;
  state_counts?: Record<string, number>;
  total_memories?: number;
  eligible_memories?: number;
  records?: LifecycleRecord[];
  recent_retrieval_traces?: Array<{ trace_id: string; query: string; route: string; selected_count: number; excluded_count: number; excluded?: Array<{ reason?: string; capsule_id?: string }> }>;
  retrieval_policy?: { eligible_by_default?: string[]; excluded_by_default?: string[]; raw_memory_preserved?: boolean };
};
type CompactionCandidate = {
  cluster_id: string;
  reason: string;
  memory_ids: string[];
  memory_count: number;
  avg_similarity?: number;
  redundancy_score?: number;
  importance_score?: number;
  estimated_token_savings_ratio?: number;
  source_titles: string[];
  estimated_saved_tokens: number;
  canonical_preview: string;
};
type CompactionResult = {
  result_id: string;
  status: string;
  canonical_content: string;
  source_ids: string[];
  method: string;
  verifier_passed?: boolean;
  warnings?: string[];
  estimated_saved_tokens?: number;
  estimated_token_savings_ratio?: number;
  canonical_id?: string;
  external_llm_attempted?: boolean;
  external_llm?: Record<string, unknown> | null;
};
type SettingsPayload = {
  llm_config?: Record<string, unknown>;
  compaction_config?: Record<string, unknown>;
  scoring_config?: Record<string, unknown>;
  lifecycle_multipliers?: Record<string, unknown>;
  privacy_config?: Record<string, unknown>;
};

const apiBase = process.env.NEXT_PUBLIC_DYSON_API_BASE || "http://127.0.0.1:37777";

function numberFormatter(locale: Locale) {
  return new Intl.NumberFormat(locale === "zh" ? "zh-CN" : "en", { maximumFractionDigits: 1, notation: "compact" });
}

function percentFormatter(locale: Locale) {
  return new Intl.NumberFormat(locale === "zh" ? "zh-CN" : "en", { maximumFractionDigits: 1, style: "percent" });
}

function compactNumber(value: number | undefined, locale: Locale) {
  return typeof value === "number" && Number.isFinite(value) ? numberFormatter(locale).format(value) : "";
}

function formatPercent(value: number | undefined, locale: Locale) {
  return percentFormatter(locale).format(Math.max(0, Math.min(1, Number(value || 0))));
}

function formatDate(value: string | undefined, locale: Locale) {
  if (!value) {
    return locale === "zh" ? "未知" : "unknown";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat(locale === "zh" ? "zh-CN" : "en", { dateStyle: "medium", timeStyle: "short" }).format(date);
}

function tokenTrend(events: ApiTokenEvent[] | undefined, fallback: ReturnType<typeof getDashboardData>["tokenTrend"], locale: Locale) {
  if (!events?.length) {
    return fallback;
  }
  const byDay = new Map<string, number>();
  for (const event of events) {
    const date = event.updated_at ? new Date(event.updated_at) : null;
    if (!date || Number.isNaN(date.getTime())) {
      continue;
    }
    const key = date.toISOString().slice(0, 10);
    byDay.set(key, (byDay.get(key) || 0) + Number(event.estimated_saved_tokens || 0));
  }
  const labels = locale === "zh" ? ["日", "一", "二", "三", "四", "五", "六"] : ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  const today = new Date();
  return Array.from({ length: 7 }, (_, offset) => {
    const date = new Date(today);
    date.setDate(today.getDate() - (6 - offset));
    return { label: labels[date.getDay()], saved: byDay.get(date.toISOString().slice(0, 10)) || 0 };
  });
}

const copy = {
  en: {
    actions: { compact: "Run deterministic compaction", openRecords: "Open records", retrieve: "Explain retrieval", saveSettings: "Save settings", searchMemory: "Search memory" },
    calculation: {
      title: "Token saving logic",
      description: "Saved tokens are estimated as max(0, original context tokens - final memory pack tokens - retrieval overhead). The 1h, 24h, 7d, and total cards sum the recorded session estimates.",
      example: "Example: 9,200 original tokens - 2,150 memory pack tokens - 320 retrieval overhead = 6,730 saved tokens."
    },
    hero: {
      eyebrow: "Memory OS Cockpit",
      title: "Control what the system remembers.",
      lead: "Inspect active, stale, superseded, compacted, and excluded memory. Keep token savings visible while every retrieval decision remains traceable."
    },
    labels: { live: "live", memoryQuery: "Memory query", savedTokenMetrics: "Saved token metrics", tokenEconomyState: "Token economy state" },
    memoryEditor: { content: "Memory content", empty: "No memory records found.", saved: "saved", saveMemory: "Save memory", searchAria: "Memory records", searchPlaceholder: "Search memories", sources: "Source IDs", title: "Title", why: "Why this state" },
    nav: {
      brandSubtitle: "Memory OS",
      languageLabel: "Switch language",
      links: [
        { href: "#tokens", label: "Tokens" },
        { href: "#lifecycle", label: "Lifecycle" },
        { href: "#memories", label: "Memories" },
        { href: "#retrieval", label: "Retrieval" },
        { href: "#settings", label: "Settings" }
      ],
      toggleNavigation: "Toggle navigation"
    },
    panels: {
      compaction: "Compaction queue",
      compactionDescription: "Deterministic compaction creates a canonical memory and preserves every raw source.",
      lifecycle: "Lifecycle map",
      lifecycleDescription: "Counts reflect retrieval behavior: active, stable, and canonical memories are eligible by default; stale states stay traceable but are excluded.",
      memoryDescription: "Review and edit existing memories directly from the console.",
      memoryTitle: "Memory records",
      retrieval: "Retrieval inspector",
      retrievalDescription: "Ask why the system selected or ignored memories for a task.",
      settings: "LLM and privacy settings",
      settingsDescription: "External LLM compaction is optional and disabled by default. Local-only mode remains the default.",
      trend: "Token savings trend"
    },
    query: { placeholder: "Ask for prior decisions, implementation notes, constraints, or a compact context pack..." },
    chart: { aria: "Seven day saved-token trend chart", peak: "peak", subtitle: "7-day memory reuse signal", title: "Saved-token trend" },
    settings: { external: "External LLM access", externalCompaction: "Enable LLM compaction", localOnly: "Local-only mode", raw: "Allow raw memory external", sourceIds: "Require source IDs", verifier: "Require verifier" },
    states: { active: "Active", archived: "Archived", canonical: "Canonical", compacted: "Compacted", contradicted: "Contradicted", deprecated: "Deprecated", stable: "Stable", superseded: "Superseded" }
  },
  zh: {
    actions: { compact: "运行确定性压缩", openRecords: "打开记忆记录", retrieve: "解释召回", saveSettings: "保存设置", searchMemory: "搜索记忆" },
    calculation: {
      title: "Token 节省计算逻辑",
      description: "节省 token 估算为 max(0, 原始上下文 token - 最终记忆包 token - 检索开销)。近 1 小时、近 24 小时、近 7 天和累计卡片会汇总已记录会话估算。",
      example: "示例：9,200 原始 token - 2,150 记忆包 token - 320 检索开销 = 6,730 节省 token。"
    },
    hero: {
      eyebrow: "记忆操作系统控制台",
      title: "控制系统记住什么。",
      lead: "查看 active、stale、superseded、compacted 与 excluded 记忆。在展示 token 节省的同时，让每次召回决策都有迹可查。"
    },
    labels: { live: "实时", memoryQuery: "记忆查询", savedTokenMetrics: "节省 token 指标", tokenEconomyState: "Token economy 状态" },
    memoryEditor: { content: "记忆内容", empty: "没有找到记忆记录。", saved: "已保存", saveMemory: "保存记忆", searchAria: "记忆记录", searchPlaceholder: "搜索记忆", sources: "来源 ID", title: "标题", why: "状态原因" },
    nav: {
      brandSubtitle: "记忆操作系统",
      languageLabel: "切换语言",
      links: [
        { href: "#tokens", label: "Token" },
        { href: "#lifecycle", label: "生命周期" },
        { href: "#memories", label: "记忆" },
        { href: "#retrieval", label: "召回" },
        { href: "#settings", label: "设置" }
      ],
      toggleNavigation: "展开导航"
    },
    panels: {
      compaction: "压缩队列",
      compactionDescription: "确定性压缩会创建 canonical 记忆，并保留所有原始来源。",
      lifecycle: "生命周期地图",
      lifecycleDescription: "计数对应召回行为：active、stable 和 canonical 默认可用；陈旧状态保留追踪但默认排除。",
      memoryDescription: "直接在控制台中查看并编辑已有记忆。",
      memoryTitle: "记忆记录",
      retrieval: "召回解释器",
      retrievalDescription: "询问系统为何选择或忽略某些记忆。",
      settings: "LLM 与隐私设置",
      settingsDescription: "外部 LLM 压缩可选且默认关闭。本地优先模式仍为默认。",
      trend: "Token 节省趋势"
    },
    query: { placeholder: "询问历史决策、实现笔记、约束，或请求紧凑上下文包..." },
    chart: { aria: "七天节省 token 趋势图", peak: "峰值", subtitle: "7 天记忆复用信号", title: "节省 token 趋势" },
    settings: { external: "外部 LLM 访问", externalCompaction: "启用 LLM 压缩", localOnly: "仅本地模式", raw: "允许原始记忆发往外部", sourceIds: "要求来源 ID", verifier: "要求 verifier" },
    states: { active: "活跃", archived: "归档", canonical: "规范记忆", compacted: "已压缩", contradicted: "冲突", deprecated: "弃用", stable: "稳定", superseded: "被替代" }
  }
};

export default function Home() {
  const [locale, setLocale] = useState<Locale>("en");
  const [query, setQuery] = useState("");
  const [retrieval, setRetrieval] = useState<Record<string, unknown> | null>(null);
  const [tokenSummary, setTokenSummary] = useState<ApiTokenSummary | null>(null);
  const [lifecycle, setLifecycle] = useState<LifecycleSummary | null>(null);
  const [candidates, setCandidates] = useState<CompactionCandidate[]>([]);
  const [settings, setSettings] = useState<SettingsPayload>({});
  const [notice, setNotice] = useState("");
  const [memoryStateFilter, setMemoryStateFilter] = useState("all");
  const [memoryTextFilter, setMemoryTextFilter] = useState("");
  const [memorySourceFilter, setMemorySourceFilter] = useState("");
  const [memoryTypeFilter, setMemoryTypeFilter] = useState("all");
  const [memorySort, setMemorySort] = useState("recency");
  const [memoryView, setMemoryView] = useState<"cards" | "table">("cards");
  const [selectedMemoryId, setSelectedMemoryId] = useState("");
  const [confirmExternalCall, setConfirmExternalCall] = useState(false);
  const [compactionResult, setCompactionResult] = useState<CompactionResult | null>(null);
  const [successorId, setSuccessorId] = useState("");
  const recallRef = useRef<HTMLTextAreaElement>(null);

  const t = copy[locale];
  const fallback = useMemo(() => getDashboardData(locale), [locale]);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      fetch(`${apiBase}/api/token-economy`, { signal: controller.signal }).then((response) => (response.ok ? response.json() : null)),
      fetch(`${apiBase}/api/lifecycle/summary`, { signal: controller.signal }).then((response) => (response.ok ? response.json() : null)),
      fetch(`${apiBase}/api/lifecycle/compaction-candidates`, { signal: controller.signal }).then((response) => (response.ok ? response.json() : null)),
      fetch(`${apiBase}/api/settings`, { signal: controller.signal }).then((response) => (response.ok ? response.json() : null))
    ])
      .then(([tokens, life, compact, runtime]) => {
        if (tokens?.status === "ok") setTokenSummary(tokens);
        if (life?.status === "ok") setLifecycle(life);
        if (compact?.status === "ok") setCandidates(compact.candidates || []);
        if (runtime?.status === "ok") setSettings(runtime.settings || {});
      })
      .catch(() => undefined);
    return () => controller.abort();
  }, []);

  const tokenSavings = useMemo(() => {
    const source = [tokenSummary?.windows?.["1h"], tokenSummary?.windows?.["24h"], tokenSummary?.windows?.["7d"], tokenSummary?.llm_prompt_token_economy || tokenSummary?.windows?.["30d"]];
    const icons = [Clock3, TrendingUp, Archive, WalletCards];
    return fallback.tokenSavings.map((item, index) => {
      const window = source[index];
      if (!window) return { ...item, icon: icons[index] };
      return {
        ...item,
        icon: icons[index],
        value: compactNumber(window.estimated_saved_tokens, locale) || item.value,
        trend: locale === "zh" ? `${formatPercent(window.saving_ratio || window.saved_ratio, locale)} 节省率，${window.event_count ?? 0} 条记录` : `${formatPercent(window.saving_ratio || window.saved_ratio, locale)} saved from ${window.event_count ?? 0} records`
      };
    });
  }, [fallback.tokenSavings, locale, tokenSummary]);

  const memoryRecords = useMemo(() => {
    const records: Array<Record<string, unknown>> = lifecycle?.records?.length ? lifecycle.records : fallback.memoryRecords;
    return records.map((record) => ({
      id: String(record.id || ""),
      title: String(record.title || ""),
      scope: String(record.scope || "memory"),
      state: String(record.state || "active"),
      updatedAt: record.updated_at ? formatDate(String(record.updated_at), locale) : String(record.updatedAt || ""),
      updatedRaw: String(record.updated_at || record.updatedAt || ""),
      tags: Array.isArray(record.tags) ? record.tags.map(String) : [],
      content: String(record.content || ""),
      why: record.why ? String(record.why) : undefined,
      sourceIds: Array.isArray(record.source_ids) ? record.source_ids.map(String) : undefined,
      raw: record
    })).filter((record) => {
      const haystack = `${record.id} ${record.title} ${record.content} ${record.tags.join(" ")} ${record.why || ""}`.toLowerCase();
      const sourceHaystack = (record.sourceIds || []).join(" ").toLowerCase();
      const type = record.tags.includes("canonical") || record.scope === "canonical" ? "canonical" : record.state === "compacted" ? "raw" : "raw";
      return (memoryStateFilter === "all" || record.state === memoryStateFilter)
        && (!memoryTextFilter.trim() || haystack.includes(memoryTextFilter.trim().toLowerCase()))
        && (!memorySourceFilter.trim() || sourceHaystack.includes(memorySourceFilter.trim().toLowerCase()))
        && (memoryTypeFilter === "all" || type === memoryTypeFilter);
    }).sort((a, b) => {
      if (memorySort === "state") return a.state.localeCompare(b.state);
      if (memorySort === "title") return a.title.localeCompare(b.title);
      return String(b.updatedRaw).localeCompare(String(a.updatedRaw));
    });
  }, [fallback.memoryRecords, lifecycle?.records, locale, memorySourceFilter, memorySort, memoryStateFilter, memoryTextFilter, memoryTypeFilter]);

  const conflictRecords = useMemo(() => {
    const records = lifecycle?.records || [];
    return records.filter((record) => ["superseded", "deprecated", "contradicted"].includes(record.state || "")).slice(0, 8);
  }, [lifecycle?.records]);

  const lifecycleCards = useMemo(() => {
    const counts = lifecycle?.state_counts || {};
    return (["active", "stable", "canonical", "compacted", "superseded", "deprecated", "contradicted", "archived"] as const).map((state) => ({
      name: t.states[state],
      status: String(counts[state] || 0),
      detail: state === "canonical" ? "Consolidated memory eligible for retrieval." : `${state} memory lifecycle state.`,
      progress: Math.min(100, Number(counts[state] || 0) * 10)
    }));
  }, [lifecycle?.state_counts, t.states]);

  async function retrieveContext() {
    const text = query.trim();
    if (!text) {
      recallRef.current?.focus();
      return;
    }
    const response = await fetch(`${apiBase}/api/retrieval/inspect`, {
      body: JSON.stringify({ query: text, limit: 8, max_tokens: 1800 }),
      headers: { "Content-Type": "application/json" },
      method: "POST"
    });
    setRetrieval(response.ok ? await response.json() : { status: "error" });
  }

  const externalCompactionReady = Boolean(settings.llm_config?.external_llm_enabled)
    && Boolean(settings.compaction_config?.external_llm_compaction_enabled)
    && !Boolean(settings.llm_config?.local_only)
    && !Boolean(settings.privacy_config?.local_only);

  async function compact(candidate: CompactionCandidate) {
    const selectedMode = String(settings.compaction_config?.mode || (candidate.reason.includes("near") ? "local_semantic" : "deterministic"));
    if (["hybrid", "llm"].includes(selectedMode) && (!externalCompactionReady || !confirmExternalCall)) {
      setNotice(locale === "zh" ? "LLM 压缩需要在设置中启用外部访问、启用 LLM 压缩、关闭仅本地模式，并勾选本次确认。" : "LLM compaction requires external access, LLM compaction enabled, local-only off, and per-run confirmation.");
      return;
    }
    const response = await fetch(`${apiBase}/api/compaction/clusters/${encodeURIComponent(candidate.cluster_id)}/run`, {
      body: JSON.stringify({ mode: selectedMode, verifier: "ui", confirm_external_call: confirmExternalCall }),
      headers: { "Content-Type": "application/json" },
      method: "POST"
    });
    const payload = await response.json();
    setCompactionResult(payload.result || null);
    setNotice(payload.status === "ok" ? `Compaction preview ready: ${payload.result?.result_id}` : String(payload.error || "Compaction failed"));
  }

  async function actOnCompaction(action: "verify" | "commit" | "reject") {
    if (!compactionResult) return;
    const response = await fetch(`${apiBase}/api/compaction/results/${encodeURIComponent(compactionResult.result_id)}/${action}`, {
      body: JSON.stringify({ reason: "Rejected from UI" }),
      headers: { "Content-Type": "application/json" },
      method: "POST"
    });
    const payload = await response.json();
    setCompactionResult(payload.result || null);
    setNotice(action === "commit" && payload.canonical_id ? `Canonical memory committed: ${payload.canonical_id}` : `${action}: ${payload.status}`);
    const [life, compacted] = await Promise.all([fetch(`${apiBase}/api/lifecycle/summary`).then((item) => item.json()), fetch(`${apiBase}/api/lifecycle/compaction-candidates`).then((item) => item.json())]);
    if (life?.status === "ok") setLifecycle(life);
    if (compacted?.status === "ok") setCandidates(compacted.candidates || []);
  }

  async function markMemory(record: LifecycleRecord, action: "deprecated" | "superseded") {
    const endpoint = action === "deprecated" ? "mark-deprecated" : "mark-superseded";
    const body = action === "deprecated" ? { reason: "Marked from Memory Cockpit" } : { by: successorId, reason: "Superseded from Memory Cockpit" };
    if (action === "superseded" && !successorId.trim()) {
      setNotice("Enter a successor memory id first.");
      return;
    }
    const response = await fetch(`${apiBase}/api/memory/${encodeURIComponent(record.id)}/${endpoint}`, {
      body: JSON.stringify(body),
      headers: { "Content-Type": "application/json" },
      method: "POST"
    });
    const payload = await response.json();
    setNotice(payload.status === "ok" ? `${record.id} marked ${action}` : String(payload.error || "Update failed"));
    const life = await fetch(`${apiBase}/api/lifecycle/summary`).then((item) => item.json());
    if (life?.status === "ok") setLifecycle(life);
  }

  async function memoryAction(record: LifecycleRecord, action: "mark-contradicted" | "reactivate" | "archive") {
    if (action === "mark-contradicted" && !successorId.trim()) {
      setNotice("Enter a conflicting/successor memory id first.");
      return;
    }
    const response = await fetch(`${apiBase}/api/memory/${encodeURIComponent(record.id)}/${action}`, {
      body: JSON.stringify({ by: successorId, reason: `${action} from Memory Cockpit` }),
      headers: { "Content-Type": "application/json" },
      method: "POST"
    });
    const payload = await response.json();
    setNotice(payload.status === "ok" ? `${record.id}: ${action}` : String(payload.error || "Update failed"));
    const life = await fetch(`${apiBase}/api/lifecycle/summary`).then((item) => item.json());
    if (life?.status === "ok") setLifecycle(life);
  }

  async function saveSettings() {
    const response = await fetch(`${apiBase}/api/settings`, {
      body: JSON.stringify(settings),
      headers: { "Content-Type": "application/json" },
      method: "POST"
    });
    const payload = await response.json();
    setSettings(payload.config || settings);
    setNotice(payload.status === "ok" ? "Settings saved" : "Settings save failed");
  }

  function setSetting(group: "llm_config" | "compaction_config" | "scoring_config" | "lifecycle_multipliers" | "privacy_config", key: string, value: unknown) {
    setSettings((current) => ({ ...current, [group]: { ...(current[group] || {}), [key]: value } }));
  }

  const selected = Array.isArray(retrieval?.final_context) ? retrieval.final_context : [];
  const excluded = (Array.isArray(retrieval?.excluded_evidence) ? retrieval.excluded_evidence : lifecycle?.recent_retrieval_traces?.[0]?.excluded || []) as Array<{ reason?: string; capsule_id?: string; memory_id?: string }>;
  const stageCounts = (retrieval?.stage_counts || {}) as Record<string, number>;
  const selectedMemory = memoryRecords.find((record) => record.id === selectedMemoryId) || memoryRecords[0];

  return (
    <main className="app-shell" id="top">
      <div className="dashboard">
        <TopNav labels={t.nav} locale={locale} onToggleLocale={() => setLocale((value) => (value === "en" ? "zh" : "en"))} />

        <section className="hero-grid" aria-labelledby="page-title">
          <div className="hero-panel soft-raised">
            <span className="eyebrow soft-inset">
              <Sparkles size={17} strokeWidth={2.3} aria-hidden="true" />
              {t.hero.eyebrow}
            </span>
            <h1 className="headline font-display" id="page-title">{t.hero.title}</h1>
            <p className="lead">{t.hero.lead}</p>
            <div className="action-row">
              <SoftButton icon={Search} variant="primary" onClick={() => document.getElementById("retrieval")?.scrollIntoView({ behavior: "smooth" })}>{t.actions.searchMemory}</SoftButton>
              <SoftButton icon={DatabaseZap} variant="secondary" onClick={() => document.getElementById("memories")?.scrollIntoView({ behavior: "smooth" })}>{t.actions.openRecords}</SoftButton>
            </div>
          </div>
          <aside className="status-stack" aria-label={t.labels.tokenEconomyState}>
            <div className="depth-orb soft-raised"><div className="orb-core soft-raised"><BrainCircuit size={32} strokeWidth={2.1} aria-hidden="true" /></div></div>
            <SoftPanel title={t.panels.lifecycle} description={t.panels.lifecycleDescription} icon={ShieldCheck}>
              <div className="route-list">
                <div className="route-item soft-inset">
                  <div className="route-row"><strong>{locale === "zh" ? "默认可召回" : "Eligible by default"}</strong><span className="badge soft-small good">{lifecycle?.eligible_memories ?? 0}/{lifecycle?.total_memories ?? 0}</span></div>
                  <p>{lifecycle?.retrieval_policy?.raw_memory_preserved ? "Raw memory is preserved during compaction." : "Lifecycle data will appear after the daemon is connected."}</p>
                </div>
              </div>
            </SoftPanel>
          </aside>
        </section>

        <section className="metric-grid section" id="tokens" aria-label={t.labels.savedTokenMetrics}>
          {tokenSavings.map((metric) => <MetricCard key={metric.label} {...metric} liveLabel={t.labels.live} />)}
        </section>

        <section className="workspace-grid section">
          <SoftPanel title={t.panels.trend} description={t.calculation.description} icon={Calculator}>
            <TokenTrendChart ariaLabel={t.chart.aria} data={tokenTrend(tokenSummary?.events, fallback.tokenTrend, locale)} locale={locale} peakLabel={t.chart.peak} subtitle={t.chart.subtitle} title={t.chart.title} />
          </SoftPanel>
          <SoftPanel title={t.calculation.title} description={t.calculation.description} icon={Lock}>
            <div className="calculation-box soft-inset">
              <p>{t.calculation.example}</p>
              <code>saved = max(0, original_context - memory_pack - retrieval_overhead)</code>
            </div>
          </SoftPanel>
        </section>

        <section className="section" id="lifecycle">
          <SoftPanel title={t.panels.lifecycle} description={t.panels.lifecycleDescription} icon={Layers3}>
            <div className="route-list lifecycle-grid">
              {lifecycleCards.map((item) => <RouteCard key={item.name} {...item} />)}
            </div>
          </SoftPanel>
        </section>

        <section className="workspace-grid section">
          <SoftPanel title={t.panels.compaction} description={t.panels.compactionDescription} icon={FilePenLine}>
            <div className="route-list">
              {candidates.length ? candidates.slice(0, 4).map((candidate) => (
                <div className="route-item soft-inset" key={candidate.cluster_id}>
                  <div className="route-row"><strong>{candidate.source_titles[0] || candidate.cluster_id}</strong><span className="badge soft-small good">{compactNumber(candidate.estimated_saved_tokens, locale)} tokens</span></div>
                  <p>{candidate.reason} · {candidate.memory_count} memories · similarity {formatPercent(candidate.avg_similarity || candidate.redundancy_score, locale)} · savings {formatPercent(candidate.estimated_token_savings_ratio, locale)}</p>
                  <p>{candidate.canonical_preview}</p>
                  {String(settings.compaction_config?.mode || "deterministic").match(/hybrid|llm/) ? (
                    <label className="confirm-row soft-small">
                      <input type="checkbox" checked={confirmExternalCall} onChange={(event) => setConfirmExternalCall(event.target.checked)} />
                      <span>{locale === "zh" ? "我确认本次可调用外部 LLM 压缩" : "Confirm external LLM call for this run"}</span>
                    </label>
                  ) : null}
                  <SoftButton icon={CheckCircle2} variant="secondary" onClick={() => compact(candidate)}>{t.actions.compact}</SoftButton>
                </div>
              )) : <p className="panel-copy">No duplicate compaction candidates found.</p>}
              {compactionResult ? (
                <div className="result-box soft-inset">
                  <strong>{locale === "zh" ? "压缩结果预览" : "Compaction result preview"}</strong>
                  <p>{compactionResult.result_id} · {compactionResult.status} · {compactionResult.method} · saved {compactNumber(compactionResult.estimated_saved_tokens, locale)} tokens</p>
                  <p>{compactionResult.canonical_content}</p>
                  <p>sources: {compactionResult.source_ids.join(", ")}</p>
                  {compactionResult.external_llm_attempted ? <p>external LLM: {compactionResult.external_llm ? "used" : "not used"}</p> : null}
                  {compactionResult.warnings?.length ? <p>warnings: {compactionResult.warnings.join(", ")}</p> : null}
                  <div className="editor-actions">
                    <SoftButton icon={ShieldCheck} variant="secondary" onClick={() => actOnCompaction("verify")}>Verify</SoftButton>
                    <SoftButton icon={CheckCircle2} variant="primary" onClick={() => actOnCompaction("commit")}>Commit</SoftButton>
                    <SoftButton icon={Archive} variant="secondary" onClick={() => actOnCompaction("reject")}>Reject</SoftButton>
                  </div>
                </div>
              ) : null}
              {notice ? <div className="result-box soft-inset">{notice}</div> : null}
            </div>
          </SoftPanel>
          <SoftPanel title={t.panels.retrieval} description={t.panels.retrievalDescription} icon={Search}>
            <form className="query-box" onSubmit={(event) => event.preventDefault()}>
              <label>
                <span className="sr-only">{t.labels.memoryQuery}</span>
                <textarea ref={recallRef} className="soft-input textarea" value={query} onChange={(event) => setQuery(event.target.value)} placeholder={t.query.placeholder} />
              </label>
              <SoftButton icon={Send} variant="primary" onClick={retrieveContext}>{t.actions.retrieve}</SoftButton>
              <div className="retrieval-grid">
                <div className="result-box soft-inset"><strong>{locale === "zh" ? "流水线" : "Pipeline"}</strong><p>{Object.keys(stageCounts).length ? Object.entries(stageCounts).map(([key, value]) => `${key}: ${value}`).join(" · ") : "No retrieval run yet."}</p></div>
                <div className="result-box soft-inset"><strong>{locale === "zh" ? "已选择" : "Selected"}</strong><p>{selected.length ? selected.map((item: any) => `${item.rank}. ${item.title || item.memory_id} (${item.final_score ?? "n/a"})`).slice(0, 5).join(" · ") : "No retrieval run yet."}</p></div>
                <div className="result-box soft-inset"><strong>{locale === "zh" ? "已排除" : "Excluded"}</strong><p>{excluded.length ? excluded.map((item) => item.reason || item.memory_id || item.capsule_id).slice(0, 5).join(" · ") : "No excluded evidence yet."}</p></div>
              </div>
            </form>
          </SoftPanel>
        </section>

        <section className="section" id="memories" aria-labelledby="memory-title">
          <SoftPanel title={t.panels.memoryTitle} description={t.panels.memoryDescription} icon={DatabaseZap}>
            <div className="tag-row filter-row" aria-label="Memory state filters">
              {["all", "active", "stable", "canonical", "compacted", "superseded", "deprecated", "contradicted", "archived"].map((state) => (
                <button className={`badge soft-small ${memoryStateFilter === state ? "good" : ""}`} type="button" key={state} onClick={() => setMemoryStateFilter(state)}>
                  {state}
                </button>
              ))}
            </div>
            <div className="settings-grid memory-tools">
              <label><span className="field-label">{locale === "zh" ? "文本 / 项目 / 主题" : "Text / project / topic"}</span><input className="soft-input compact-input" value={memoryTextFilter} onChange={(event) => setMemoryTextFilter(event.target.value)} placeholder={t.memoryEditor.searchPlaceholder} /></label>
              <label><span className="field-label">{locale === "zh" ? "来源 ID" : "Source ID"}</span><input className="soft-input compact-input" value={memorySourceFilter} onChange={(event) => setMemorySourceFilter(event.target.value)} placeholder="cap_..." /></label>
              <label><span className="field-label">{locale === "zh" ? "类型" : "Type"}</span><select className="soft-input compact-input" value={memoryTypeFilter} onChange={(event) => setMemoryTypeFilter(event.target.value)}><option value="all">all</option><option value="raw">raw</option><option value="canonical">canonical</option></select></label>
              <label><span className="field-label">{locale === "zh" ? "排序" : "Sort"}</span><select className="soft-input compact-input" value={memorySort} onChange={(event) => setMemorySort(event.target.value)}><option value="recency">recency</option><option value="state">state</option><option value="title">title</option></select></label>
            </div>
            <div className="editor-actions">
              <SoftButton icon={DatabaseZap} variant={memoryView === "cards" ? "primary" : "secondary"} onClick={() => setMemoryView("cards")}>Cards</SoftButton>
              <SoftButton icon={Layers3} variant={memoryView === "table" ? "primary" : "secondary"} onClick={() => setMemoryView("table")}>Table</SoftButton>
            </div>
            {memoryView === "table" ? (
              <div className="memory-table soft-inset" role="table" aria-label={t.memoryEditor.searchAria}>
                <div className="memory-table-row memory-table-head" role="row"><span>Title</span><span>State</span><span>Updated</span><span>Sources</span></div>
                {memoryRecords.map((record) => (
                  <button className="memory-table-row" type="button" role="row" key={record.id} onClick={() => setSelectedMemoryId(record.id)}>
                    <span>{record.title || record.id}</span><span>{record.state}</span><span>{record.updatedAt}</span><span>{record.sourceIds?.length || 0}</span>
                  </button>
                ))}
              </div>
            ) : <MemoryEditor labels={t.memoryEditor} records={memoryRecords} apiBase={apiBase} />}
            {selectedMemory ? (
              <details className="result-box soft-inset memory-detail" open>
                <summary>{locale === "zh" ? "所选记忆详情 / 审计线索" : "Selected memory detail / audit trail"}</summary>
                <p><strong>{selectedMemory.title || selectedMemory.id}</strong> · {selectedMemory.state} · {selectedMemory.updatedAt}</p>
                <p>{selectedMemory.why || selectedMemory.content}</p>
                <pre>{JSON.stringify(selectedMemory.raw, null, 2)}</pre>
              </details>
            ) : null}
          </SoftPanel>
        </section>

        <section className="section" id="conflicts">
          <SoftPanel title={locale === "zh" ? "替代与冲突审查" : "Supersession and conflict review"} description={locale === "zh" ? "查看被替代、弃用和冲突记忆，保留旧记录但默认从召回中排除。" : "Review superseded, deprecated, and contradicted memories. Old records remain preserved and traceable."} icon={ShieldCheck}>
            <div className="route-list">
              <label>
                <span className="field-label">{locale === "zh" ? "替代目标 memory id" : "Successor memory id"}</span>
                <input className="soft-input compact-input" value={successorId} onChange={(event) => setSuccessorId(event.target.value)} placeholder="cap_..." />
              </label>
              {conflictRecords.length ? conflictRecords.map((record) => (
                <div className="route-item soft-inset" key={record.id}>
                  <div className="route-row"><strong>{record.title}</strong><span className="badge soft-small warn">{record.state}</span></div>
                  <p>{record.why || record.content}</p>
                  {record.source_ids?.length ? <p>sources: {record.source_ids.join(", ")}</p> : null}
                  <div className="editor-actions">
                    <SoftButton icon={Archive} variant="secondary" onClick={() => markMemory(record, "deprecated")}>Mark Deprecated</SoftButton>
                    <SoftButton icon={ShieldCheck} variant="secondary" onClick={() => markMemory(record, "superseded")}>Mark Superseded</SoftButton>
                    <SoftButton icon={FilePenLine} variant="secondary" onClick={() => memoryAction(record, "mark-contradicted")}>Mark Contradicted</SoftButton>
                    <SoftButton icon={CheckCircle2} variant="secondary" onClick={() => memoryAction(record, "reactivate")}>Reactivate</SoftButton>
                    <SoftButton icon={Archive} variant="secondary" onClick={() => memoryAction(record, "archive")}>Archive</SoftButton>
                  </div>
                </div>
              )) : <p className="panel-copy">{locale === "zh" ? "当前没有需要审查的替代或冲突记忆。" : "No supersession or conflict records need review."}</p>}
            </div>
          </SoftPanel>
        </section>

        <section className="section" id="settings">
          <SoftPanel title={t.panels.settings} description={t.panels.settingsDescription} icon={Settings}>
            <div className="settings-grid">
              <label><span className="field-label">Provider</span><input className="soft-input compact-input" value={String(settings.llm_config?.provider || "auto")} onChange={(event) => setSetting("llm_config", "provider", event.target.value)} /></label>
              <label><span className="field-label">API base URL</span><input className="soft-input compact-input" value={String(settings.llm_config?.api_base_url || "")} onChange={(event) => setSetting("llm_config", "api_base_url", event.target.value)} /></label>
              <label><span className="field-label">Model</span><input className="soft-input compact-input" value={String(settings.llm_config?.model || "")} onChange={(event) => setSetting("llm_config", "model", event.target.value)} /></label>
              <label><span className="field-label">API key</span><input className="soft-input compact-input" type="password" value={String(settings.llm_config?.api_key || "")} onChange={(event) => setSetting("llm_config", "api_key", event.target.value)} /></label>
              <label><span className="field-label">Mode</span><select className="soft-input compact-input" value={String(settings.compaction_config?.mode || "deterministic")} onChange={(event) => setSetting("compaction_config", "mode", event.target.value)}><option value="deterministic">deterministic</option><option value="local_semantic">local_semantic</option><option value="hybrid">hybrid</option></select></label>
              <label><span className="field-label">Max input memories</span><input className="soft-input compact-input" type="number" value={Number(settings.compaction_config?.max_input_memories || 24)} onChange={(event) => setSetting("compaction_config", "max_input_memories", Number(event.target.value))} /></label>
              <label><span className="field-label">Near duplicate threshold</span><input className="soft-input compact-input" type="number" min="0.1" max="1" step="0.01" value={Number(settings.compaction_config?.near_duplicate_threshold || 0.9)} onChange={(event) => setSetting("compaction_config", "near_duplicate_threshold", Number(event.target.value))} /></label>
              <label><span className="field-label">Minimum cluster size</span><input className="soft-input compact-input" type="number" value={Number(settings.compaction_config?.min_cluster_size || 3)} onChange={(event) => setSetting("compaction_config", "min_cluster_size", Number(event.target.value))} /></label>
              <label><span className="field-label">Max output tokens</span><input className="soft-input compact-input" type="number" value={Number(settings.compaction_config?.max_output_tokens || 700)} onChange={(event) => setSetting("compaction_config", "max_output_tokens", Number(event.target.value))} /></label>
              <label><span className="field-label">Timeout seconds</span><input className="soft-input compact-input" type="number" value={Number(settings.compaction_config?.timeout_seconds || 45)} onChange={(event) => setSetting("compaction_config", "timeout_seconds", Number(event.target.value))} /></label>
              <label><span className="field-label">Token budget</span><input className="soft-input compact-input" type="number" value={Number(settings.scoring_config?.token_budget || 2000)} onChange={(event) => setSetting("scoring_config", "token_budget", Number(event.target.value))} /></label>
              <label><span className="field-label">Recency half-life days</span><input className="soft-input compact-input" type="number" value={Number(settings.scoring_config?.recency_half_life_days || 30)} onChange={(event) => setSetting("scoring_config", "recency_half_life_days", Number(event.target.value))} /></label>
              <label><span className="field-label">Importance weight</span><input className="soft-input compact-input" type="number" min="0" max="1" step="0.01" value={Number(settings.scoring_config?.importance_weight || 0.18)} onChange={(event) => setSetting("scoring_config", "importance_weight", Number(event.target.value))} /></label>
              <label><span className="field-label">Redundancy penalty</span><input className="soft-input compact-input" type="number" min="0" max="1" step="0.01" value={Number(settings.scoring_config?.redundancy_weight || 0.2)} onChange={(event) => setSetting("scoring_config", "redundancy_weight", Number(event.target.value))} /></label>
            </div>
            <div className="route-list two-column settings-toggles">
              {[
                ["llm_config", "external_llm_enabled", t.settings.external],
                ["compaction_config", "external_llm_compaction_enabled", t.settings.externalCompaction],
                ["llm_config", "local_only", t.settings.localOnly],
                ["llm_config", "allow_raw_memory_external", t.settings.raw],
                ["llm_config", "require_source_ids", t.settings.sourceIds],
                ["llm_config", "require_verifier", t.settings.verifier],
                ["privacy_config", "require_external_call_confirmation", "Confirm external calls"],
                ["privacy_config", "show_external_call_preview", "Show external call preview"],
                ["compaction_config", "near_duplicate_enabled", "Near duplicate clustering"],
                ["compaction_config", "auto_commit", "Auto-commit compaction"]
              ].map(([group, key, label]) => (
                <button className="route-item soft-inset" type="button" key={key} onClick={() => setSetting(group as "llm_config" | "privacy_config" | "compaction_config", key, !(settings[group as keyof SettingsPayload] as Record<string, unknown> | undefined)?.[key])}>
                  <div className="route-row"><strong>{label}</strong><span className={`badge soft-small ${(settings[group as keyof SettingsPayload] as Record<string, unknown> | undefined)?.[key] ? "good" : "warn"}`}>{(settings[group as keyof SettingsPayload] as Record<string, unknown> | undefined)?.[key] ? "on" : "off"}</span></div>
                </button>
              ))}
            </div>
            <div className="editor-actions"><SoftButton icon={CheckCircle2} variant="primary" onClick={saveSettings}>{t.actions.saveSettings}</SoftButton></div>
          </SoftPanel>
        </section>
      </div>
    </main>
  );
}
