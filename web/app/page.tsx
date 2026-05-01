"use client";

import { useMemo, useRef, useState } from "react";
import { ArrowRight, Calculator, CheckCircle2, DatabaseZap, Search, Send, Sparkles } from "lucide-react";
import { ArtifactCard } from "@/components/artifact-card";
import { MemoryEditor } from "@/components/memory-editor";
import { MetricCard } from "@/components/metric-card";
import { RouteCard } from "@/components/route-card";
import { SoftButton } from "@/components/soft-button";
import { SoftPanel } from "@/components/soft-panel";
import { TimelineItem } from "@/components/timeline-item";
import { TokenTrendChart } from "@/components/token-trend-chart";
import { TopNav } from "@/components/top-nav";
import { getDashboardData, type Locale } from "@/lib/dashboard-data";

const copy = {
  en: {
    actions: {
      buildPack: "Build prompt pack",
      openRecords: "Open records",
      retrieve: "Retrieve context",
      searchMemory: "Search memory"
    },
    calculation: {
      description:
        "Saved tokens are estimated per session as max(0, original context tokens - recalled memory pack tokens - retrieval overhead tokens). Window totals sum that value for sessions in the selected time range.",
      example: "Example: 9,200 original tokens - 2,150 memory pack tokens - 320 retrieval overhead = 6,730 saved tokens.",
      title: "How token savings are calculated"
    },
    controls: {
      autoDetail: "Recurring project background is compressed into reusable memory records.",
      autoOff: "off",
      autoOn: "on",
      autoTitle: "Auto-summarize repeated context",
      editableDetail: "New memory writes can be inspected and corrected before they become long-lived context.",
      editableOff: "direct",
      editableOn: "review",
      editableTitle: "Require editable writeback",
      title: "Memory controls",
      description: "Every control changes visible page state now; future API wiring can reuse the same handlers."
    },
    hero: {
      eyebrow: "Memory reuse and token economy",
      lead:
        "DysonSpherain keeps reusable project memory close to the work surface. Track saved tokens, recall compact context, and edit existing memories before they become stale.",
      title: "Spend fewer tokens repeating context."
    },
    labels: {
      memoryQuery: "Memory query",
      savedTokenMetrics: "Saved token metrics",
      tokenEconomyState: "Token economy state",
      live: "live"
    },
    memoryEditor: {
      content: "Memory content",
      empty: "No memory records found.",
      saved: "saved just now",
      saveMemory: "Save memory",
      searchAria: "Memory records",
      searchPlaceholder: "Search memories",
      title: "Title"
    },
    nav: {
      brandSubtitle: "Memory Console",
      languageLabel: "Switch language",
      links: [
        { href: "#tokens", label: "Tokens" },
        { href: "#memories", label: "Memories" },
        { href: "#retrieval", label: "Retrieval" },
        { href: "#runtime", label: "Runtime" }
      ],
      toggleNavigation: "Toggle navigation"
    },
    panels: {
      memoryDescription: "Review and edit existing memories directly from the console.",
      memoryTitle: "Memory records",
      recallDescription: "Ask for a compact evidence pack before starting a long task.",
      recallTitle: "Recall workspace",
      reuseDescription: "Recent sessions are drawing from saved memory instead of re-sending long background blocks.",
      reuseMetric: "Context reuse rate",
      reuseText: "Most repeated project background is now supplied by compact memory records.",
      reuseTitle: "Reuse posture",
      retrievalDescription: "Memory value depends on reusable, compact, and editable evidence.",
      retrievalTitle: "Retrieval health",
      timelineDescription: "The console keeps token saving, recall, editing, and measurement in one loop.",
      timelineTitle: "Runtime timeline",
      trendDescription: "A compact 7-day view of memory-assisted context reduction.",
      trendTitle: "Token savings trend",
      workflowDescription: "Each card is a reusable action surface for the next API-backed version.",
      workflowTitle: "Memory workflow"
    },
    query: {
      generatedPack:
        "Prompt pack ready: reused 4 memory records, compressed repeated background, and reserved the remaining context for new task-specific evidence.",
      placeholder: "Ask for prior decisions, implementation notes, project constraints, or a compact context pack...",
      retrieved: "Retrieved 4 memory records and estimated 6.7K saved tokens for this prompt.",
      seed: "token savings dashboard requirement"
    },
    chart: {
      aria: "Seven day saved-token trend chart",
      peak: "peak",
      subtitle: "7-day memory reuse signal",
      title: "Saved-token trend"
    }
  },
  zh: {
    actions: {
      buildPack: "生成提示词包",
      openRecords: "打开记忆记录",
      retrieve: "召回上下文",
      searchMemory: "搜索记忆"
    },
    calculation: {
      description: "节省 token 按每次会话估算：max(0, 原始上下文 token - 召回记忆包 token - 检索开销 token)。各时间窗口统计该范围内所有会话的总和。",
      example: "示例：9,200 原始 token - 2,150 记忆包 token - 320 检索开销 = 6,730 节省 token。",
      title: "Token 节省计算逻辑"
    },
    controls: {
      autoDetail: "重复项目背景会被压缩成可复用记忆记录。",
      autoOff: "关闭",
      autoOn: "开启",
      autoTitle: "自动摘要重复上下文",
      editableDetail: "新写入的记忆可先检查和修正，再成为长期上下文。",
      editableOff: "直接",
      editableOn: "审阅",
      editableTitle: "要求可编辑写回",
      title: "记忆控制",
      description: "每个控制都会改变当前页面状态；未来接入 API 时可复用相同处理逻辑。"
    },
    hero: {
      eyebrow: "记忆复用与 Token 经济",
      lead: "DysonSpherain 将可复用项目记忆放在工作表面附近。你可以跟踪节省 token、召回紧凑上下文，并在记忆过期前编辑它们。",
      title: "少花 token 重复背景。"
    },
    labels: {
      memoryQuery: "记忆查询",
      savedTokenMetrics: "节省 token 指标",
      tokenEconomyState: "Token 经济状态",
      live: "实时"
    },
    memoryEditor: {
      content: "记忆内容",
      empty: "没有找到记忆记录。",
      saved: "刚刚保存",
      saveMemory: "保存记忆",
      searchAria: "记忆记录",
      searchPlaceholder: "搜索记忆",
      title: "标题"
    },
    nav: {
      brandSubtitle: "记忆控制台",
      languageLabel: "切换语言",
      links: [
        { href: "#tokens", label: "Token" },
        { href: "#memories", label: "记忆" },
        { href: "#retrieval", label: "召回" },
        { href: "#runtime", label: "运行" }
      ],
      toggleNavigation: "展开导航"
    },
    panels: {
      memoryDescription: "直接在控制台中查看并编辑已有记忆。",
      memoryTitle: "记忆记录",
      recallDescription: "开始长任务前，先请求一个紧凑证据包。",
      recallTitle: "召回工作区",
      reuseDescription: "最近会话正在使用已保存记忆，而不是重新发送长背景块。",
      reuseMetric: "上下文复用率",
      reuseText: "大部分重复项目背景现在由紧凑记忆记录提供。",
      reuseTitle: "复用状态",
      retrievalDescription: "记忆价值取决于可复用、紧凑且可编辑的证据。",
      retrievalTitle: "召回健康度",
      timelineDescription: "控制台将 token 节省、召回、编辑和度量放在同一循环中。",
      timelineTitle: "运行时间线",
      trendDescription: "记忆辅助上下文压缩的 7 天紧凑视图。",
      trendTitle: "Token 节省趋势",
      workflowDescription: "每张卡片都是下一版 API 支持的可复用操作表面。",
      workflowTitle: "记忆工作流"
    },
    query: {
      generatedPack: "提示词包已生成：复用 4 条记忆记录，压缩重复背景，并为新任务证据预留上下文空间。",
      placeholder: "询问历史决策、实现笔记、项目约束，或请求紧凑上下文包...",
      retrieved: "已召回 4 条记忆记录，并为该提示词估算节省 6.7K token。",
      seed: "token 节省仪表盘需求"
    },
    chart: {
      aria: "七天节省 token 趋势图",
      peak: "峰值",
      subtitle: "7 天记忆复用信号",
      title: "节省 token 趋势"
    }
  }
};

export default function Home() {
  const [locale, setLocale] = useState<Locale>("en");
  const [query, setQuery] = useState("");
  const [retrievalResult, setRetrievalResult] = useState("");
  const [autoSummarize, setAutoSummarize] = useState(true);
  const [editableWriteback, setEditableWriteback] = useState(true);
  const recallRef = useRef<HTMLTextAreaElement>(null);

  const t = copy[locale];
  const data = useMemo(() => getDashboardData(locale), [locale]);

  function scrollTo(id: string) {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function searchMemory() {
    scrollTo("retrieval");
    setQuery(t.query.seed);
    window.setTimeout(() => recallRef.current?.focus(), 350);
  }

  function openRecords() {
    scrollTo("memories");
  }

  function retrieveContext() {
    setRetrievalResult(t.query.retrieved);
  }

  function buildPromptPack() {
    setRetrievalResult(t.query.generatedPack);
  }

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
            <h1 className="headline font-display" id="page-title">
              {t.hero.title}
            </h1>
            <p className="lead">{t.hero.lead}</p>
            <div className="action-row">
              <SoftButton icon={Search} variant="primary" onClick={searchMemory}>
                {t.actions.searchMemory}
              </SoftButton>
              <SoftButton icon={DatabaseZap} variant="secondary" onClick={openRecords}>
                {t.actions.openRecords}
              </SoftButton>
            </div>
          </div>

          <aside className="status-stack" aria-label={t.labels.tokenEconomyState}>
            <div className="depth-orb soft-raised">
              <div className="orb-core soft-raised">
                <Sparkles size={32} strokeWidth={2.1} aria-hidden="true" />
              </div>
            </div>
            <SoftPanel title={t.panels.reuseTitle} description={t.panels.reuseDescription} icon={CheckCircle2}>
              <div className="route-list">
                <div className="route-item soft-inset">
                  <div className="route-row">
                    <strong>{t.panels.reuseMetric}</strong>
                    <span className="badge soft-small good">72%</span>
                  </div>
                  <p>{t.panels.reuseText}</p>
                </div>
              </div>
            </SoftPanel>
          </aside>
        </section>

        <section className="metric-grid section" id="tokens" aria-label={t.labels.savedTokenMetrics}>
          {data.tokenSavings.map((metric) => (
            <MetricCard key={metric.label} {...metric} liveLabel={t.labels.live} />
          ))}
        </section>

        <section className="workspace-grid section">
          <SoftPanel title={t.panels.trendTitle} description={t.panels.trendDescription} icon={Sparkles}>
            <TokenTrendChart ariaLabel={t.chart.aria} data={data.tokenTrend} locale={locale} peakLabel={t.chart.peak} subtitle={t.chart.subtitle} title={t.chart.title} />
          </SoftPanel>

          <SoftPanel title={t.calculation.title} description={t.calculation.description} icon={Calculator}>
            <div className="calculation-box soft-inset">
              <p>{t.calculation.example}</p>
              <code>saved = max(0, original_context - memory_pack - retrieval_overhead)</code>
            </div>
          </SoftPanel>
        </section>

        <section className="workspace-grid section">
          <SoftPanel title={t.panels.retrievalTitle} description={t.panels.retrievalDescription} icon={CheckCircle2}>
            <div className="route-list">
              {data.retrievalHealth.map((item) => (
                <RouteCard key={item.name} {...item} />
              ))}
            </div>
          </SoftPanel>

          <SoftPanel title={t.panels.workflowTitle} description={t.panels.workflowDescription} icon={Sparkles}>
            <div className="artifact-grid compact">
              {data.workflowCards.map((card) => (
                <ArtifactCard key={card.title} {...card} />
              ))}
            </div>
          </SoftPanel>
        </section>

        <section className="section" id="memories" aria-labelledby="memory-title">
          <SoftPanel title={t.panels.memoryTitle} description={t.panels.memoryDescription} icon={DatabaseZap}>
            <MemoryEditor labels={t.memoryEditor} records={data.memoryRecords} />
          </SoftPanel>
        </section>

        <section className="workspace-grid section" id="retrieval">
          <SoftPanel title={t.panels.recallTitle} description={t.panels.recallDescription} icon={Search}>
            <form className="query-box" onSubmit={(event) => event.preventDefault()}>
              <label>
                <span className="sr-only">{t.labels.memoryQuery}</span>
                <textarea ref={recallRef} className="soft-input textarea" value={query} onChange={(event) => setQuery(event.target.value)} placeholder={t.query.placeholder} />
              </label>
              <div className="action-row">
                <SoftButton icon={Send} variant="primary" onClick={retrieveContext}>
                  {t.actions.retrieve}
                </SoftButton>
                <SoftButton icon={ArrowRight} variant="secondary" onClick={buildPromptPack}>
                  {t.actions.buildPack}
                </SoftButton>
              </div>
              {retrievalResult ? <div className="result-box soft-inset">{retrievalResult}</div> : null}
            </form>
          </SoftPanel>

          <SoftPanel title={t.panels.timelineTitle} description={t.panels.timelineDescription} icon={CheckCircle2}>
            <div className="timeline-list">
              {data.timeline.map((item) => (
                <TimelineItem key={item.title} {...item} />
              ))}
            </div>
          </SoftPanel>
        </section>

        <section className="section" id="runtime">
          <SoftPanel title={t.controls.title} description={t.controls.description} icon={DatabaseZap}>
            <div className="route-list two-column">
              <button className="route-item soft-inset" type="button" onClick={() => setAutoSummarize((value) => !value)}>
                <div className="route-row">
                  <strong>{t.controls.autoTitle}</strong>
                  <span className={`badge soft-small ${autoSummarize ? "good" : "warn"}`}>{autoSummarize ? t.controls.autoOn : t.controls.autoOff}</span>
                </div>
                <p>{t.controls.autoDetail}</p>
              </button>
              <button className="route-item soft-inset" type="button" onClick={() => setEditableWriteback((value) => !value)}>
                <div className="route-row">
                  <strong>{t.controls.editableTitle}</strong>
                  <span className={`badge soft-small ${editableWriteback ? "good" : "warn"}`}>{editableWriteback ? t.controls.editableOn : t.controls.editableOff}</span>
                </div>
                <p>{t.controls.editableDetail}</p>
              </button>
            </div>
          </SoftPanel>
        </section>
      </div>
    </main>
  );
}
