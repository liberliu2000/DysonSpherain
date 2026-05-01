import {
  Archive,
  BrainCircuit,
  Clock3,
  DatabaseZap,
  FilePenLine,
  Layers3,
  Route,
  Sparkles,
  TrendingUp,
  WalletCards
} from "lucide-react";

export type Locale = "en" | "zh";

const en = {
  tokenSavings: [
    { label: "Past hour", value: "18.4K", trend: "+12% vs previous hour", icon: Clock3 },
    { label: "Past 24 hours", value: "412K", trend: "+8.7% vs previous day", icon: TrendingUp },
    { label: "Past 7 days", value: "2.86M", trend: "7-day average 408K/day", icon: Archive },
    { label: "Total saved", value: "19.7M", trend: "All recorded memory-assisted sessions", icon: WalletCards }
  ],
  tokenTrend: [
    { label: "Mon", saved: 310000 },
    { label: "Tue", saved: 342000 },
    { label: "Wed", saved: 388000 },
    { label: "Thu", saved: 361000 },
    { label: "Fri", saved: 427000 },
    { label: "Sat", saved: 494000 },
    { label: "Sun", saved: 538000 }
  ],
  memoryRecords: [
    {
      id: "mem-1042",
      title: "Route-conditioned admission policy",
      scope: "project",
      updatedAt: "12 min ago",
      tags: ["retrieval", "routing"],
      content:
        "Use route-conditioned candidate admission so dense anchors remain protected while lexical, temporal, profile, and parent/session routes add evidence only when the query requires them."
    },
    {
      id: "mem-1037",
      title: "Token economy dashboard requirement",
      scope: "web-ui",
      updatedAt: "38 min ago",
      tags: ["tokens", "ui"],
      content:
        "The Web UI should foreground token savings for the past hour, past 24 hours, past 7 days, and lifetime total. It should also show a trend chart and allow memory review/editing."
    },
    {
      id: "mem-1019",
      title: "Artifact-backed report rule",
      scope: "reports",
      updatedAt: "Yesterday",
      tags: ["artifacts", "reports"],
      content:
        "Reports and tables should be generated from existing artifacts instead of hand-filled numbers. Missing fields should stay explicit rather than inferred."
    },
    {
      id: "mem-0998",
      title: "Local-first storage posture",
      scope: "runtime",
      updatedAt: "2 days ago",
      tags: ["sqlite", "chroma"],
      content:
        "Keep SQLite metadata and Chroma or JSON vector storage inspectable locally. Export, backup, and forget flows should remain available from the CLI and future API."
    }
  ],
  retrievalHealth: [
    {
      name: "Context reuse",
      status: "Active",
      detail: "Repeated project context is pulled from memory instead of being pasted into every prompt.",
      progress: 88
    },
    {
      name: "Compression guard",
      status: "Balanced",
      detail: "Evidence summaries are shortened before context assembly while preserving citations.",
      progress: 74
    },
    {
      name: "Writeback hygiene",
      status: "Clean",
      detail: "Duplicate notes are collapsed and edited records keep stable identifiers.",
      progress: 81
    }
  ],
  workflowCards: [
    {
      title: "Review memories",
      description: "Search, inspect, and edit project memory records from the console without leaving the workflow.",
      icon: FilePenLine,
      badge: "editable"
    },
    {
      title: "Save context",
      description: "Reuse prior decisions and summaries so long-running work spends fewer tokens on repeated background.",
      icon: BrainCircuit,
      badge: "tracked"
    },
    {
      title: "Local evidence",
      description: "SQLite metadata and vector-backed evidence stay visible as local operational surfaces.",
      icon: DatabaseZap,
      badge: "local"
    }
  ],
  timeline: [
    { title: "Recall", detail: "Find relevant memory records and summaries before a new task expands context.", icon: Route },
    { title: "Compress", detail: "Use concise evidence packs rather than repeating full logs or historical chat turns.", icon: Layers3 },
    { title: "Edit", detail: "Keep memory records current when a decision changes or stale phrasing appears.", icon: FilePenLine },
    { title: "Measure", detail: "Track saved tokens across short and long windows to quantify memory value.", icon: Sparkles }
  ]
};

const zh: typeof en = {
  tokenSavings: [
    { label: "近 1 小时", value: "18.4K", trend: "较上一小时 +12%", icon: Clock3 },
    { label: "近 24 小时", value: "412K", trend: "较前一日 +8.7%", icon: TrendingUp },
    { label: "近 7 天", value: "2.86M", trend: "7 日均值 408K/天", icon: Archive },
    { label: "累计节省", value: "19.7M", trend: "所有已记录的记忆辅助会话", icon: WalletCards }
  ],
  tokenTrend: [
    { label: "一", saved: 310000 },
    { label: "二", saved: 342000 },
    { label: "三", saved: 388000 },
    { label: "四", saved: 361000 },
    { label: "五", saved: 427000 },
    { label: "六", saved: 494000 },
    { label: "日", saved: 538000 }
  ],
  memoryRecords: [
    {
      ...en.memoryRecords[0],
      title: "路由条件候选接纳策略",
      scope: "项目",
      updatedAt: "12 分钟前",
      tags: ["检索", "路由"],
      content: "使用路由条件候选接纳策略，让 dense anchor 保持受保护；只有查询确实需要时，才引入 lexical、temporal、profile 和 parent/session 证据。"
    },
    {
      ...en.memoryRecords[1],
      title: "Token 经济仪表盘需求",
      scope: "Web UI",
      updatedAt: "38 分钟前",
      tags: ["token", "界面"],
      content: "Web UI 应优先展示近 1 小时、近 24 小时、近 7 天和累计节省 token，并显示趋势图，同时允许查看和编辑已有记忆。"
    },
    {
      ...en.memoryRecords[2],
      title: "报告数据来源规则",
      scope: "报告",
      updatedAt: "昨天",
      tags: ["artifact", "报告"],
      content: "报告和表格应由现有 artifact 生成，而不是手工填写数字。缺失字段应明确保留，不应反向推断。"
    },
    {
      ...en.memoryRecords[3],
      title: "本地优先存储姿态",
      scope: "运行时",
      updatedAt: "2 天前",
      tags: ["sqlite", "chroma"],
      content: "保持 SQLite 元数据和 Chroma/JSON 向量存储本地可检查。导出、备份和遗忘流程应继续由 CLI 和未来 API 支持。"
    }
  ],
  retrievalHealth: [
    { name: "上下文复用", status: "已启用", detail: "重复项目背景由记忆提供，而不是每次重新粘贴到提示词中。", progress: 88 },
    { name: "压缩护栏", status: "平衡", detail: "证据摘要在组装上下文前被压缩，同时保留引用线索。", progress: 74 },
    { name: "写回清洁度", status: "良好", detail: "重复记忆会被折叠，编辑后的记录保持稳定 ID。", progress: 81 }
  ],
  workflowCards: [
    { title: "查看记忆", description: "在控制台中搜索、检查并编辑项目记忆，不需要离开当前工作流。", icon: FilePenLine, badge: "可编辑" },
    { title: "节省上下文", description: "复用历史决策和摘要，让长任务少花 token 重复背景信息。", icon: BrainCircuit, badge: "已跟踪" },
    { title: "本地证据", description: "SQLite 元数据和向量证据作为本地运行表面保持可见。", icon: DatabaseZap, badge: "本地" }
  ],
  timeline: [
    { title: "召回", detail: "在新任务展开上下文之前，找到相关记忆记录和摘要。", icon: Route },
    { title: "压缩", detail: "使用简洁证据包，而不是反复发送完整日志或历史对话。", icon: Layers3 },
    { title: "编辑", detail: "当决策变化或表述过期时，及时更新记忆记录。", icon: FilePenLine },
    { title: "度量", detail: "按短期和长期窗口跟踪节省 token，量化记忆价值。", icon: Sparkles }
  ]
};

export function getDashboardData(locale: Locale) {
  return locale === "zh" ? zh : en;
}
