from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dysonspherain.memory_os.observation_store import get_observations, resume_context, search_observations, timeline, token_economy_summary
from dysonspherain.memory_runtime.config import load_runtime_config, save_runtime_config
from dysonspherain.memory_runtime.ledger import replay_events
from dysonspherain.memory_runtime.runtime import cockpit_snapshot, graph_state
from dysonspherain.memory_runtime.scheduler import enqueue_maintenance_jobs, load_pending_jobs, run_scheduler_once
from dysonspherain.product import (
    apply_maintenance_suggestion,
    configure_embedding_backend,
    configure_encryption,
    configure_product_vector_backend,
    create_context_pack,
    dismiss_maintenance_suggestion,
    doctor as product_doctor,
    get_capsule,
    get_context_pack,
    get_retrieval_trace,
    list_benchmark_runs,
    list_capsules,
    list_projects,
    maintenance_suggestions,
    privacy_policy,
    product_embedding_backends,
    product_vector_backends,
    rebuild_product_embeddings,
    rebuild_product_vector_index,
    remember,
    retrieve,
    update_capsule,
)


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DysonSpherain Memory</title>
  <style>
    :root { color-scheme: light; --bg: #f6f7f9; --panel: #ffffff; --line: #e2e6ee; --text: #10151f; --muted: #667085; --accent: #0f766e; --warn: #b45309; --danger: #b42318; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--text); background: var(--bg); letter-spacing: 0; }
    header { min-height: 64px; padding: 12px 24px; border-bottom: 1px solid var(--line); background: rgba(255,255,255,.94); display: flex; align-items: center; gap: 14px; position: sticky; top: 0; z-index: 2; backdrop-filter: blur(10px); }
    header strong { font-size: 15px; font-weight: 650; white-space: nowrap; }
    input, select { width: 100%; min-width: 120px; padding: 9px 11px; border: 1px solid var(--line); border-radius: 7px; background: #fff; color: var(--text); font-size: 14px; }
    textarea { width: 100%; min-height: 110px; padding: 9px 11px; border: 1px solid var(--line); border-radius: 7px; background: #fff; color: var(--text); font-size: 13px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    button { padding: 9px 12px; border: 1px solid #111827; border-radius: 7px; background: #111827; color: #fff; font-size: 14px; cursor: pointer; transition: transform .16s ease, background .16s ease; }
    button:hover { transform: translateY(-1px); background: #1f2937; }
    main { padding: 20px 24px; display: grid; grid-template-columns: 210px minmax(0, 1fr); gap: 18px; }
    nav { position: sticky; top: 84px; align-self: start; display: grid; gap: 7px; }
    nav button { width: 100%; text-align: left; background: #fff; color: var(--text); border-color: var(--line); }
    nav button.active { background: #10151f; color: #fff; border-color: #10151f; }
    h2 { margin: 0 0 10px; font-size: 13px; font-weight: 650; color: var(--muted); text-transform: uppercase; }
    h3 { margin: 0 0 8px; font-size: 15px; }
    .view { display: none; min-width: 0; }
    .view.active { display: block; animation: fade .18s ease; }
    @keyframes fade { from { opacity: .4; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
    .summary { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px; }
    .metric, .panel, .item, pre { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
    .metric { padding: 16px; min-height: 104px; }
    .label { color: var(--muted); font-size: 12px; }
    .value { margin-top: 8px; font-size: 28px; font-weight: 680; line-height: 1; }
    .sub { margin-top: 8px; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    .panel { padding: 14px; min-width: 0; margin-bottom: 14px; }
    .item { padding: 12px; margin-bottom: 8px; cursor: pointer; }
    .item:hover { border-color: #b9c0cc; }
    .item strong { display: block; font-size: 14px; margin-bottom: 6px; overflow-wrap: anywhere; }
    .item small { color: var(--muted); display: block; font-size: 12px; overflow-wrap: anywhere; }
    .item div { margin-top: 8px; font-size: 13px; color: #374151; overflow-wrap: anywhere; }
    .events { width: 100%; border-collapse: collapse; font-size: 13px; }
    .events th, .events td { padding: 9px 7px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
    .events th { color: var(--muted); font-size: 12px; font-weight: 600; }
    .events tr { cursor: pointer; }
    .events tr:hover td { background: #fafafa; }
    .events td { overflow-wrap: anywhere; }
    pre { margin: 12px 0 0; padding: 14px; white-space: pre-wrap; overflow: auto; max-height: 420px; font-size: 12px; line-height: 1.45; }
    .pill { display: inline-block; padding: 3px 8px; border-radius: 999px; border: 1px solid var(--line); color: var(--muted); font-size: 12px; margin: 0 5px 5px 0; }
    .ok { color: var(--accent); }
    .warn { color: var(--warn); }
    .danger { color: var(--danger); }
    .row { display: grid; grid-template-columns: 180px minmax(0, 1fr); gap: 10px; align-items: center; margin-bottom: 10px; }
    .graph-canvas { width: 100%; min-height: 420px; border: 1px solid var(--line); border-radius: 8px; background: #fbfcfd; }
    .graph-node { cursor: pointer; transition: opacity .15s ease; }
    .graph-node:hover { opacity: .72; }
    .graph-edge { stroke: #98a2b3; stroke-width: 1.4; }
    .graph-label { font-size: 11px; fill: #344054; pointer-events: none; }
    .toolbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; }
    .toolbar input { width: auto; flex: 1; min-width: 180px; }
    @media (max-width: 820px) {
      header { padding: 12px; height: auto; flex-wrap: wrap; }
      main { padding: 12px; grid-template-columns: 1fr; }
      nav { position: static; grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .summary { grid-template-columns: 1fr; }
      .grid { grid-template-columns: 1fr; }
      .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header><strong>DysonSpherain Memory</strong><input id="q" placeholder="Search observations"><button onclick="search()">Search</button></header>
  <main>
    <nav id="tabs"></nav>
    <section>
      <div id="mission" class="view active" data-title="Project Dashboard"></div>
      <div id="ledger" class="view"></div>
      <div id="graph" class="view"></div>
      <div id="router" class="view"></div>
      <div id="compiler" class="view"></div>
      <div id="evidenceSearch" class="view"></div>
      <div id="traceViewer" class="view"></div>
      <div id="evidenceTimeline" class="view"></div>
      <div id="evidenceGraph" class="view"></div>
      <div id="contextComposer" class="view"></div>
      <div id="benchmarkLab" class="view"></div>
      <div id="healthDoctor" class="view"></div>
      <div id="maintenance" class="view"></div>
      <div id="productSettings" class="view"></div>
      <div id="audit" class="view"></div>
      <div id="scheduler" class="view"></div>
      <div id="config" class="view"></div>
    </section>
  </main>
  <script>
    const fmt = new Intl.NumberFormat();
    const pct = (value) => `${((Number(value || 0)) * 100).toFixed(1)}%`;
    const pages = [
      ['mission', 'Mission Control'],
      ['ledger', 'Memory Ledger'],
      ['graph', 'Situation Graph'],
      ['router', 'Evidence Router'],
      ['compiler', 'Context Compiler'],
      ['evidenceSearch', 'Evidence Search'],
      ['traceViewer', 'Retrieval Trace Viewer'],
      ['evidenceTimeline', 'Evidence Timeline'],
      ['evidenceGraph', 'Evidence Field Graph'],
      ['contextComposer', 'Context Composer'],
      ['benchmarkLab', 'Benchmark Lab'],
      ['healthDoctor', 'Health Doctor'],
      ['maintenance', 'Maintenance'],
      ['productSettings', 'Settings'],
      ['audit', 'Recall Audit'],
      ['scheduler', 'Active Scheduler'],
      ['config', 'Configuration Studio']
    ];
    let snapshot = {};
    let graphFrame = -1;
    function esc(value) { return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;', "'":'&#39;'}[c])); }
    function activate(id) {
      document.querySelectorAll('.view').forEach(el => el.classList.toggle('active', el.id === id));
      document.querySelectorAll('nav button').forEach(el => el.classList.toggle('active', el.dataset.target === id));
    }
    function initTabs() {
      document.getElementById('tabs').innerHTML = pages.map(([id, label]) => `<button data-target="${id}" onclick="activate('${id}')">${label}</button>`).join('');
      document.querySelector('nav button').classList.add('active');
    }
    async function search() {
      const q = document.getElementById('q').value;
      const res = await fetch('/api/search?query=' + encodeURIComponent(q));
      const data = await res.json();
      document.getElementById('ledger').innerHTML = `<div class="panel"><h2>Observation Search</h2>${(data.observations || []).map(o =>
        `<div class="item" onclick="detail('${o.observation_id}')"><strong>${o.title}</strong><small>${o.kind} · ${o.token_cost} tokens · ${o.citation}</small><div>${o.snippet || ''}</div></div>`
      ).join('') || '<div class="sub">No observations.</div>'}</div><pre id="detail">Select an observation.</pre>`;
      activate('ledger');
    }
    async function tokenEconomy() {
      const res = await fetch('/api/token-economy');
      const data = await res.json();
      const windows = data.windows || {};
      return ['24h', '7d', '30d'].map(key => {
        const row = windows[key] || {};
        return `<div class="metric"><div class="label">${key} saved tokens</div><div class="value">${fmt.format(row.estimated_saved_tokens || 0)}</div><div class="sub">saving ratio ${pct(row.saving_ratio)} · ${fmt.format(row.event_count || 0)} conversations</div></div>`;
      }).join('');
    }
    function tokenTable(data) {
      const events = data.events || [];
      return `<table class="events"><thead><tr><th>Time</th><th>Saved</th><th>Ratio</th><th>Decision</th><th>Prompt</th></tr></thead><tbody>${events.map(e =>
        `<tr onclick="detail('${e.observation_id}')"><td>${(e.updated_at || '').slice(0, 16).replace('T', ' ')}</td><td>${fmt.format(e.estimated_saved_tokens || 0)}</td><td>${pct(e.saving_ratio)}</td><td>${e.decision || ''}</td><td>${e.prompt_preview || ''}</td></tr>`
      ).join('')}</tbody></table>`;
    }
    async function resume() {
      const res = await fetch('/api/resume-context');
      const data = await res.json();
      document.getElementById('resume').textContent = data.rendered_context || 'No previous session context.';
    }
    async function detail(id) {
      const res = await fetch('/api/observations/' + encodeURIComponent(id));
      document.getElementById('detail').textContent = JSON.stringify(await res.json(), null, 2);
    }
    function renderMission(tokenHtml, resumeText) {
      const m = snapshot.mission_control || {};
      document.getElementById('mission').innerHTML = `<div class="summary">
        <div class="metric"><div class="label">Events</div><div class="value">${fmt.format(m.event_count || 0)}</div><div class="sub">append-only ledger</div></div>
        <div class="metric"><div class="label">Graph Nodes</div><div class="value">${fmt.format(m.node_count || 0)}</div><div class="sub">${fmt.format(m.edge_count || 0)} edges</div></div>
        <div class="metric"><div class="label">Memory Health</div><div class="value ${m.memory_health === 'ok' ? 'ok' : 'warn'}">${esc(m.memory_health || 'empty')}</div><div class="sub">audit-aware state</div></div>
        <div class="metric"><div class="label">Index Freshness</div><div class="value ok">${esc(m.index_freshness || 'empty')}</div><div class="sub">projection replay status</div></div>
      </div><div class="grid"><div class="panel"><h2>Current Task State</h2>${cards(m.active_tasks)}</div><div class="panel"><h2>Active Constraints</h2>${cards(m.active_constraints)}</div></div><div class="panel"><h2>Resume Last Session</h2><pre>${esc(resumeText)}</pre></div><div class="summary">${tokenHtml}</div>`;
    }
    function cards(items) {
      return (items || []).map(item => `<div class="item"><strong>${esc(item.title || item.node_id || item.name || item.reason || item.candidate_id || item.section_type || 'record')}</strong><small>${esc(item.node_type || item.kind || item.status || item.updated_at || '')}</small><div>${esc(item.summary || item.content || item.message || item.reason || item.section_type || '')}</div></div>`).join('') || '<div class="sub">No records.</div>';
    }
    function renderRuntimePages(tokenData) {
      const latest = (snapshot.ledger || {}).latest_events || [];
      const graph = snapshot.graph || {};
      const packet = snapshot.packet || {};
      const audit = snapshot.audit || {};
      const config = snapshot.config || {};
      document.getElementById('ledger').innerHTML = `<div class="panel"><h2>Memory Ledger</h2>${cards(latest)}</div><div class="panel"><h2>Token Savings by Conversation</h2>${tokenTable(tokenData)}</div><pre id="detail">Select an observation.</pre>`;
      document.getElementById('graph').innerHTML = `<div class="summary">${Object.entries(((snapshot.mission_control || {}).node_counts || {})).map(([k,v]) => `<div class="metric"><div class="label">${esc(k)}</div><div class="value">${v}</div><div class="sub">situation nodes</div></div>`).join('')}</div><div class="panel"><h2>Situation Graph</h2>${renderTimelineControls()}<div id="graphCanvasHost">${renderGraphCanvas(graphForFrame())}</div></div><div class="grid"><div class="panel"><h2>Graph Nodes</h2>${cards(graph.nodes || [])}</div><div class="panel"><h2>Selected Node</h2><pre id="graphDetail">Select a node in the graph.</pre></div></div>`;
      const trace = (packet.compiler_trace || {}).router || {};
      document.getElementById('router').innerHTML = `<div class="panel"><h2>Evidence Router</h2><div class="pill">intent ${esc((packet.intent || {}).intent_type || 'none')}</div><div class="pill">merge ${esc(((trace.program || {}).merge_policy) || '')}</div><div class="pill">candidates ${fmt.format(trace.candidate_count || 0)}</div><pre>${esc(JSON.stringify(trace, null, 2))}</pre></div>`;
      document.getElementById('compiler').innerHTML = `<div class="summary"><div class="metric"><div class="label">Budget</div><div class="value">${fmt.format(packet.budget_tokens || 0)}</div></div><div class="metric"><div class="label">Used</div><div class="value">${fmt.format(packet.used_tokens || 0)}</div></div><div class="metric"><div class="label">Sections</div><div class="value">${fmt.format((packet.sections || []).length)}</div></div><div class="metric"><div class="label">Omitted</div><div class="value">${fmt.format((packet.omitted_candidates || []).length)}</div></div></div><div class="grid"><div class="panel"><h2>Selected Sections</h2>${cards(packet.sections || [])}</div><div class="panel"><h2>Omitted Candidates</h2>${cards(packet.omitted_candidates || [])}</div></div>`;
      document.getElementById('audit').innerHTML = `<div class="panel"><h2>Recall Audit</h2><div class="value ${audit.risk_level === 'low' ? 'ok' : 'warn'}">${esc(audit.risk_level || 'empty')}</div>${cards(audit.checks || [])}<pre>${esc(JSON.stringify(audit.suggested_followup_ops || [], null, 2))}</pre></div>`;
      document.getElementById('scheduler').innerHTML = `<div class="panel"><h2>Active Scheduler</h2><div class="toolbar"><button onclick="enqueueRefresh()">Queue Refresh</button><button onclick="runSchedulerOnce()">Run Once</button></div>${(config.scheduler_triggers || []).map(t => `<span class="pill">${esc(t)}</span>`).join('')}<pre id="schedulerState">${esc(JSON.stringify({cache_policy: config.cache_policy, audit_checks: config.audit_checks}, null, 2))}</pre></div>`;
      document.getElementById('config').innerHTML = `<div class="panel"><h2>Configuration Studio</h2>
        <div class="row"><label>Context Budget</label><input id="cfgBudget" value="${esc(config.context_budget || 1200)}"></div>
        <div class="row"><label>Embedding Backend</label><input id="cfgEmbedding" value="${esc(config.embedding_backend || '')}"></div>
        <div class="row"><label>Lexical Backend</label><input id="cfgLexical" value="${esc(config.lexical_backend || '')}"></div>
        <div class="row"><label>Projection Backend</label><input id="cfgProjection" value="${esc(config.projection_backend || '')}"></div>
        <div class="row"><label>Cache Policy</label><input id="cfgCache" value="${esc(config.cache_policy || '')}"></div>
        <div class="row"><label>Enabled Operators</label><input id="cfgOperators" value="${esc((config.enabled_operators || []).join(','))}"></div>
        <div class="row"><label>Scheduler Triggers</label><input id="cfgTriggers" value="${esc((config.scheduler_triggers || []).join(','))}"></div>
        <div class="row"><label>Operator Weights</label><textarea id="cfgWeights">${esc(JSON.stringify(config.operator_weights || {}, null, 2))}</textarea></div>
        <div class="row"><label>Section Limits</label><textarea id="cfgSections">${esc(JSON.stringify(config.section_limits || {}, null, 2))}</textarea></div>
        <div class="row"><label>Import / Export</label><textarea id="cfgImport">${esc(JSON.stringify(config, null, 2))}</textarea></div>
        <div class="row"><label>UI Animation</label><select id="cfgAnim"><option>low</option><option>medium</option><option>high</option></select></div>
        <div class="toolbar"><button onclick="saveConfig()">Save Configuration</button><button onclick="exportConfig()">Export</button><button onclick="importConfig()">Import</button></div><pre>${esc(JSON.stringify(config, null, 2))}</pre></div>`;
      const anim = document.getElementById('cfgAnim'); if (anim) anim.value = config.ui_animation_intensity || 'medium';
    }
    function renderGraphCanvas(graph) {
      const nodes = (graph.nodes || []).slice(-28);
      const edges = (graph.edges || []).filter(e => nodes.some(n => n.node_id === e.source_node_id) && nodes.some(n => n.node_id === e.target_node_id)).slice(-48);
      const w = 980, h = 420, cx = w / 2, cy = h / 2, r = Math.min(w, h) * .36;
      const pos = {};
      nodes.forEach((n, i) => {
        const angle = (Math.PI * 2 * i / Math.max(1, nodes.length)) - Math.PI / 2;
        pos[n.node_id] = {x: cx + Math.cos(angle) * r, y: cy + Math.sin(angle) * r};
      });
      const edgeSvg = edges.map(e => {
        const a = pos[e.source_node_id], b = pos[e.target_node_id];
        if (!a || !b) return '';
        const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
        return `<line class="graph-edge" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"><title>${esc(e.edge_type)}</title></line><text class="graph-label" x="${mx}" y="${my}">${esc(e.edge_type)}</text>`;
      }).join('');
      const nodeSvg = nodes.map((n) => {
        const p = pos[n.node_id];
        const color = n.node_type === 'Task' ? '#0f766e' : n.node_type === 'Regression' ? '#b42318' : n.node_type === 'Constraint' ? '#b45309' : '#344054';
        return `<g class="graph-node" onclick='showGraphNode(${JSON.stringify(JSON.stringify(n))})'><circle cx="${p.x}" cy="${p.y}" r="18" fill="${color}"><title>${esc(n.title)}</title></circle><text class="graph-label" x="${p.x + 22}" y="${p.y + 4}">${esc((n.title || n.node_id || '').slice(0, 28))}</text></g>`;
      }).join('');
      return `<svg class="graph-canvas" viewBox="0 0 ${w} ${h}" role="img" aria-label="Situation Graph">${edgeSvg}${nodeSvg}</svg>`;
    }
    function renderTimelineControls() {
      const events = ((snapshot.ledger || {}).latest_events || []);
      if (graphFrame < 0) graphFrame = Math.max(0, events.length - 1);
      return `<div class="toolbar"><button onclick="stepGraph(-1)">Prev</button><input id="graphTimeline" type="range" min="0" max="${Math.max(0, events.length - 1)}" value="${graphFrame}" oninput="setGraphFrame(Number(this.value))"><button onclick="stepGraph(1)">Next</button><span class="pill" id="graphFrameLabel">${events.length ? graphFrame + 1 : 0} / ${events.length}</span></div>`;
    }
    function graphForFrame() {
      const graph = snapshot.graph || {};
      const events = ((snapshot.ledger || {}).latest_events || []);
      if (!events.length || graphFrame >= events.length - 1) return graph;
      const allowed = new Set(events.slice(0, graphFrame + 1).map(e => e.event_id));
      const edges = (graph.edges || []).filter(e => (e.source_event_ids || []).some(id => allowed.has(id)));
      const nodeIds = new Set();
      edges.forEach(e => { nodeIds.add(e.source_node_id); nodeIds.add(e.target_node_id); });
      return {nodes: (graph.nodes || []).filter(n => nodeIds.has(n.node_id)), edges};
    }
    function setGraphFrame(value) {
      graphFrame = value;
      const host = document.getElementById('graphCanvasHost');
      if (host) host.innerHTML = renderGraphCanvas(graphForFrame());
      const events = ((snapshot.ledger || {}).latest_events || []);
      const label = document.getElementById('graphFrameLabel');
      if (label) label.textContent = `${events.length ? graphFrame + 1 : 0} / ${events.length}`;
    }
    function stepGraph(delta) {
      const events = ((snapshot.ledger || {}).latest_events || []);
      setGraphFrame(Math.max(0, Math.min(Math.max(0, events.length - 1), graphFrame + delta)));
      const slider = document.getElementById('graphTimeline');
      if (slider) slider.value = graphFrame;
    }
    function showGraphNode(raw) {
      const el = document.getElementById('graphDetail');
      if (el) el.textContent = JSON.stringify(JSON.parse(raw), null, 2);
    }
    async function productCapsules(limit = 50) {
      const res = await fetch('/api/capsules?limit=' + limit);
      return await res.json();
    }
    async function renderEvidenceSearch() {
      const data = await productCapsules(30);
      document.getElementById('evidenceSearch').innerHTML = `<div class="panel"><h2>Evidence Search</h2>
        <div class="toolbar"><input id="productQuery" placeholder="Search evidence capsules"><select id="productValidity"><option value="">active only</option><option value="true">include invalid</option></select><button onclick="runProductSearch()">Search</button><button onclick="exportSelectedEvidence()">Export</button></div>
        <div id="productResults">${productCards(data.capsules || [])}</div></div><pre id="productDetail">Select a capsule.</pre>`;
    }
    function productCards(items) {
      return (items || []).map(c => `<div class="item" onclick="productCapsuleDetail('${c.id}')"><strong>${esc(c.title || c.id)}</strong><small>${esc(c.evidence_type)} · ${esc(c.validity_state)} · ${esc((c.timestamp || '').slice(0, 19))}</small><div>${esc(c.summary || '')}</div><div>${(c.tags || []).map(t => `<span class="pill">${esc(t)}</span>`).join('')}</div></div>`).join('') || '<div class="sub">No capsules.</div>';
    }
    async function runProductSearch() {
      const q = document.getElementById('productQuery').value;
      const includeInvalid = document.getElementById('productValidity').value === 'true';
      const res = await fetch('/api/retrieve', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({query: q, show_audit: true, context_pack: true, include_invalid: includeInvalid})});
      const data = await res.json();
      document.getElementById('productResults').innerHTML = productCards((data.candidates || []).map(c => c.capsule));
      document.getElementById('productDetail').textContent = JSON.stringify(data.retrieval_trace || data, null, 2);
      window.latestTraceId = (data.retrieval_trace || {}).trace_id;
      renderTraceViewer(data);
    }
    async function productCapsuleDetail(id) {
      const res = await fetch('/api/capsules/' + encodeURIComponent(id));
      const data = await res.json();
      document.getElementById('productDetail').textContent = JSON.stringify(data, null, 2);
    }
    async function exportSelectedEvidence() {
      const data = await productCapsules(200);
      document.getElementById('productDetail').textContent = JSON.stringify({status: 'ok', export_preview: data.capsules || []}, null, 2);
    }
    function renderTraceViewer(data = null) {
      const trace = data ? (data.retrieval_trace || {}) : {};
      const probes = trace.probe_results || {};
      document.getElementById('traceViewer').innerHTML = `<div class="panel"><h2>Retrieval Trace Viewer</h2>
        <div class="toolbar"><input id="traceQuery" placeholder="Run retrieval trace"><button onclick="runTraceViewer()">Run</button></div>
        <div class="summary">${Object.entries(probes).map(([k, v]) => `<div class="metric"><div class="label">${esc(k)}</div><div class="value">${fmt.format(v.count || 0)}</div><div class="sub">${fmt.format(v.latency_ms || 0)} ms · ${esc(v.status || 'ok')}</div></div>`).join('') || '<div class="metric"><div class="label">Trace</div><div class="value">0</div><div class="sub">Run a query to inspect admission.</div></div>'}</div>
        <div class="grid"><div class="panel"><h2>Admission</h2>${cards(trace.final_candidates || [])}</div><div class="panel"><h2>Excluded Invalid Evidence</h2>${cards(trace.filtered_candidates || [])}</div></div><pre>${esc(JSON.stringify(trace, null, 2))}</pre></div>`;
    }
    async function runTraceViewer() {
      const q = document.getElementById('traceQuery').value;
      const res = await fetch('/api/retrieve', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({query: q, show_audit: true, context_pack: true, include_debug_trace: true})});
      renderTraceViewer(await res.json());
      activate('traceViewer');
    }
    async function renderEvidenceTimeline() {
      const data = await productCapsules(100);
      const items = (data.capsules || []).sort((a, b) => String(b.timestamp).localeCompare(String(a.timestamp)));
      document.getElementById('evidenceTimeline').innerHTML = `<div class="panel"><h2>Evidence Timeline</h2>${items.map(c => `<div class="item" onclick="productCapsuleDetail('${c.id}')"><strong>${esc((c.timestamp || '').slice(0, 19))} · ${esc(c.evidence_type)}</strong><small>${esc(c.source_type)} · ${esc(c.validity_state)}</small><div>${esc(c.title || c.summary || c.id)}</div></div>`).join('') || '<div class="sub">No timeline records.</div>'}</div><pre id="timelineDetail">Select a timeline item.</pre>`;
    }
    async function renderEvidenceGraph() {
      const data = await productCapsules(80);
      const nodes = data.capsules || [];
      const edges = [];
      nodes.forEach(c => ['supports', 'contradicts', 'supersedes', 'superseded_by', 'related_ids'].forEach(k => (c[k] || []).forEach(t => edges.push({source_node_id: c.id, target_node_id: t, edge_type: k}))));
      const graph = {nodes: nodes.map(c => ({node_id: c.id, title: c.title || c.id, node_type: c.evidence_type, status: c.validity_state})), edges};
      document.getElementById('evidenceGraph').innerHTML = `<div class="panel"><h2>Evidence Field Graph</h2>${renderGraphCanvas(graph)}</div><div class="grid"><div class="panel"><h2>Relations</h2>${cards(edges)}</div><div class="panel"><h2>Capsules</h2>${cards(graph.nodes)}</div></div>`;
    }
    async function renderContextComposer() {
      document.getElementById('contextComposer').innerHTML = `<div class="panel"><h2>Context Composer</h2>
        <div class="row"><label>Task</label><input id="ctxQuery" value=""></div>
        <div class="row"><label>Agent Role</label><select id="ctxRole"><option>coder</option><option>reviewer</option><option>researcher</option><option>benchmarker</option><option>writer</option><option>planner</option><option>debugger</option><option>custom</option></select></div>
        <div class="row"><label>Max Tokens</label><input id="ctxTokens" value="2000"></div>
        <div class="row"><label>Sections</label><input id="ctxSections" placeholder="Mission State,Supporting Evidence"></div>
        <div class="row"><label>Format</label><select id="ctxFormat"><option>markdown</option><option>json</option><option>yaml</option><option>text</option></select></div>
        <div class="toolbar"><button onclick="buildProductContext()">Build Context</button></div><pre id="ctxPreview">Context preview.</pre></div>`;
    }
    async function buildProductContext() {
      const sections = document.getElementById('ctxSections').value.split(',').map(x => x.trim()).filter(Boolean);
      const res = await fetch('/api/context-pack', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({query: document.getElementById('ctxQuery').value, agent_role: document.getElementById('ctxRole').value, max_tokens: Number(document.getElementById('ctxTokens').value || 2000), format: document.getElementById('ctxFormat').value, sections, include_debug_trace: true})});
      const data = await res.json();
      document.getElementById('ctxPreview').textContent = data.rendered || data.markdown || JSON.stringify(data, null, 2);
    }
    async function renderBenchmarkLab() {
      const res = await fetch('/api/benchmark-dashboard');
      const data = await res.json();
      const runs = ((data.benchmark_runs || {}).runs || []);
      const candidateRows = ((data.candidate_admission_report || {}).rows || []);
      const latencyRows = ((data.latency_report || {}).rows || []);
      document.getElementById('benchmarkLab').innerHTML = `<div class="summary"><div class="metric"><div class="label">Runs</div><div class="value">${fmt.format(runs.length)}</div></div><div class="metric"><div class="label">Candidate Rows</div><div class="value">${fmt.format(candidateRows.length)}</div></div><div class="metric"><div class="label">Latency Rows</div><div class="value">${fmt.format(latencyRows.length)}</div></div><div class="metric"><div class="label">Dashboard</div><div class="value">${esc(data.status)}</div><div class="sub">${esc(data.dashboard_dir || '')}</div></div></div><div class="panel"><h2>Benchmark Lab</h2>${cards(runs)}</div><pre>${esc(JSON.stringify(data.regression_report || {}, null, 2))}</pre>`;
    }
    async function renderHealthDoctor() {
      const res = await fetch('/api/health');
      const data = await res.json();
      const checks = Object.entries(((data.product || {}).checks || {})).map(([name, value]) => ({name, ...(value || {})}));
      document.getElementById('healthDoctor').innerHTML = `<div class="panel"><h2>Health Doctor</h2><div class="value ${((data.product || {}).status === 'ok') ? 'ok' : 'warn'}">${esc((data.product || {}).status || 'unknown')}</div>${checks.map(c => `<div class="item"><strong>${esc(c.name)}</strong><small>${esc(c.severity || '')}</small><div>${esc(JSON.stringify(c).slice(0, 240))}</div></div>`).join('')}</div>`;
    }
    async function renderMaintenance() {
      const [res, vectorRes, embeddingRes] = await Promise.all([fetch('/api/maintenance'), fetch('/api/index/vector-backends'), fetch('/api/index/embedding-backends')]);
      const data = await res.json();
      const vector = await vectorRes.json();
      const embedding = await embeddingRes.json();
      const items = data.suggestions || [];
      const vectorConfigured = (vector.configured || {}).backend || 'sqlite_inline';
      const embeddingConfigured = (embedding.configured || {}).backend || 'local_hash_embedding';
      document.getElementById('maintenance').innerHTML = `<div class="panel"><h2>Maintenance</h2><div class="toolbar"><button onclick="rebuildProductIndex()">Rebuild Index</button><button onclick="renderMaintenance()">Refresh</button></div><div class="summary"><div class="metric"><div class="label">Suggestions</div><div class="value">${fmt.format(data.count || 0)}</div></div><div class="metric"><div class="label">Open</div><div class="value">${fmt.format(data.open_count || 0)}</div></div><div class="metric"><div class="label">Vector Backend</div><div class="value">${esc(vectorConfigured)}</div><div class="sub">${esc(((vector.backends || {})[vectorConfigured] || {}).description || '')}</div></div><div class="metric"><div class="label">Embedding</div><div class="value">${esc(embeddingConfigured)}</div></div></div>
        <div class="grid"><div class="panel"><h2>Vector Backend</h2><div class="row"><label>Backend</label><select id="vectorBackend"><option>sqlite_inline</option><option>chroma</option></select></div><div class="row"><label>Collection</label><input id="vectorCollection" value="${esc((vector.configured || {}).collection || 'product_capsules')}"></div><div class="toolbar"><button onclick="configureVectorBackend()">Save Vector Backend</button><button onclick="rebuildProductVector()">Rebuild Vector Index</button></div><pre>${esc(JSON.stringify(vector, null, 2))}</pre></div>
        <div class="panel"><h2>Embedding Backend</h2><div class="row"><label>Backend</label><select id="embeddingBackend"><option>local_hash_embedding</option><option>sentence_transformers</option></select></div><div class="row"><label>Model</label><input id="embeddingModel" value="${esc((embedding.configured || {}).model || '')}"></div><div class="toolbar"><button onclick="configureEmbeddingBackend()">Save Embedding Backend</button></div><pre>${esc(JSON.stringify(embedding, null, 2))}</pre></div></div>
        ${items.map(s => `<div class="item"><strong>${esc(s.type)} · ${esc(s.status)}</strong><small>${esc(s.suggestion_id)}</small><div>${esc(s.reason || JSON.stringify(s).slice(0, 160))}</div><div class="toolbar"><button onclick="applySuggestion('${s.suggestion_id}')">Apply</button><button onclick="dismissSuggestion('${s.suggestion_id}')">Dismiss</button></div></div>`).join('') || '<div class="sub">No maintenance suggestions.</div>'}</div><pre id="maintenanceDetail">${esc(JSON.stringify(data, null, 2))}</pre>`;
      const vb = document.getElementById('vectorBackend'); if (vb) vb.value = vectorConfigured;
      const eb = document.getElementById('embeddingBackend'); if (eb) eb.value = embeddingConfigured;
    }
    async function rebuildProductIndex() {
      const res = await fetch('/api/index/rebuild', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({})});
      document.getElementById('maintenanceDetail').textContent = JSON.stringify(await res.json(), null, 2);
    }
    async function configureVectorBackend() {
      const res = await fetch('/api/index/configure-vector', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({backend: document.getElementById('vectorBackend').value, collection: document.getElementById('vectorCollection').value, allow_unavailable: true})});
      document.getElementById('maintenanceDetail').textContent = JSON.stringify(await res.json(), null, 2);
      await renderMaintenance();
    }
    async function rebuildProductVector() {
      const res = await fetch('/api/index/rebuild-vector', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({})});
      document.getElementById('maintenanceDetail').textContent = JSON.stringify(await res.json(), null, 2);
    }
    async function configureEmbeddingBackend() {
      const res = await fetch('/api/index/configure-embedding', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({backend: document.getElementById('embeddingBackend').value, model: document.getElementById('embeddingModel').value || null, allow_unavailable: true})});
      document.getElementById('maintenanceDetail').textContent = JSON.stringify(await res.json(), null, 2);
      await renderMaintenance();
    }
    async function applySuggestion(id) {
      if (!confirm('Apply this maintenance suggestion? This may change capsule validity and aliases.')) return;
      const res = await fetch('/api/maintenance/apply', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({suggestion_id: id})});
      document.getElementById('maintenanceDetail').textContent = JSON.stringify(await res.json(), null, 2);
      await renderMaintenance();
    }
    async function dismissSuggestion(id) {
      if (!confirm('Dismiss this maintenance suggestion?')) return;
      const res = await fetch('/api/maintenance/dismiss', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({suggestion_id: id, reason: 'dismissed in UI'})});
      document.getElementById('maintenanceDetail').textContent = JSON.stringify(await res.json(), null, 2);
      await renderMaintenance();
    }
    async function renderProductSettings() {
      const res = await fetch('/api/settings');
      const data = await res.json();
      const privacy = data.privacy || {};
      document.getElementById('productSettings').innerHTML = `<div class="panel"><h2>Settings</h2>
        <div class="grid"><div><h3>Storage And Privacy</h3><span class="pill">local only ${esc(privacy.local_only)}</span><span class="pill">encryption ${esc(((privacy.encryption_at_rest || {}).status) || 'unknown')}</span><pre>${esc(JSON.stringify(privacy, null, 2))}</pre></div><div><h3>Runtime Settings</h3><pre>${esc(JSON.stringify(data.settings || {}, null, 2))}</pre></div></div></div>`;
    }
    async function renderProductPages() {
      await Promise.all([renderEvidenceSearch(), renderEvidenceTimeline(), renderEvidenceGraph(), renderContextComposer(), renderBenchmarkLab(), renderHealthDoctor(), renderMaintenance(), renderProductSettings()]);
      renderTraceViewer();
    }
    async function loadCockpit() {
      initTabs();
      const [snapRes, tokenRes, resumeRes] = await Promise.all([fetch('/api/runtime/cockpit'), fetch('/api/token-economy'), fetch('/api/resume-context')]);
      snapshot = await snapRes.json();
      const tokenData = await tokenRes.json();
      const resumeData = await resumeRes.json();
      renderMission(await tokenEconomy(), resumeData.rendered_context || 'No previous session context.');
      renderRuntimePages(tokenData);
      await renderProductPages();
    }
    async function saveConfig() {
      const csv = (id) => document.getElementById(id).value.split(',').map(x => x.trim()).filter(Boolean);
      const jsonField = (id) => JSON.parse(document.getElementById(id).value || '{}');
      const payload = {
        context_budget: Number(document.getElementById('cfgBudget').value || 1200),
        embedding_backend: document.getElementById('cfgEmbedding').value,
        lexical_backend: document.getElementById('cfgLexical').value,
        projection_backend: document.getElementById('cfgProjection').value,
        cache_policy: document.getElementById('cfgCache').value,
        enabled_operators: csv('cfgOperators'),
        scheduler_triggers: csv('cfgTriggers'),
        operator_weights: jsonField('cfgWeights'),
        section_limits: jsonField('cfgSections'),
        ui_animation_intensity: document.getElementById('cfgAnim').value
      };
      if (!confirm('Save high-risk memory runtime configuration changes?')) return;
      await fetch('/api/runtime/config', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
      await loadCockpit();
      activate('config');
    }
    function exportConfig() {
      const el = document.getElementById('cfgImport');
      if (el) el.value = JSON.stringify((snapshot.config || {}), null, 2);
    }
    async function importConfig() {
      const payload = JSON.parse(document.getElementById('cfgImport').value || '{}');
      if (!confirm('Import high-risk memory runtime configuration?')) return;
      await fetch('/api/runtime/config', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
      await loadCockpit();
      activate('config');
    }
    async function enqueueRefresh() {
      const res = await fetch('/api/runtime/scheduler/enqueue', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({trigger: 'artifact_updated'})});
      document.getElementById('schedulerState').textContent = JSON.stringify(await res.json(), null, 2);
    }
    async function runSchedulerOnce() {
      const res = await fetch('/api/runtime/scheduler/run-once', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({limit: 10})});
      document.getElementById('schedulerState').textContent = JSON.stringify(await res.json(), null, 2);
    }
    loadCockpit();
  </script>
</body>
</html>
"""


class DysonMemoryHandler(BaseHTTPRequestHandler):
    base_dir: Path = Path.cwd()
    project: str = "DysonSpherain"

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send(self, payload: object, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._send_cors_headers()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _html(self) -> None:
        data = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self._send_cors_headers()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        payload = json.loads(body or "{}")
        return payload if isinstance(payload, dict) else {"payload": payload}

    @staticmethod
    def _bool(value: object, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).lower() in {"1", "true", "yes", "on"}

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        project = query.get("project", [self.project])[0]
        try:
            if parsed.path in {"/", "/index.html"}:
                self._html()
            elif parsed.path == "/api/projects":
                self._send(list_projects(self.base_dir))
            elif parsed.path == "/api/capsules":
                self._send(
                    list_capsules(
                        self.base_dir,
                        project_id=project,
                        limit=int(query.get("limit", ["50"])[0]),
                        offset=int(query.get("offset", ["0"])[0]),
                        include_archived=query.get("include_archived", ["false"])[0].lower() in {"1", "true", "yes"},
                        evidence_type=query.get("type", [None])[0],
                    )
                )
            elif parsed.path.startswith("/api/capsules/"):
                capsule_id = parsed.path.rsplit("/", 1)[-1]
                self._send({"status": "ok", "capsule": get_capsule(self.base_dir, capsule_id, project_id=project)})
            elif parsed.path.startswith("/api/retrieval-traces/"):
                trace_id = parsed.path.rsplit("/", 1)[-1]
                self._send(get_retrieval_trace(self.base_dir, trace_id, project_id=project))
            elif parsed.path.startswith("/api/context-packs/"):
                context_pack_id = parsed.path.rsplit("/", 1)[-1]
                self._send(get_context_pack(self.base_dir, context_pack_id, project_id=project))
            elif parsed.path == "/api/benchmark-runs":
                self._send(list_benchmark_runs(self.base_dir, project_id=project, limit=int(query.get("limit", ["50"])[0])))
            elif parsed.path == "/api/maintenance":
                self._send(maintenance_suggestions(self.base_dir, project_id=project, limit=int(query.get("limit", ["100"])[0])))
            elif parsed.path == "/api/index/embedding-backends":
                self._send(product_embedding_backends(self.base_dir))
            elif parsed.path == "/api/index/vector-backends":
                self._send(product_vector_backends(self.base_dir))
            elif parsed.path == "/api/benchmark-dashboard":
                lab = self.base_dir / ".memory" / "artifacts" / "benchmark_lab"
                payload = {"status": "ok", "project_id": project, "dashboard_dir": str(lab)}
                for name in ("benchmark_runs", "metric_trends", "regression_report", "candidate_admission_report", "latency_report"):
                    path = lab / f"{name}.json"
                    payload[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "missing", "path": str(path)}
                self._send(payload)
            elif parsed.path == "/api/settings":
                config = load_runtime_config(self.base_dir).to_dict()
                self._send({"status": "ok", "project": project, "settings": config, "privacy": privacy_policy(self.base_dir)})
            elif parsed.path == "/api/health":
                self._send({"status": "ok", "project": project, "base_dir": str(self.base_dir), "product": product_doctor(self.base_dir, project_id=project)})
            elif parsed.path == "/api/search":
                self._send(search_observations(self.base_dir, project=project, query=query.get("query", [""])[0], limit=int(query.get("limit", ["20"])[0])))
            elif parsed.path == "/api/timeline":
                self._send(
                    timeline(
                        self.base_dir,
                        project=project,
                        observation_id=query.get("observation_id", [None])[0],
                        session_id=query.get("session_id", [None])[0],
                        limit=int(query.get("limit", ["20"])[0]),
                    )
                )
            elif parsed.path.startswith("/api/observations/"):
                observation_id = parsed.path.rsplit("/", 1)[-1]
                self._send(get_observations(self.base_dir, project=project, observation_ids=[observation_id]))
            elif parsed.path == "/api/resume-context":
                self._send(
                    resume_context(
                        self.base_dir,
                        project=project,
                        session_id=query.get("session_id", [None])[0],
                        lookback_hours=int(query.get("lookback_hours", ["24"])[0]),
                        token_budget=int(query.get("token_budget", ["1200"])[0]),
                    )
                )
            elif parsed.path == "/api/token-economy":
                self._send(token_economy_summary(self.base_dir, project=project))
            elif parsed.path == "/api/runtime/ledger":
                events = replay_events(self.base_dir, project=project)
                self._send({"status": "ok", "event_count": len(events), "events": [event.to_dict() for event in events[-100:]]})
            elif parsed.path == "/api/runtime/graph":
                self._send(graph_state(self.base_dir, project=project))
            elif parsed.path == "/api/runtime/cockpit":
                self._send(cockpit_snapshot(self.base_dir, project=project))
            elif parsed.path == "/api/runtime/config":
                config = load_runtime_config(self.base_dir).to_dict()
                self._send({"status": "ok", "config": config, "export": config})
            elif parsed.path == "/api/runtime/scheduler":
                pending = load_pending_jobs(self.base_dir, project=project)
                self._send({"status": "ok", "pending_count": len(pending), "pending_jobs": [job.to_dict() for job in pending]})
            elif parsed.path == "/api/runtime/latest-packet":
                packet = self.base_dir / "data" / "projections" / "latest_context_packet.json"
                self._send(json.loads(packet.read_text(encoding="utf-8")) if packet.exists() else {"status": "empty"})
            elif parsed.path == "/api/runtime/latest-audit":
                audit = self.base_dir / "data" / "projections" / "latest_recall_audit.json"
                self._send(json.loads(audit.read_text(encoding="utf-8")) if audit.exists() else {"status": "empty"})
            else:
                self._send({"status": "error", "error": "not_found"}, status=404)
        except KeyError as exc:
            self._send({"status": "error", "error": f"not_found: {exc}"}, status=404)
        except Exception as exc:
            self._send({"status": "error", "error": str(exc)}, status=500)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        project = query.get("project", [self.project])[0]
        try:
            payload = self._read_json()
            if parsed.path == "/api/capsules":
                text = str(payload.get("text") or payload.get("raw_text") or payload.get("summary") or "")
                if not text:
                    self._send({"status": "error", "error": "text is required"}, status=400)
                    return
                self._send(
                    remember(
                        self.base_dir,
                        project_id=project,
                        text=text,
                        evidence_type=str(payload.get("evidence_type") or payload.get("type") or "note"),
                        source_type=str(payload.get("source_type") or "api"),
                        title=payload.get("title"),
                        summary=payload.get("summary"),
                        session_id=payload.get("session_id"),
                        task_id=payload.get("task_id"),
                        agent_id=payload.get("agent_id"),
                        validity_state=str(payload.get("validity_state") or "active"),
                        tags=[str(item) for item in payload.get("tags") or []],
                        file_refs=[str(item) for item in payload.get("file_refs") or []],
                        command_refs=[str(item) for item in payload.get("command_refs") or []],
                        artifact_refs=[str(item) for item in payload.get("artifact_refs") or []],
                        benchmark_refs=[str(item) for item in payload.get("benchmark_refs") or []],
                        metadata=dict(payload.get("metadata") or {}),
                    )
                )
            elif parsed.path == "/api/retrieve":
                self._send(
                    retrieve(
                        self.base_dir,
                        project_id=project,
                        query=str(payload.get("query") or ""),
                        limit=int(payload.get("limit") or 10),
                        show_audit=self._bool(payload.get("show_audit"), True),
                        context_pack=self._bool(payload.get("context_pack"), False),
                        max_tokens=int(payload.get("max_tokens") or 2000),
                        task_type=payload.get("task_type"),
                        context_format=str(payload.get("format") or payload.get("context_format") or "markdown"),
                        sections=[str(item) for item in payload.get("sections") or []],
                        section_budget={str(key): int(value) for key, value in dict(payload.get("section_budget") or {}).items()},
                        agent_role=str(payload.get("agent_role") or "coder"),
                        include_raw_quotes=self._bool(payload.get("include_raw_quotes"), False),
                        include_artifact_refs=self._bool(payload.get("include_artifact_refs"), True),
                        include_debug_trace=self._bool(payload.get("include_debug_trace"), False),
                    )
                )
            elif parsed.path == "/api/context-pack":
                self._send(
                    create_context_pack(
                        self.base_dir,
                        project_id=project,
                        query=str(payload.get("query") or ""),
                        max_tokens=int(payload.get("max_tokens") or 2000),
                        agent_role=str(payload.get("agent_role") or "coder"),
                        task_type=payload.get("task_type"),
                        sections=[str(item) for item in payload.get("sections") or []],
                        section_budget={str(key): int(value) for key, value in dict(payload.get("section_budget") or {}).items()},
                        include_raw_quotes=self._bool(payload.get("include_raw_quotes"), False),
                        include_artifact_refs=self._bool(payload.get("include_artifact_refs"), True),
                        include_debug_trace=self._bool(payload.get("include_debug_trace"), False),
                        fmt=str(payload.get("format") or "markdown"),
                    )
                )
            elif parsed.path == "/api/index/rebuild":
                embedding = rebuild_product_embeddings(self.base_dir, project_id=project, include_archived=self._bool(payload.get("include_archived"), False), backend=payload.get("backend"), model=payload.get("model"))
                vector = rebuild_product_vector_index(self.base_dir, project_id=project)
                self._send({"status": "ok", "embedding_rebuild": embedding, "vector_rebuild": vector})
            elif parsed.path == "/api/index/configure-embedding":
                self._send(configure_embedding_backend(self.base_dir, backend=str(payload.get("backend") or ""), model=payload.get("model"), allow_unavailable=self._bool(payload.get("allow_unavailable"), False)))
            elif parsed.path == "/api/index/configure-vector":
                self._send(configure_product_vector_backend(self.base_dir, backend=str(payload.get("backend") or ""), path=Path(payload["path"]) if payload.get("path") else None, collection=str(payload.get("collection") or "product_capsules"), allow_unavailable=self._bool(payload.get("allow_unavailable"), False)))
            elif parsed.path == "/api/index/rebuild-vector":
                self._send(rebuild_product_vector_index(self.base_dir, project_id=project, backend=payload.get("backend"), limit=payload.get("limit")))
            elif parsed.path == "/api/index/configure-encryption":
                self._send(configure_encryption(self.base_dir, provider=str(payload.get("provider") or ""), key_env=str(payload.get("key_env") or "DYSON_MEMORY_SQLCIPHER_KEY"), scope=str(payload.get("scope") or "product_sqlite"), allow_unavailable=self._bool(payload.get("allow_unavailable"), False)))
            elif parsed.path == "/api/maintenance/apply":
                self._send(
                    apply_maintenance_suggestion(
                        self.base_dir,
                        project_id=project,
                        suggestion_id=str(payload.get("suggestion_id") or ""),
                        canonical_id=payload.get("canonical_id"),
                    )
                )
            elif parsed.path == "/api/maintenance/dismiss":
                self._send(
                    dismiss_maintenance_suggestion(
                        self.base_dir,
                        project_id=project,
                        suggestion_id=str(payload.get("suggestion_id") or ""),
                        reason=payload.get("reason"),
                    )
                )
            elif parsed.path in {"/api/settings", "/api/runtime/config"}:
                self._send(save_runtime_config(self.base_dir, dict(payload), project=project))
            elif parsed.path == "/api/runtime/scheduler/enqueue":
                trigger = str(payload.get("trigger") or "artifact_updated")
                event_ids = [str(item) for item in payload.get("event_ids") or []]
                jobs = enqueue_maintenance_jobs(self.base_dir, trigger, event_ids, project=project)
                self._send({"status": "ok", "jobs": [job.to_dict() for job in jobs]})
            elif parsed.path == "/api/runtime/scheduler/run-once":
                self._send(run_scheduler_once(self.base_dir, project=project, limit=int(payload.get("limit") or 10)))
            else:
                self._send({"status": "error", "error": "not_found"}, status=404)
        except Exception as exc:
            self._send({"status": "error", "error": str(exc)}, status=500)

    def do_PATCH(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        project = query.get("project", [self.project])[0]
        try:
            payload = self._read_json()
            if parsed.path.startswith("/api/capsules/"):
                capsule_id = parsed.path.rsplit("/", 1)[-1]
                self._send(update_capsule(self.base_dir, capsule_id, project_id=project, updates=payload))
            elif parsed.path == "/api/settings":
                current = load_runtime_config(self.base_dir).to_dict()
                current.update(payload)
                self._send(save_runtime_config(self.base_dir, current, project=project))
            else:
                self._send({"status": "error", "error": "not_found"}, status=404)
        except KeyError as exc:
            self._send({"status": "error", "error": f"not_found: {exc}"}, status=404)
        except Exception as exc:
            self._send({"status": "error", "error": str(exc)}, status=500)

    def log_message(self, format: str, *args: object) -> None:
        return


def run_server(base_dir: Path, *, host: str = "127.0.0.1", port: int = 37777, project: str = "DysonSpherain") -> None:
    handler = type("ConfiguredDysonMemoryHandler", (DysonMemoryHandler,), {"base_dir": base_dir.resolve(), "project": project})
    server = ThreadingHTTPServer((host, port), handler)
    print(json.dumps({"status": "ok", "url": f"http://{host}:{port}", "project": project}, ensure_ascii=False), flush=True)
    server.serve_forever()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--project", default="DysonSpherain")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=37777)
    args = parser.parse_args(argv)
    run_server(Path(args.base_dir), host=args.host, port=args.port, project=args.project)


if __name__ == "__main__":
    main()
