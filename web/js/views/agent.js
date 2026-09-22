// 调研页：把 Agent 的「规划 → 调工具 → 记观察 → 收尾」整条链路摊开给人看。
// 这一页存在的意义是「不做黑盒」：每一步调了什么、为什么、拿到什么、花了多久，全部写在时间线上。

import { api } from '../api.js';
import { targetDocIds } from '../state.js';
import { el, qs, clear, toast, fmtPercent, spinner } from '../dom.js';

const VERDICT_LABEL = { allow: '放行', downgrade: '待核实', block: '拦下' };
const STEP_LABEL = { running: '执行中', done: '完成', failed: '失败' };

let controller = null;
const stepNodes = new Map();

function meta(text) {
  qs('#agent-meta').textContent = text;
}

function renderTools(tools) {
  const box = qs('#agent-tools');
  clear(box);
  for (const tool of tools) {
    box.append(el('div', { class: 'tool-item' }, [
      el('b', { text: tool.name }),
      tool.network ? el('span', { class: 'net', text: '需要外网' }) : '',
      el('div', { class: 'why', text: tool.description }),
    ]));
  }
}

function planNote(payload) {
  const box = qs('#agent-timeline');
  const source = payload.source === 'llm' ? '模型规划' : payload.source === 'llm-replan' ? '模型再规划' : '确定性规划（无模型）';
  box.append(el('div', { class: 'agent-plan-note' }, [
    el('b', { text: payload.replanned ? '追加计划' : '执行计划' }),
    el('span', { text: `　${source} · ${(payload.steps || []).length} 步` }),
    payload.stop_when ? el('div', { class: 'why', text: `收尾条件：${payload.stop_when}` }) : '',
  ]));
}

function ensureStep(payload) {
  const index = payload.step;
  let node = stepNodes.get(index);
  if (!node) {
    node = el('div', { class: 'agent-step running' }, [
      el('div', { class: 'head' }, [
        el('span', { class: 'tool', text: payload.tool || '' }),
        el('span', { class: 'status', text: '执行中' }),
        el('span', { class: 'ms muted', text: '' }),
      ]),
      el('div', { class: 'why', text: payload.label || payload.why || '' }),
      el('div', { class: 'obs', text: '' }),
    ]);
    stepNodes.set(index, node);
    qs('#agent-timeline').append(node);
  }
  return node;
}

function updateStep(payload) {
  const node = ensureStep(payload);
  node.className = `agent-step ${payload.status === 'running' ? 'running' : payload.ok === false ? 'failed' : 'done'}`;
  node.querySelector('.status').textContent = STEP_LABEL[payload.status] || payload.status;
  node.querySelector('.ms').textContent = payload.ms ? `${payload.ms}ms` : '';
  if (payload.summary) node.querySelector('.obs').textContent = payload.summary;
}

function attachObservation(payload) {
  const node = ensureStep({ step: payload.step, tool: payload.tool, label: payload.tool });
  node.querySelector('.obs').textContent = payload.summary || '';
  const obs = node.querySelector('.obs');
  for (const line of (payload.highlights || []).slice(0, 2)) {
    obs.append(el('span', { class: 'hl', text: `· ${line}` }));
  }
}

function renderFindings(findings) {
  const box = qs('#agent-findings');
  clear(box);
  const rows = [];
  if (findings.library) {
    rows.push(['库情', `${findings.library.papers} 篇 / ${findings.library.total_pages ?? '?'} 页 / ${findings.library.total_chunks ?? '?'} 切块`]);
  }
  if (findings.graph_roles) {
    rows.push(['图谱角色', Object.entries(findings.graph_roles).map(([k, v]) => `${k} ${v.length}`).join('、')]);
  }
  for (const node of (findings.key_nodes || []).slice(0, 3)) {
    rows.push([node.role || '核心', `${node.title || node.doc_id}：${node.reason || ''}`]);
  }
  if (findings.future_work_status) {
    rows.push(['Future Work', Object.entries(findings.future_work_status).map(([k, v]) => `${k} ${v}`).join('、')]);
  }
  for (const item of (findings.future_work || []).slice(0, 2)) {
    rows.push([item.status || '待办', item.text || '']);
  }
  for (const cand of (findings.external_candidates || []).slice(0, 2)) {
    rows.push(['外部候选', `${cand.title}（${cand.year || '?'}）`]);
  }
  rows.push(['证据池', `${(findings.evidence || []).length} 条带锚点的原文`]);
  for (const [key, value] of rows) {
    box.append(el('div', { class: 'row' }, [el('b', { text: `${key}：` }), el('span', { text: String(value) })]));
  }
}

function claimCard(claim) {
  const decision = claim.decision || {};
  const verdict = decision.verdict || 'downgrade';
  const card = el('div', { class: `claim-card ${verdict}` }, [
    el('div', { class: 'head' }, [
      el('span', { class: `badge ${verdict}`, text: VERDICT_LABEL[verdict] || verdict }),
      el('span', { text: `置信 ${fmtPercent(decision.confidence)}` }),
    ]),
    el('div', { class: 'body', text: claim.text }),
  ]);
  if (decision.reasons && decision.reasons.length) {
    card.append(el('div', { class: 'reasons', text: decision.reasons.slice(0, 3).join('；') }));
  }
  for (const evidence of claim.evidence || []) {
    const anchor = evidence.anchor || {};
    card.append(el('div', {
      class: 'ev',
      text: `${evidence.doc_title || evidence.doc_id} · 第 ${anchor.page} 页：「${String(anchor.quote || '').slice(0, 88)}…」`,
      onclick: () => document.dispatchEvent(new CustomEvent('zhiwei:locate', {
        detail: { docId: evidence.doc_id, page: anchor.page || 1, quote: anchor.quote || '' },
      })),
    }));
  }
  return card;
}

function reset() {
  stepNodes.clear();
  clear(qs('#agent-timeline'));
  clear(qs('#agent-claims'));
  clear(qs('#agent-text'));
  clear(qs('#agent-findings'));
  qs('#agent-answer-meta').textContent = '';
}

async function runPlanOnly() {
  const goal = qs('#agent-goal').value.trim();
  if (!goal) {
    toast('先写清楚要调研什么');
    return;
  }
  const button = qs('#agent-plan-only');
  button.disabled = true;
  try {
    reset();
    const payload = await api.agentPlan(goal, targetDocIds());
    meta(`规划来源：${payload.source === 'llm' ? '模型' : '确定性规则'} · ${(payload.steps || []).length} 步`);
    planNote(payload);
    (payload.steps || []).forEach((step) => {
      const node = el('div', { class: 'agent-step' }, [
        el('div', { class: 'head' }, [
          el('span', { class: 'tool', text: `${step.index + 1}. ${step.tool}` }),
          el('span', { class: 'status', text: '待执行' }),
        ]),
        el('div', { class: 'why', text: step.why || step.label }),
      ]);
      qs('#agent-timeline').append(node);
    });
  } catch (err) {
    toast(`规划失败：${err.message}`, { kind: 'error' });
  } finally {
    button.disabled = false;
  }
}

async function run() {
  const goal = qs('#agent-goal').value.trim();
  if (!goal) {
    toast('先写清楚要调研什么');
    return;
  }
  reset();
  meta('Agent 正在规划…');
  const runButton = qs('#agent-run');
  const stopButton = qs('#agent-stop');
  runButton.disabled = true;
  runButton.replaceChildren(spinner(), document.createTextNode(' 调研中'));
  stopButton.hidden = false;
  controller = new AbortController();
  let lastMemory = null;
  try {
    await api.agentRun({ goal, doc_ids: targetDocIds(), mode: qs('#agent-mode').value }, (event, payload) => {
      if (event === 'plan') {
        planNote(payload);
        meta(`规划来源：${payload.source === 'llm' ? '模型' : payload.source === 'llm-replan' ? '模型再规划' : '确定性规则'} · 共 ${(payload.steps || []).length} 步`);
      } else if (event === 'step') {
        updateStep(payload);
      } else if (event === 'observation') {
        attachObservation(payload);
        lastMemory = payload.memory || lastMemory;
        if (lastMemory) {
          meta(`已执行 ${lastMemory.observations + lastMemory.folded} 步 · 观察占用 ${lastMemory.chars} / ${lastMemory.budget_chars} 字符` +
            (lastMemory.folded ? ` · 已折叠 ${lastMemory.folded} 条较早观察` : ''));
        }
      } else if (event === 'claim') {
        qs('#agent-claims').append(claimCard(payload));
      } else if (event === 'done') {
        qs('#agent-text').textContent = payload.text || '';
        const counts = payload.claim_counts || {};
        qs('#agent-answer-meta').textContent =
          `支持率 ${fmtPercent(payload.support_rate)} · 放行 ${counts.allow || 0} / 待核实 ${counts.downgrade || 0} / 拦下 ${counts.block || 0}` +
          ` · 用时 ${(payload.ms / 1000).toFixed(1)}s` +
          (payload.stopped_early ? ` · ${payload.stop_reason}` : '');
        renderFindings(payload.findings || {});
      } else if (event === 'error') {
        toast(`${payload.kind}：${payload.message}`, { kind: 'error' });
      }
    }, controller.signal);
  } catch (err) {
    if (err.name !== 'AbortError') {
      toast(`调研失败：${err.message}`, { kind: 'error', retry: run });
    }
  } finally {
    controller = null;
    runButton.disabled = false;
    runButton.textContent = '开始调研';
    stopButton.hidden = true;
  }
}

export async function mount() {
  qs('#agent-run').addEventListener('click', run);
  qs('#agent-plan-only').addEventListener('click', runPlanOnly);
  qs('#agent-stop').addEventListener('click', () => {
    if (controller) controller.abort();
    meta('已停止');
  });
  qs('#agent-goal').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      run();
    }
  });
  try {
    const payload = await api.agentTools();
    renderTools(payload.tools || []);
  } catch (err) {
    qs('#agent-tools').textContent = '工具名册加载失败（后端未启动？）';
  }
}
