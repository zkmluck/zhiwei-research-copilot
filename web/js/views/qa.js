// 问答页：SSE 流式回答 + 逐句溯源面板。这是本项目最该被看见的一屏。

import { api } from '../api.js';
import { state, targetDocIds } from '../state.js';
import { el, qs, clear, toast, fmtPercent, spinner } from '../dom.js';

const VERDICT_LABEL = { allow: '放行', downgrade: '待核实', block: '拦下' };

let controller = null;

function renderTrace(trace) {
  const box = qs('#qa-trace');
  clear(box);
  for (const step of trace) {
    box.append(el('div', { class: 'step done' }, [
      el('b', { text: step.label || step.step || '步骤' }),
      el('span', { text: `  ${step.detail || ''}` }),
      step.ms ? el('span', { class: 'muted', text: `  ${step.ms}ms` }) : '',
    ]));
  }
}

function claimCard(claim, onLocate) {
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
      text: `${evidence.doc_title || evidence.doc_id} · 第 ${anchor.page} 页：「${String(anchor.quote || '').slice(0, 90)}…」`,
      onclick: () => onLocate(evidence),
    }));
  }
  return card;
}

async function run() {
  const question = qs('#qa-question').value.trim();
  if (!question) {
    toast('先写一个问题');
    return;
  }
  const docIds = targetDocIds();
  const mode = qs('#qa-mode').value;
  if (!docIds.length) {
    toast('先选一篇文献（左栏里点一下）');
    return;
  }

  const answerBox = qs('#qa-text');
  const claimsBox = qs('#qa-claims');
  clear(answerBox);
  clear(claimsBox);
  clear(qs('#qa-trace'));
  qs('#qa-meta').textContent = '';
  qs('#qa-run').disabled = true;
  qs('#qa-run').replaceChildren(spinner(), document.createTextNode(' 推理中'));

  const trace = [];
  const claims = [];
  let claimIndex = 0;

  const locate = (evidence) => {
    document.dispatchEvent(new CustomEvent('zhiwei:locate', {
      detail: { docId: evidence.doc_id, page: (evidence.anchor || {}).page || 1, quote: (evidence.anchor || {}).quote || '' },
    }));
  };

  controller = new AbortController();
  try {
    await api.qaStream({ question, doc_ids: docIds, mode, top_k: 8 }, (event, payload) => {
      if (event === 'trace') {
        trace.push(payload);
        renderTrace(trace);
      } else if (event === 'evidence') {
        const count = (payload.evidence || []).length;
        qs('#qa-meta').textContent = `检索到 ${count} 条证据`;
      } else if (event === 'claim') {
        claims.push(payload);
        claimIndex += 1;
        const card = claimCard(payload, locate);
        card.style.animation = 'none';
        claimsBox.prepend(card);
      } else if (event === 'token') {
        answerBox.append(document.createTextNode(payload.text || ''));
      } else if (event === 'done') {
        qs('#qa-meta').textContent =
          `支持率 ${fmtPercent(payload.support_rate)} · ${payload.refused ? '已拒答' : '未拒答'} · ` +
          `模型调用 ${(payload.usage && payload.usage.calls) || 0} 次`;
      } else if (event === 'error') {
        toast(`${payload.kind}：${payload.message}`, { kind: 'error' });
      }
    }, controller.signal);
  } catch (err) {
    if (err.name !== 'AbortError') {
      toast(`问答失败：${err.message}`, { kind: 'error', retry: run });
    }
  } finally {
    controller = null;
    const button = qs('#qa-run');
    button.disabled = false;
    button.textContent = '提问';
  }
}

export function mount() {
  qs('#qa-run').addEventListener('click', run);
  qs('#qa-question').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      run();
    }
  });
}
