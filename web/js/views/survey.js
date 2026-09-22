// 综述页：从图谱长出来的可溯源综述 + 时间线 + Future Work 时效性扫描。

import { api } from '../api.js';
import { state, targetDocIds } from '../state.js';
import { el, qs, clear, toast, renderMarkdown } from '../dom.js';

const STATUS_LABEL = {
  open: '仍待解决',
  resolved: '已被后续工作解决',
  partially: '部分被解决',
  stale: '可能已过时',
  unknown: '暂未判定',
};

function renderTimeline(rows) {
  const box = qs('#survey-timeline');
  clear(box);
  if (!rows.length) {
    box.append(el('p', { class: 'muted', text: '暂无时间线' }));
    return;
  }
  for (const row of rows) {
    box.append(el('div', { class: 'timeline-row' }, [
      el('span', { class: 'y', text: String(row.year || '?') }),
      el('span', { text: `${row.title || (row.doc_ids || []).join(', ')}${row.role ? ` · ${row.role}` : ''}` }),
    ]));
  }
}

async function run() {
  const docIds = targetDocIds();
  if (!docIds.length) {
    toast('先选文献（左栏可多选）');
    return;
  }
  const button = qs('#survey-run');
  const box = qs('#survey-body');
  button.disabled = true;
  button.textContent = '生成中…';
  box.innerHTML = '<p class="muted">正在从引用图谱里挑材料，并逐句过闸门…</p>';
  try {
    const payload = await api.survey(docIds, qs('#survey-topic').value.trim());
    box.innerHTML = renderMarkdown(payload.markdown || '（没有生成内容）');
    renderTimeline(payload.timeline || []);
    const stats = payload.stats || {};
    if (stats.materials !== undefined) {
      box.prepend(el('p', {
        class: 'muted',
        text: `材料 ${stats.materials} 条 · 放行 ${stats.allowed} 句 · 待核实 ${stats.downgraded} 句 · 拦下 ${stats.blocked} 句`,
      }));
    }
  } catch (err) {
    box.innerHTML = '';
    box.append(el('p', { class: 'muted', text: `生成失败：${err.message}` }));
    toast(`综述生成失败：${err.message}`, { kind: 'error', retry: run });
  } finally {
    button.disabled = false;
    button.textContent = '生成综述';
  }
}

async function scanFuture() {
  const docIds = targetDocIds();
  if (!docIds.length) {
    toast('先选文献');
    return;
  }
  const box = qs('#survey-future-list');
  box.replaceChildren(el('p', { class: 'muted', text: '扫描中（会去 OpenAlex 查后续工作）…' }));
  try {
    const payload = await api.futureWork(docIds, true);
    clear(box);
    const items = payload.items || [];
    if (!items.length) {
      box.append(el('p', { class: 'muted', text: '没有抽到可用的 Future Work 段落' }));
      return;
    }
    for (const item of items) {
      box.append(el('div', { class: 'claim-card' }, [
        el('div', { class: 'head' }, [
          el('span', { class: 'badge', text: STATUS_LABEL[item.status] || item.status || '未判定' }),
          el('span', { text: item.doc_title || item.doc_id }),
        ]),
        el('div', { class: 'body', text: item.text }),
        item.evidence ? el('div', { class: 'reasons', text: item.evidence }) : '',
        (item.resolved_by || []).length
          ? el('div', { class: 'reasons', text: `疑似已被解决：${item.resolved_by.join('、')}` }) : '',
      ]));
    }
  } catch (err) {
    box.replaceChildren(el('p', { class: 'muted', text: `扫描失败：${err.message}` }));
    toast(`Future Work 扫描失败：${err.message}`, { kind: 'error' });
  }
}

export function mount() {
  qs('#survey-run').addEventListener('click', run);
  qs('#survey-future').addEventListener('click', scanFuture);
}
