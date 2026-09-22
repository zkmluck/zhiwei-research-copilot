// 左栏 · 文献库：上传、列表、状态徽标、版本族谱、按库内检索过滤。

import { api } from '../api.js';
import { state, update, activePaper } from '../state.js';
import { el, qs, clear, toast, spinner } from '../dom.js';

const STATUS_LABEL = {
  active: '在用',
  superseded: '被取代',
  duplicate: '重复',
  failed: '解析失败',
};

function badge(record) {
  const badges = [];
  if (record.is_latest) badges.push(el('span', { class: 'badge latest', text: '最新版' }));
  else if (record.status === 'superseded') badges.push(el('span', { class: 'badge superseded', text: '被取代' }));
  if (record.needs_ocr) badges.push(el('span', { class: 'badge ocr', text: '扫描件' }));
  if (record.arxiv_id) badges.push(el('span', { class: 'badge', text: `arXiv ${record.arxiv_id}` }));
  if (record.version && record.version > 1) badges.push(el('span', { class: 'badge', text: `v${record.version}` }));
  return badges;
}

export function renderPapers() {
  const list = qs('#paper-list');
  const keyword = (qs('#library-search').value || '').trim().toLowerCase();
  const papers = state.papers.filter((p) => {
    if (!keyword) return true;
    const haystack = `${p.title} ${(p.authors || []).join(' ')} ${p.arxiv_id || ''}`.toLowerCase();
    return haystack.includes(keyword);
  });

  clear(list);
  if (!papers.length) {
    list.append(el('li', { class: 'muted', text: keyword ? '没有匹配的文献' : '还没有文献，先上传一篇 PDF' }));
  }

  const families = new Map();
  for (const paper of papers) {
    const key = paper.family_id || paper.doc_id;
    if (!families.has(key)) families.set(key, []);
    families.get(key).push(paper);
  }

  for (const [, group] of families) {
    group.sort((a, b) => (b.version || 1) - (a.version || 1));
    for (const paper of group) {
      const pick = el('input', {
        type: 'checkbox',
        class: 'pick',
        title: '勾选后可做跨文献对比 / 图谱',
        onclick: (event) => event.stopPropagation(),
      });
      pick.checked = state.selectedDocIds.has(paper.doc_id);
      pick.addEventListener('change', () => {
        if (pick.checked) state.selectedDocIds.add(paper.doc_id);
        else state.selectedDocIds.delete(paper.doc_id);
        renderFoot();
      });

      const item = el('li', {
        class: `paper-item${paper.doc_id === state.activeDocId ? ' is-active' : ''}`,
        onclick: () => selectPaper(paper.doc_id),
      }, [
        el('span', { class: 'title-row' }, [
          pick,
          el('span', { class: 't', text: paper.title || paper.file_name || paper.doc_id }),
        ]),
        el('span', { class: 'm' }, [
          el('span', { text: `${paper.year || '年份未知'} · ${paper.page_count || '?'} 页` }),
          ...badge(paper),
        ]),
      ]);
      if (group.length > 1) {
        item.append(el('span', {
          class: 'm',
          text: `版本族谱：${group.map((p) => `v${p.version || 1}`).join(' → ')}`,
        }));
      }
      if (paper.status === 'duplicate' || !paper.is_latest) {
        const actions = el('span', { class: 'm' }, [
          el('button', {
            class: 'ghost small', type: 'button', text: '设为当前版本',
            onclick: async (event) => {
              event.stopPropagation();
              try {
                await api.resolveDedup(paper.doc_id, 'replace');
                toast('已把这一版设为当前版本');
                await refresh();
              } catch (err) { toast(`操作失败：${err.message}`, { kind: 'error' }); }
            },
          }),
          el('button', {
            class: 'ghost small', type: 'button', text: '删除',
            onclick: async (event) => {
              event.stopPropagation();
              try {
                await api.removePaper(paper.doc_id);
                toast('已移出文献库');
                await refresh();
              } catch (err) { toast(`删除失败：${err.message}`, { kind: 'error' }); }
            },
          }),
        ]);
        item.append(actions);
      }
      list.append(item);
    }
  }

  renderFoot();
}

function renderFoot() {
  const total = state.papers.length;
  const picked = state.selectedDocIds.size;
  qs('#library-foot').textContent = total
    ? `共 ${total} 篇 · 已勾选 ${picked} 篇（勾选多篇可做跨文献对比 / 图谱）`
    : '库是空的';
}

async function selectPaper(docId) {
  update({ activeDocId: docId, page: 1 });
  renderPapers();
  document.dispatchEvent(new CustomEvent('zhiwei:open-paper', { detail: { docId, page: 1 } }));
}

export async function refresh() {
  const data = await api.papers();
  let papers = data.papers || [];
  // 同一篇论文只保留"最新版在前"的顺序，其余按时间倒序
  papers = papers.slice().sort((a, b) => String(b.ingested_at || '').localeCompare(String(a.ingested_at || '')));
  update({ papers });
  if (!state.activeDocId && papers.length) {
    const first = papers.find((p) => p.is_latest) || papers[0];
    update({ activeDocId: first.doc_id });
    document.dispatchEvent(new CustomEvent('zhiwei:open-paper', { detail: { docId: first.doc_id, page: 1 } }));
  }
  renderPapers();
}

async function handleFiles(files) {
  const list = qs('#upload-list');
  const items = Array.from(files).filter((f) => f.name.toLowerCase().endsWith('.pdf'));
  if (!items.length) {
    toast('只支持 PDF 文件', { kind: 'error' });
    return;
  }
  const form = new FormData();
  for (const file of items) {
    form.append('files', file);
    list.append(el('div', { text: `解析中 ${file.name}` }));
  }
  try {
    const response = await fetch('/api/papers/upload', { method: 'POST', body: form });
    const data = await response.json();
    if (!response.ok) throw new Error((data.error && data.error.message) || '上传失败');
    clear(list);
    for (const result of data.results || []) {
      const label = result.status === 'failed' ? 'fail' : 'done';
      const name = result.file_name || result.doc_id;
      if (result.status === 'needs_decision') {
        list.append(el('div', { class: 'done', text: `${name}：检测到版本冲突` }));
        await askDedup(result);
      } else if (result.status === 'duplicate') {
        list.append(el('div', { class: 'done', text: `${name}：库中已有同一版本` }));
      } else if (result.status === 'failed') {
        const reason = result.error && result.error.message ? result.error.message : '解析失败';
        list.append(el('div', { class: 'fail', text: `${name}：${reason}` }));
      } else {
        list.append(el('div', { class: 'done', text: `${name}：已入库` }));
      }
    }
    await refresh();
    document.dispatchEvent(new CustomEvent('zhiwei:library-changed'));
  } catch (err) {
    clear(list);
    list.append(el('div', { class: 'fail', text: `上传失败：${err.message}` }));
    toast(`上传失败：${err.message}`, { kind: 'error' });
  }
}

async function askDedup(result) {
  const dedup = result.dedup || {};
  const message = dedup.message || '检测到同一篇论文的另一个版本。';
  const action = window.confirm(
    `${message}\n\n建议：${dedup.recommendation || 'replace'}\n\n` +
    '点「确定」用新版本替换旧版本；点「取消」保留两个版本。',
  );
  try {
    await api.resolveDedup(result.doc_id, action ? 'replace' : 'keep_both');
    toast(action ? '已用新版本替换旧版本' : '两个版本都保留了');
  } catch (err) {
    toast(`版本处置失败：${err.message}`, { kind: 'error' });
  }
}

export function mount() {
  const uploader = qs('#uploader');
  const input = qs('#file-input');
  qs('#btn-pick').addEventListener('click', () => input.click());
  input.addEventListener('change', () => handleFiles(input.files));

  ['dragenter', 'dragover'].forEach((type) => uploader.addEventListener(type, (event) => {
    event.preventDefault();
    uploader.classList.add('dragover');
  }));
  ['dragleave', 'drop'].forEach((type) => uploader.addEventListener(type, (event) => {
    event.preventDefault();
    uploader.classList.remove('dragover');
  }));
  uploader.addEventListener('drop', (event) => handleFiles(event.dataTransfer.files));

  qs('#btn-refresh-library').addEventListener('click', async () => {
    try { await refresh(); toast('文献库已刷新'); } catch (err) { toast(`刷新失败：${err.message}`, { kind: 'error' }); }
  });
  qs('#library-search').addEventListener('input', renderPapers);
}
