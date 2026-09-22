// 发现页：检索、入库下钻、按库推荐（带开源代码加分）、arXiv 追踪。

import { api } from '../api.js';
import { state, targetDocIds } from '../state.js';
import { el, qs, clear, toast, fmtNumber } from '../dom.js';

function candidateCard(item) {
  const authors = (item.authors || []).slice(0, 4).join(', ');
  const card = el('div', { class: 'cand' }, [
    el('div', { class: 't', text: item.title || item.identifier || '(无标题)' }),
    el('div', { class: 'm' }, [
      el('span', { text: `${item.year || '?'} · ${item.source || ''}` }),
      item.citation_count !== undefined && item.citation_count !== null
        ? el('span', { class: 'badge', text: `被引 ${item.citation_count}` }) : '',
      item.has_code ? el('span', { class: 'badge latest', text: '有开源代码' }) : '',
    ]),
    authors ? el('div', { class: 'a', text: authors }) : '',
    item.reason ? el('div', { class: 'reason', text: `推荐理由：${item.reason}` }) : '',
  ]);
  if (item.abstract) card.append(el('div', { class: 'a', text: `${item.abstract.slice(0, 260)}…` }));

  const actions = el('div', { class: 'acts' });
  if (item.url) actions.append(el('a', { href: item.url, target: '_blank', rel: 'noreferrer', class: 'ghost small', text: '打开' }));
  if (item.code_url) actions.append(el('a', { href: item.code_url, target: '_blank', rel: 'noreferrer', class: 'ghost small', text: '代码' }));
  if (item.identifier && (item.source === 'arxiv' || !item.source)) {
    actions.append(el('button', {
      class: 'primary small', type: 'button', text: '下载入库',
      onclick: async (event) => {
        const button = event.target;
        button.disabled = true;
        button.textContent = '下载中…';
        try {
          const res = await api.discoverIngest(item.identifier, 'arxiv');
          toast(`已入库（${res.status}）`);
          document.dispatchEvent(new CustomEvent('zhiwei:library-changed'));
        } catch (err) {
          toast(`入库失败：${err.message}`, { kind: 'error' });
        } finally {
          button.disabled = false;
          button.textContent = '下载入库';
        }
      },
    }));
  }
  actions.append(el('button', {
    class: 'ghost small', type: 'button', text: '赞',
    onclick: async () => {
      try { await api.feedback(item.identifier || item.title, 'like'); toast('已记录偏好'); }
      catch (err) { toast(err.message, { kind: 'error' }); }
    },
  }));
  actions.append(el('button', {
    class: 'ghost small', type: 'button', text: '踩',
    onclick: async () => {
      try { await api.feedback(item.identifier || item.title, 'dislike'); toast('已记录偏好'); }
      catch (err) { toast(err.message, { kind: 'error' }); }
    },
  }));
  card.append(actions);
  return card;
}

async function runSearch() {
  const query = qs('#discover-query').value.trim();
  if (!query) {
    toast('先写检索词');
    return;
  }
  const box = qs('#discover-results');
  box.replaceChildren(el('p', { class: 'muted', text: '检索中（arXiv / OpenAlex）…' }));
  try {
    const payload = await api.discoverSearch(query, 20);
    clear(box);
    const results = payload.results || [];
    if (!results.length) {
      box.append(el('p', { class: 'muted', text: '没有结果（可能断网或被限流）' }));
      return;
    }
    for (const item of results) box.append(candidateCard(item));
  } catch (err) {
    box.replaceChildren(el('p', { class: 'muted', text: `检索失败：${err.message}` }));
    toast(`检索失败：${err.message}`, { kind: 'error', retry: runSearch });
  }
}

async function runRecommend() {
  const box = qs('#discover-recommend-list');
  box.replaceChildren(el('p', { class: 'muted', text: '按你的文献库打分中…' }));
  try {
    const payload = await api.recommend(targetDocIds()[0] || '');
    clear(box);
    const items = payload.items || [];
    if (!items.length) {
      box.append(el('p', { class: 'muted', text: payload.note || '暂时没有推荐' }));
      return;
    }
    for (const item of items) {
      const card = candidateCard(item);
      if (item.score_parts) {
        const parts = Object.entries(item.score_parts)
          .map(([key, value]) => `${key} ${fmtNumber(value, 2)}`).join(' · ');
        card.append(el('div', { class: 'reason', text: parts }));
      }
      box.append(card);
    }
  } catch (err) {
    box.replaceChildren(el('p', { class: 'muted', text: `推荐失败：${err.message}` }));
    toast(`推荐失败：${err.message}`, { kind: 'error' });
  }
}

async function refreshTrack() {
  const box = qs('#discover-track-status');
  try {
    const payload = await api.trackStatus();
    box.textContent = payload.last_run
      ? `上次 ${payload.last_run} · 主题 ${(payload.tracked_topics || []).join('、') || '（未设置）'} · 新文献 ${payload.new_items}`
      : '还没跑过追踪';
  } catch (err) {
    box.textContent = `读取失败：${err.message}`;
  }
}

async function runTrack() {
  const box = qs('#discover-track-status');
  box.textContent = '追踪中…';
  try {
    const myDocs = state.papers.slice(0, 3).map((p) => p.title).filter(Boolean);
    const payload = await api.trackRun(myDocs, 30);
    box.textContent = `本次扫描 ${payload.scanned || 0} 篇，新增 ${payload.new_count || 0} 篇（${payload.ran_at || ''}）`;
    const list = qs('#discover-recommend-list');
    clear(list);
    for (const item of payload.new_items || []) list.append(candidateCard(item));
  } catch (err) {
    box.textContent = `追踪失败：${err.message}`;
    toast(`追踪失败：${err.message}`, { kind: 'error', retry: runTrack });
  }
}

export function mount() {
  qs('#discover-search').addEventListener('click', runSearch);
  qs('#discover-recommend').addEventListener('click', runRecommend);
  qs('#discover-track').addEventListener('click', runTrack);
  qs('#discover-query').addEventListener('keydown', (event) => {
    if (event.key === 'Enter') { event.preventDefault(); runSearch(); }
  });
  refreshTrack();
}
