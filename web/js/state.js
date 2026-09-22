// 全局状态。文件不大，直接用一个可订阅对象，省掉一层框架。

const listeners = new Set();

export const state = {
  papers: [],
  activeDocId: '',
  page: 1,
  zoom: 1,
  tab: 'reader',
  selectedDocIds: new Set(),
  highlight: null,
};

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function update(patch) {
  Object.assign(state, patch);
  for (const fn of listeners) {
    try { fn(state); } catch (err) { console.error(err); }
  }
}

export function activePaper() {
  return state.papers.find((p) => p.doc_id === state.activeDocId) || null;
}

export function targetDocIds() {
  if (state.selectedDocIds.size) return Array.from(state.selectedDocIds);
  return state.activeDocId ? [state.activeDocId] : [];
}
