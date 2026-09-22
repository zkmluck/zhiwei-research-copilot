// 应用外壳：标签页、健康状态、用量挂件，以及各视图之间的事件桥。

import { api } from './api.js';
import { state, update } from './state.js';
import { qs, qsa, toast, fmtNumber } from './dom.js';
import { warmup } from './libs.js';
import * as library from './views/library.js';
import * as reader from './views/reader.js';
import * as qa from './views/qa.js';
import * as graph from './views/graph.js';
import * as survey from './views/survey.js';
import * as writing from './views/writing.js';
import * as discover from './views/discover.js';

function setTab(name) {
  update({ tab: name });
  qsa('.tab').forEach((button) => button.classList.toggle('is-active', button.dataset.tab === name));
  qsa('.tab-panel').forEach((panel) => panel.classList.toggle('is-active', panel.dataset.panel === name));
  // 面板从 display:none 变成可见后，图表容器才有真实尺寸，需要让它重新量一次
  window.dispatchEvent(new Event('resize'));
}

async function refreshHealth() {
  const dot = qs('#health-dot');
  const text = qs('#health-text');
  try {
    const payload = await api.health();
    const ok = payload.llm_configured;
    dot.className = `dot ${ok ? 'ok' : 'warn'}`;
    text.textContent = ok ? '模型就绪' : '离线词法模式';
    text.title = JSON.stringify(payload.models || {}, null, 2);
  } catch (err) {
    dot.className = 'dot bad';
    text.textContent = '后端未连接';
  }
}

async function refreshUsage() {
  try {
    const payload = await api.usage();
    const node = qs('#usage-cost');
    node.textContent = `¥${fmtNumber(payload.total_cost_yuan || 0, 4)}`;
    node.title = `调用 ${payload.calls || 0} 次 · 失败 ${payload.failed_calls || 0} 次 · 输入 ${payload.prompt_tokens || 0} tokens`;
  } catch (err) {
    qs('#usage-cost').textContent = '—';
  }
}

async function boot() {
  warmup();
  qsa('.tab').forEach((button) => button.addEventListener('click', () => setTab(button.dataset.tab)));
  qs('#btn-usage').addEventListener('click', refreshUsage);
  qs('#btn-health').addEventListener('click', refreshHealth);

  library.mount();
  reader.mount();
  qa.mount();
  graph.mount();
  survey.mount();
  writing.mount();
  discover.mount();

  document.addEventListener('zhiwei:open-paper', (event) => {
    setTab('reader');
    reader.open(event.detail.docId, event.detail.page || 1);
  });
  document.addEventListener('zhiwei:locate', (event) => {
    const { docId, page, quote } = event.detail;
    setTab('reader');
    update({ activeDocId: docId });
    library.renderPapers();
    reader.open(docId, page).then(() => reader.highlight(docId, page, quote));
  });
  document.addEventListener('zhiwei:library-changed', async () => {
    graph.invalidate();
    await library.refresh();
    await refreshUsage();
  });

  try {
    await library.refresh();
  } catch (err) {
    toast(`文献库加载失败：${err.message}（后端可能没启动）`, { kind: 'error' });
  }
  await refreshHealth();
  await refreshUsage();
  setInterval(refreshUsage, 30000);
}

window.addEventListener('error', (event) => {
  console.error(event.error || event.message);
});

boot();
