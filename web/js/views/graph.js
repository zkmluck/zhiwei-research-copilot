// 图谱页：ECharts 力导向引用网络。边分「实质引用 / 罗列式引用」，节点按角色着色。

import { api } from '../api.js';
import { state, targetDocIds } from '../state.js';
import { el, qs, clear, toast, fmtNumber } from '../dom.js';
import { ensure } from '../libs.js';

const ROLE_COLOR = {
  cornerstone: '#d9a441',
  bridge: '#5b9bd5',
  derivative: '#4f9d69',
  peripheral: '#6b7885',
  isolated: '#3f4a55',
};
const ROLE_LABEL = {
  cornerstone: '基石工作',
  bridge: '桥接工作',
  derivative: '边缘衍生',
  peripheral: '外围工作',
  isolated: '孤岛节点',
};

let chart = null;
let lastPayload = null;
let lastNodeId = '';
let observer = null;

function buildOption(payload, substantiveOnly) {
  const nodes = (payload.nodes || []).map((node) => {
    const id = node.node_id || node.doc_id;
    return {
      id,
      name: node.title || id,
      symbolSize: 12 + Math.min(46, (node.pagerank || 0) * 1200),
      itemStyle: { color: ROLE_COLOR[node.role] || ROLE_COLOR.peripheral },
      value: node,
    };
  });
  const ids = new Set(nodes.map((n) => n.id));
  const edges = (payload.edges || [])
    .map((edge) => ({
      source: edge.src_doc_id,
      target: edge.dst_doc_id || `ext:${(edge.dst_title || '').toLowerCase().replace(/[^a-z0-9\u4e00-\u9fff]+/g, '-').slice(0, 60)}`,
      value: edge,
    }))
    .filter((edge) => ids.has(edge.source) && ids.has(edge.target))
    .filter((edge) => !substantiveOnly || edge.value.substantive)
    .map((edge) => ({
      ...edge,
      lineStyle: {
        width: edge.value.substantive ? 1.6 : 0.8,
        type: edge.value.substantive ? 'solid' : 'dashed',
        opacity: 0.75,
        color: edge.value.substantive ? '#7d8896' : '#46515d',
      },
    }));

  return {
    backgroundColor: 'transparent',
    tooltip: {
      formatter: (params) => {
        if (params.dataType === 'edge') {
          const edge = params.data.value || {};
          return `${edge.intent || ''}<br/>含金量 ${fmtNumber(edge.value_score, 2)}`;
        }
        const node = params.data.value || {};
        return `${node.title || node.node_id}<br/>${ROLE_LABEL[node.role] || ''}<br/>PageRank ${fmtNumber(node.pagerank, 4)}`;
      },
    },
    series: [{
      type: 'graph',
      layout: 'force',
      roam: true,
      draggable: true,
      label: { show: true, fontSize: 10, color: '#c9d2dc', formatter: (p) => String(p.name || '').slice(0, 22) },
      force: { repulsion: 220, edgeLength: [60, 160], gravity: 0.08 },
      data: nodes,
      links: edges,
      emphasis: { focus: 'adjacency' },
    }],
  };
}

function renderDetail(nodeId) {
  const box = qs('#graph-detail');
  clear(box);
  const node = (lastPayload?.nodes || []).find((n) => (n.node_id || n.doc_id) === nodeId);
  if (!node) return;
  lastNodeId = nodeId;
  const edges = (lastPayload?.edges || []).filter((e) =>
    e.src_doc_id === nodeId || e.dst_doc_id === nodeId ||
    (e.dst_title && node.title && e.dst_title === node.title));

  box.append(el('h3', { text: node.title || nodeId }));
  box.append(el('dl', { class: 'kv' }, [
    el('dt', { text: '角色' }), el('dd', { text: ROLE_LABEL[node.role] || node.role || '—' }),
    el('dt', { text: 'PageRank' }), el('dd', { text: fmtNumber(node.pagerank, 4) }),
    el('dt', { text: '中介中心性' }), el('dd', { text: fmtNumber(node.betweenness, 4) }),
    el('dt', { text: '入/出度' }), el('dd', { text: `${node.in_degree ?? '—'} / ${node.out_degree ?? '—'}` }),
    el('dt', { text: 'k-core' }), el('dd', { text: String(node.k_core ?? '—') }),
    el('dt', { text: '在本库' }), el('dd', { text: node.in_library ? '是' : '否（外部文献）' }),
  ]));
  if (node.reason) box.append(el('p', { class: 'muted', text: node.reason }));

  box.append(el('h3', { text: `引用关系（${edges.length}）` }));
  for (const edge of edges.slice(0, 24)) {
    box.append(el('div', { class: 'claim-card ' + (edge.substantive ? 'allow' : '') }, [
      el('div', { class: 'head' }, [
        el('span', { class: 'badge', text: edge.intent || 'unknown' }),
        el('span', { text: `含金量 ${fmtNumber(edge.value_score, 2)}${edge.substantive ? ' · 实质引用' : ' · 罗列式'}` }),
      ]),
      el('div', { class: 'body', text: edge.context || edge.dst_ref || '' }),
      edge.page ? el('div', { class: 'reasons', text: `第 ${edge.page} 页` }) : '',
    ]));
  }
}

function renderListFallback(payload) {
  const box = qs('#graph-canvas');
  clear(box);
  box.append(el('p', { class: 'muted', text: '图形库未加载（可能断网），下面是同一份数据的列表视图：' }));
  const list = el('div', {});
  for (const node of payload.nodes || []) {
    list.append(el('div', { class: 'claim-card', onclick: () => renderDetail(node.node_id || node.doc_id) }, [
      el('div', { class: 'head' }, [
        el('span', { class: 'badge', text: ROLE_LABEL[node.role] || node.role || '—' }),
        el('span', { text: `PageRank ${fmtNumber(node.pagerank, 4)}` }),
      ]),
      el('div', { class: 'body', text: node.title || node.node_id }),
    ]));
  }
  box.append(list);
}

async function draw(substantiveOnly) {
  if (!lastPayload) return;
  const echarts = await ensure('echarts');
  if (!echarts) {
    renderListFallback(lastPayload);
    return;
  }
  const container = qs('#graph-canvas');
  if (!chart) {
    clear(container);
    chart = echarts.init(container, null, { renderer: 'canvas' });
    chart.on('click', (params) => {
      if (params.dataType === 'node') renderDetail(params.data.id);
    });
    window.addEventListener('resize', () => chart && chart.resize());
    // 面板从隐藏切到可见时容器尺寸才有效，用 ResizeObserver 兜住这种时序
    if (typeof ResizeObserver !== 'undefined') {
      observer = new ResizeObserver(() => { if (chart) chart.resize(); });
      observer.observe(container);
    }
  }
  chart.setOption(buildOption(lastPayload, substantiveOnly), true);
}

async function runBuild() {
  const docIds = targetDocIds();
  if (!docIds.length) {
    toast('先选至少一篇文献（左栏可多选）');
    return;
  }
  const button = qs('#graph-build');
  button.disabled = true;
  button.textContent = '构建中…';
  try {
    const classify = qs('#graph-classify').checked;
    await api.graphBuild({ doc_ids: docIds, expand_external: true, classify });
    const payload = await api.graph(docIds);
    lastPayload = payload;
    const stats = payload.metrics || {};
    const costly = classify ? ' · 引用意图由模型精判' : ' · 引用意图用离线启发式';
    qs('#graph-stats').textContent =
      `节点 ${stats.nodes ?? '—'} · 边 ${stats.edges ?? '—'} · 密度 ${fmtNumber(stats.density, 4)} · 弱连通分量 ${stats.components ?? '—'}${costly}`;
    await draw(qs('#graph-substantive').checked);
    const cornerstone = (payload.nodes || []).filter((n) => n.role === 'cornerstone');
    if (cornerstone.length) renderDetail(cornerstone[0].node_id || cornerstone[0].doc_id);
  } catch (err) {
    toast(`图谱构建失败：${err.message}`, { kind: 'error', retry: runBuild });
  } finally {
    button.disabled = false;
    button.textContent = '构建引用图谱';
  }
}

export function mount() {
  qs('#graph-build').addEventListener('click', runBuild);
  qs('#graph-substantive').addEventListener('change', (event) => draw(event.target.checked));
}

export function invalidate() {
  lastPayload = null;
  if (chart) chart.clear();
}
