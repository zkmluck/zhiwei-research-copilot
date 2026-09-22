// 写作页：Idea→框架、架构图三种格式、CSV→图+Caption。

import { api } from '../api.js';
import { targetDocIds } from '../state.js';
import { el, qs, clear, toast, renderMarkdown } from '../dom.js';
import { ensure } from '../libs.js';

function outlineToMarkdown(payload) {
  const parts = [];
  if (payload.abstract) parts.push('## Abstract', payload.abstract, '');
  if (payload.introduction) parts.push('## 1 Introduction', payload.introduction, '');
  if (payload.related_work) parts.push('## 2 Related Work', payload.related_work, '');
  if ((payload.references || []).length) {
    parts.push('## References');
    for (const ref of payload.references) parts.push(`${ref.index}. ${ref.raw || ref.title || ''}`);
  }
  if ((payload.unsupported || []).length) {
    parts.push('', '## 被闸门拦下的句子（不进入正文）');
    for (const item of payload.unsupported) parts.push(`- ${item.text}（${item.reason || '无证据'}）`);
  }
  const stats = payload.stats || {};
  if (stats.candidate_claims !== undefined) {
    parts.push('', `> 候选结论 ${stats.candidate_claims} 条，放行 ${stats.allowed} 条，拦下 ${stats.blocked} 条；引用 ${stats.references} 篇（全部来自你的文献库）`);
  }
  return parts.join('\n');
}

async function runOutline() {
  const idea = qs('#writing-idea').value.trim();
  if (!idea) {
    toast('先写一句想法');
    return;
  }
  const box = qs('#writing-outline');
  box.innerHTML = '<p class="muted">生成中…</p>';
  try {
    const payload = await api.outline(idea, targetDocIds(), qs('#writing-venue').value);
    box.innerHTML = renderMarkdown(outlineToMarkdown(payload));
  } catch (err) {
    box.innerHTML = '';
    box.append(el('p', { class: 'muted', text: `生成失败：${err.message}` }));
    toast(`框架生成失败：${err.message}`, { kind: 'error' });
  }
}

const DEMO_SPEC = {
  nodes: [
    { id: 'pdf', label: 'PDF 解析' },
    { id: 'chunk', label: '切块与索引' },
    { id: 'retrieval', label: '混合检索' },
    { id: 'gate', label: '主张闸门' },
  ],
  edges: [
    { from: 'pdf', to: 'chunk' },
    { from: 'chunk', to: 'retrieval' },
    { from: 'retrieval', to: 'gate' },
  ],
};

async function buildDiagram(fromGraph) {
  const kind = qs('#diagram-kind').value;
  const codeBox = qs('#diagram-code');
  const preview = qs('#diagram-preview');
  codeBox.textContent = '生成中…';
  clear(preview);
  try {
    const extra = fromGraph ? { from_graph: true, doc_ids: targetDocIds() } : {};
    const payload = await api.diagram(kind, fromGraph ? null : DEMO_SPEC, extra);
    codeBox.textContent = payload.code || '（无代码）';
    if (payload.kind === 'mermaid') {
      const mermaid = await ensure('mermaid');
      if (mermaid) {
        mermaid.initialize({ startOnLoad: false, theme: 'dark' });
        const id = `mmd-${Date.now()}`;
        const rendered = await mermaid.render(id, payload.code);
        preview.innerHTML = rendered.svg;
      } else {
        preview.append(el('p', { class: 'muted', text: 'Mermaid 未加载（可能断网），下面的代码可直接复制使用' }));
      }
    } else {
      preview.append(el('p', { class: 'muted', text: payload.render_hint || '该格式需要外部工具渲染，代码已生成' }));
    }
  } catch (err) {
    codeBox.textContent = '';
    preview.append(el('p', { class: 'muted', text: `生成失败：${err.message}` }));
    toast(`图表生成失败：${err.message}`, { kind: 'error' });
  }
}

async function runFigure() {
  const csv = qs('#figure-csv').value.trim();
  if (!csv) {
    toast('先把 CSV 粘进来');
    return;
  }
  const box = qs('#figure-output');
  box.replaceChildren(el('p', { class: 'muted', text: '出图中…' }));
  try {
    const payload = await api.figure({
      csv_text: csv,
      chart: qs('#figure-chart').value,
      title: qs('#figure-title').value,
    });
    clear(box);
    if (payload.image_url) box.append(el('img', { src: payload.image_url, alt: payload.title || 'figure' }));
    if (payload.caption) box.append(el('p', { class: 'caption', text: payload.caption }));
    if (payload.script) {
      box.append(el('details', {}, [
        el('summary', { text: '查看 matplotlib 脚本' }),
        el('pre', { class: 'code', text: payload.script }),
      ]));
    }
  } catch (err) {
    box.replaceChildren(el('p', { class: 'muted', text: `出图失败：${err.message}` }));
    toast(`出图失败：${err.message}`, { kind: 'error' });
  }
}

export function mount() {
  qs('#writing-run').addEventListener('click', runOutline);
  qs('#diagram-from-graph').addEventListener('click', () => buildDiagram(true));
  qs('#diagram-kind').addEventListener('change', () => buildDiagram(false));
  qs('#figure-run').addEventListener('click', runFigure);
  buildDiagram(false);
}
