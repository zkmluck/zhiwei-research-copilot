// 阅读页：标题树 + PDF 渲染 + 块级文本层（可划词、可点 [11]）+ 中英对照。

import { api } from '../api.js';
import { state, update } from '../state.js';
import { el, qs, clear, toast, renderMarkdown } from '../dom.js';
import { ensure } from '../libs.js';

let pdfDoc = null;
let pdfjsLib = null;
let currentDetail = null;
let currentPage = 1;
let pageBlocks = [];
let loading = false;
let textOnly = false;

export function currentDocId() {
  return state.activeDocId;
}

async function loadPdf(docId) {
  pdfjsLib = pdfjsLib || await ensure('pdfjs');
  if (!pdfjsLib) {
    toast('PDF 渲染库没加载出来（可能断网）。目录与检索依然可用。', { kind: 'error' });
    return null;
  }
  const pdfjs = pdfjsLib.default || pdfjsLib;
  if (pdfjs.GlobalWorkerOptions) {
    pdfjs.GlobalWorkerOptions.workerSrc = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@4.7.76/build/pdf.worker.min.mjs';
  }
  const task = pdfjs.getDocument({ url: api.fileUrl(docId) });
  return task.promise;
}

async function renderPage(pageNo) {
  const stage = qs('#reader-stage');
  clear(qs('#pdf-overlay'));
  if (!pdfDoc) {
    renderTextMode();
    return;
  }
  loading = true;
  try {
    const page = await pdfDoc.getPage(pageNo);
    const base = page.getViewport({ scale: 1 });
    const scale = Number(state.zoom || 1) * Math.min(2.2, Math.max(1, (stage.clientWidth - 48) / base.width));
    const viewport = page.getViewport({ scale });
    const canvas = qs('#pdf-canvas');
    const context = canvas.getContext('2d');
    canvas.width = viewport.width;
    canvas.height = viewport.height;
    canvas.hidden = false;
    qs('#reader-empty').hidden = true;
    await page.render({ canvasContext: context, viewport }).promise;
    drawOverlay(viewport);
  } catch (err) {
    // PDF.js 挂掉不该让人读不了论文：退回结构化文本视图
    textOnly = true;
    renderTextMode();
  } finally {
    loading = false;
  }
}

/** 没有 PDF 渲染能力时的降级阅读：按块展示本页结构，锚点依然可用。 */
function renderTextMode() {
  const box = qs('#reader-empty');
  const canvas = qs('#pdf-canvas');
  canvas.hidden = true;
  box.hidden = false;
  box.replaceChildren();
  box.append(el('p', { class: 'muted', text: `第 ${currentPage} 页 · 结构化文本视图（PDF 渲染不可用时的降级，内容与锚点不受影响）` }));
  if (!pageBlocks.length) {
    box.append(el('p', { class: 'muted', text: '这一页没有解析出内容' }));
    return;
  }
  for (const block of pageBlocks) {
    const node = el('div', { class: 'text-block', dataset: { kind: block.kind } }, [block.text]);
    node.addEventListener('click', (event) => {
      const mark = event.target.textContent && event.target.textContent.match(/\[\d{1,3}\]/);
      if (mark) jumpToReference(Number(mark[0].slice(1, -1)));
    });
    box.append(node);
  }
  if (state.highlight && state.highlight.page === currentPage && state.highlight.quote) {
    const needle = state.highlight.quote.slice(0, 40);
    const hit = Array.from(box.querySelectorAll('.text-block')).find((n) => n.textContent.includes(needle));
    if (hit) {
      hit.classList.add('hl');
      hit.scrollIntoView({ block: 'center' });
    }
  }
}

function drawOverlay(viewport) {
  const overlay = qs('#pdf-overlay');
  const canvas = qs('#pdf-canvas');
  clear(overlay);
  overlay.style.width = `${canvas.width}px`;
  overlay.style.height = `${canvas.height}px`;
  overlay.style.left = `${canvas.offsetLeft}px`;
  overlay.style.top = `${canvas.offsetTop}px`;

  for (const block of pageBlocks) {
    const box = block.bbox;
    if (!box) continue;
    const node = el('div', {
      class: 'block-hotspot',
      dataset: { kind: block.kind },
      title: block.text.slice(0, 160),
      style: [
        `left:${box.x0 * viewport.scale}px`,
        `top:${box.y0 * viewport.scale}px`,
        `width:${Math.max(8, (box.x1 - box.x0) * viewport.scale)}px`,
        `height:${Math.max(8, (box.y1 - box.y0) * viewport.scale)}px`,
      ].join(';'),
    }, [block.text]);
    node.addEventListener('click', (event) => {
      const mark = event.target.textContent && event.target.textContent.match(/\[\d{1,3}\]/);
      if (mark) jumpToReference(Number(mark[0].slice(1, -1)));
    });
    overlay.append(node);
  }
  if (state.highlight && state.highlight.page === currentPage) {
    markHighlight(state.highlight, viewport);
  }
}

function markHighlight(highlight, viewport) {
  const overlay = qs('#pdf-overlay');
  const target = pageBlocks.find((b) => b.text && highlight.quote && b.text.includes(highlight.quote.slice(0, 40)));
  if (!target || !target.bbox) return;
  const box = target.bbox;
  overlay.append(el('div', {
    class: 'hl',
    style: [
      `left:${box.x0 * viewport.scale}px`,
      `top:${box.y0 * viewport.scale}px`,
      `width:${Math.max(8, (box.x1 - box.x0) * viewport.scale)}px`,
      `height:${Math.max(8, (box.y1 - box.y0) * viewport.scale)}px`,
    ].join(';'),
  }));
}

async function gotoPage(pageNo) {
  if (!currentDetail) return;
  const total = currentDetail.stats.pages || 1;
  currentPage = Math.min(Math.max(1, pageNo), total);
  update({ page: currentPage });
  qs('#reader-page').textContent = String(currentPage);
  qs('#reader-pages').textContent = String(total);
  qs('#reader-jump').value = String(currentPage);
  try {
    const data = await api.page(state.activeDocId, currentPage);
    pageBlocks = data.blocks || [];
  } catch (err) {
    pageBlocks = [];
    toast(`第 ${currentPage} 页结构读取失败：${err.message}`, { kind: 'error' });
  }
  await renderPage(currentPage);
}

function renderOutline(nodes, depth = 0) {
  const box = qs('#reader-outline');
  for (const node of nodes || []) {
    const button = el('button', {
      type: 'button',
      text: `${'　'.repeat(depth)}${node.number ? node.number + ' ' : ''}${node.title}`,
      onclick: () => gotoPage(node.page),
    });
    box.append(button);
    if (node.children && node.children.length) renderOutline(node.children, depth + 1);
  }
}

async function renderReferences() {
  const box = qs('#reader-refs');
  clear(box);
  try {
    const data = await api.references(state.activeDocId);
    for (const ref of data.references || []) {
      const item = el('li', {
        class: ref.in_library ? 'in-library' : '',
        text: `${ref.resolved_title || ref.raw.slice(0, 90)}${ref.year ? ` (${ref.year})` : ''}`,
        title: ref.raw,
        onclick: () => jumpFromReference(ref.index),
      });
      box.append(item);
    }
  } catch (err) {
    box.append(el('li', { class: 'muted', text: `参考文献读取失败：${err.message}` }));
  }
}

async function jumpToReference(index) {
  try {
    const data = await api.xref(state.activeDocId, index);
    const first = (data.in_text || [])[0];
    if (!first) {
      const target = qs('#reader-refs').children[index - 1];
      if (target) target.scrollIntoView({ block: 'center' });
      toast(`正文里没找到 [${index}] 的出处`);
      return;
    }
    update({ highlight: { page: first.page, quote: first.quote } });
    await gotoPage(first.page);
    const inTextNode = document.querySelector(`#reader-refs li:nth-child(${index})`);
    if (inTextNode) inTextNode.classList.add('in-library');
    toast(`已跳到第 ${first.page} 页正文出处；再次点击参考文献可回到条目`);
  } catch (err) {
    toast(`引用跳转失败：${err.message}`, { kind: 'error' });
  }
}

async function jumpFromReference(index) {
  const target = qs('#reader-refs').children[index - 1];
  if (target) target.scrollIntoView({ block: 'center' });
  await jumpToReference(index);
}

async function translatePage() {
  const box = qs('#reader-translation');
  box.replaceChildren(el('p', { class: 'muted', text: '翻译中…' }));
  try {
    const data = await api.translatePage(state.activeDocId, currentPage, 'zh');
    clear(box);
    if (!data.segments || !data.segments.length) {
      box.append(el('p', { class: 'muted', text: '这一页没有可翻译的正文（可能是图表页）' }));
      return;
    }
    for (const seg of data.segments) {
      const node = el('div', { class: 'seg' }, [
        el('div', { class: 'src', text: seg.src }),
        el('div', { class: 'dst', text: seg.dst || '（未翻译）' }),
      ]);
      node.addEventListener('click', () => {
        if (seg.bbox) {
          update({ highlight: { page: currentPage, quote: seg.src.slice(0, 60) } });
          renderPage(currentPage);
        }
      });
      box.append(node);
    }
  } catch (err) {
    box.replaceChildren(el('p', { class: 'muted', text: `翻译失败：${err.message}` }));
  }
}

export async function open(docId, page = 1) {
  if (!docId) return;
  update({ activeDocId: docId, highlight: null });
  textOnly = false;
  clear(qs('#reader-outline'));
  clear(qs('#reader-refs'));
  clear(qs('#reader-translation'));
  try {
    currentDetail = await api.paper(docId);
  } catch (err) {
    toast(`打开文献失败：${err.message}`, { kind: 'error' });
    return;
  }
  qs('#reader-title').textContent = currentDetail.record.title || currentDetail.record.file_name || docId;
  renderOutline(currentDetail.outline || []);
  renderReferences();
  try {
    pdfDoc = await loadPdf(docId);
  } catch (err) {
    // PDF 取不到（mock 后端、断网、或后端没存原件）时退回结构化文本阅读
    pdfDoc = null;
    textOnly = true;
  }
  await gotoPage(page);
}

export function highlight(docId, page, quote) {
  update({ highlight: { page, quote } });
  if (state.activeDocId === docId) {
    if (pdfDoc) renderPage(currentPage);
    else renderTextMode();
  }
}

export function mount() {
  qs('#reader-prev').addEventListener('click', () => gotoPage(currentPage - 1));
  qs('#reader-next').addEventListener('click', () => gotoPage(currentPage + 1));
  qs('#reader-jump').addEventListener('change', (event) => gotoPage(Number(event.target.value) || 1));
  qs('#reader-zoom').addEventListener('change', (event) => {
    update({ zoom: Number(event.target.value) || 1 });
    renderPage(currentPage);
  });
  qs('#reader-translate').addEventListener('click', translatePage);

  // 划词翻译浮层
  document.addEventListener('mouseup', async (event) => {
    if (!event.target.closest('#pdf-overlay')) return;
    const selection = window.getSelection();
    const text = selection ? String(selection).trim() : '';
    if (text.length < 4 || text.length > 600) return;
    try {
      const data = await api.translateSegment(text, 'zh');
      const bubble = el('div', {
        class: 'sel-bubble',
        style: `left:${event.clientX + 8}px;top:${event.clientY + 8}px`,
        text: data.translated || '（无译文）',
      });
      document.body.append(bubble);
      setTimeout(() => bubble.remove(), 9000);
      bubble.addEventListener('click', () => bubble.remove());
    } catch (err) {
      toast(`划词翻译失败：${err.message}`, { kind: 'error' });
    }
  });
}
