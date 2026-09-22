// 极简 DOM 助手：不引框架，够用就行。

export function qs(selector, root = document) {
  return root.querySelector(selector);
}

export function qsa(selector, root = document) {
  return Array.from(root.querySelectorAll(selector));
}

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'html') node.innerHTML = value;
    else if (key === 'text') node.textContent = value;
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? '' : String(value));
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function clear(node) {
  if (node) node.replaceChildren();
}

export function escapeHtml(text) {
  return String(text ?? '').replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[ch]);
}

let toastTimer = null;

export function toast(message, { kind = 'info', retry = null, timeout = 6000 } = {}) {
  const box = qs('#toast');
  if (!box) return;
  box.className = `toast ${kind === 'error' ? 'error' : ''}`;
  box.replaceChildren(document.createTextNode(message));
  if (retry) {
    const button = el('button', { class: 'ghost small retry', type: 'button', text: '重试', onclick: () => { hideToast(); retry(); } });
    box.append(button);
  }
  box.hidden = false;
  clearTimeout(toastTimer);
  if (timeout) toastTimer = setTimeout(hideToast, timeout);
}

export function hideToast() {
  const box = qs('#toast');
  if (box) box.hidden = true;
}

export function spinner() {
  return el('span', { class: 'spinner' });
}

export function fmtPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return `${Math.round(Number(value) * 100)}%`;
}

export function fmtNumber(value, digits = 4) {
  if (value === null || value === undefined || value === '') return '—';
  const num = Number(value);
  if (Number.isNaN(num)) return String(value);
  return num.toFixed(digits);
}

// Markdown 渲染：有 marked 就用它，没有就退化成一个只处理标题/列表的安全渲染
export function renderMarkdown(text) {
  const source = String(text ?? '');
  if (window.marked && typeof window.marked.parse === 'function') {
    try {
      return window.marked.parse(source, { breaks: false, gfm: true });
    } catch (err) {
      /* 落到下面的降级渲染 */
    }
  }
  const lines = source.split('\n');
  const out = [];
  let inList = false;
  for (const raw of lines) {
    const line = raw.trimEnd();
    if (/^\s*[-*]\s+/.test(line)) {
      if (!inList) { out.push('<ul>'); inList = true; }
      out.push(`<li>${escapeHtml(line.replace(/^\s*[-*]\s+/, ''))}</li>`);
      continue;
    }
    if (inList) { out.push('</ul>'); inList = false; }
    if (line.startsWith('### ')) out.push(`<h3>${escapeHtml(line.slice(4))}</h3>`);
    else if (line.startsWith('## ')) out.push(`<h2>${escapeHtml(line.slice(3))}</h2>`);
    else if (line.startsWith('# ')) out.push(`<h1>${escapeHtml(line.slice(2))}</h1>`);
    else if (line.startsWith('> ')) out.push(`<blockquote>${escapeHtml(line.slice(2))}</blockquote>`);
    else if (line.trim() === '') out.push('');
    else out.push(`<p>${escapeHtml(line)}</p>`);
  }
  if (inList) out.push('</ul>');
  return out.join('\n');
}
