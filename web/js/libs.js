// CDN 依赖的按需加载与降级。评审环境可能断网，所以每一项都要能"没有也能用"。

const SOURCES = {
  echarts: { url: 'https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js', global: 'echarts' },
  marked: { url: 'https://cdn.jsdelivr.net/npm/marked@14.1.3/marked.min.js', global: 'marked' },
  mermaid: { url: 'https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js', global: 'mermaid' },
  katex: { url: 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js', global: 'katex' },
  pdfjs: { module: 'https://cdn.jsdelivr.net/npm/pdfjs-dist@4.7.76/build/pdf.min.mjs' },
};

const pending = new Map();
export const failures = new Set();

function loadScript(url) {
  return new Promise((resolve, reject) => {
    const tag = document.createElement('script');
    tag.src = url;
    tag.async = true;
    tag.onload = () => resolve();
    tag.onerror = () => reject(new Error(`加载失败：${url}`));
    document.head.append(tag);
  });
}

/** 确保某个 CDN 库可用；失败时不抛给调用方，而是返回 null 并记录降级。 */
export async function ensure(name) {
  const spec = SOURCES[name];
  if (!spec) return null;
  if (spec.global && window[spec.global]) return window[spec.global];
  if (pending.has(name)) return pending.get(name);

  const job = (async () => {
    try {
      if (spec.module) {
        const mod = await import(/* @vite-ignore */ spec.module);
        return mod;
      }
      await loadScript(spec.url);
      return window[spec.global] || null;
    } catch (err) {
      failures.add(name);
      markOffline();
      return null;
    }
  })();
  pending.set(name, job);
  return job;
}

let offlineMarked = false;

function markOffline() {
  if (offlineMarked) return;
  offlineMarked = true;
  const warn = document.getElementById('cdn-warn');
  if (warn) warn.hidden = false;
}

/** 提前把不重的库拉起来，失败也不影响首屏。 */
export function warmup() {
  ['marked', 'echarts'].forEach((name) => { ensure(name); });
}
