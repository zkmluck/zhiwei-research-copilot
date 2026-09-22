// 后端接口封装。所有请求走同源 /api，SSE 用 fetch + ReadableStream 手解。

const BASE = '';

async function request(path, { method = 'GET', body, formData } = {}) {
  const init = { method, headers: {} };
  if (formData) {
    init.body = formData;
  } else if (body !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
  }
  const response = await fetch(`${BASE}${path}`, init);
  const text = await response.text();
  let data = null;
  if (text) {
    try { data = JSON.parse(text); } catch (err) { data = { raw: text }; }
  }
  if (!response.ok) {
    const message = (data && data.error && data.error.message) || `${response.status} ${response.statusText}`;
    const error = new Error(message);
    error.status = response.status;
    error.payload = data;
    throw error;
  }
  return data;
}

function get(path) { return request(path); }
function post(path, body) { return request(path, { method: 'POST', body: body || {} }); }

/** 解析 SSE 流。onEvent(event, payload)。 */
export async function stream(path, body, onEvent, signal) {
  const response = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
    signal,
  });
  if (!response.ok || !response.body) {
    throw new Error(`流式接口失败：${response.status}`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut = buffer.indexOf('\n\n');
    while (cut >= 0) {
      const chunk = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      handleChunk(chunk, onEvent);
      cut = buffer.indexOf('\n\n');
    }
  }
  if (buffer.trim()) handleChunk(buffer, onEvent);
}

function handleChunk(chunk, onEvent) {
  let event = 'message';
  const dataLines = [];
  for (const line of chunk.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return;
  let payload = null;
  try { payload = JSON.parse(dataLines.join('\n')); } catch (err) { payload = { raw: dataLines.join('\n') }; }
  onEvent(event, payload);
}

export const api = {
  health: () => get('/api/health'),
  usage: () => get('/api/usage'),
  papers: () => get('/api/papers'),
  paper: (docId) => get(`/api/papers/${encodeURIComponent(docId)}`),
  removePaper: (docId) => request(`/api/papers/${encodeURIComponent(docId)}`, { method: 'DELETE' }),
  resolveDedup: (docId, action) => post(`/api/papers/${encodeURIComponent(docId)}/dedup/resolve`, { action }),
  versions: (docId) => get(`/api/papers/${encodeURIComponent(docId)}/versions`),
  diff: (docId, against) => get(`/api/papers/${encodeURIComponent(docId)}/diff?against=${encodeURIComponent(against)}`),
  page: (docId, page) => get(`/api/papers/${encodeURIComponent(docId)}/page/${page}`),
  fileUrl: (docId) => `/api/papers/${encodeURIComponent(docId)}/file`,
  references: (docId) => get(`/api/papers/${encodeURIComponent(docId)}/references`),
  xref: (docId, index) => get(`/api/papers/${encodeURIComponent(docId)}/xref?index=${index}`),
  translatePage: (docId, page, to = 'zh') => get(`/api/papers/${encodeURIComponent(docId)}/translate?to=${to}&page=${page}`),
  translateSegment: (text, to = 'zh') => post('/api/translate/segment', { text, to }),
  qaStream: (body, onEvent, signal) => stream('/api/qa/stream', body, onEvent, signal),
  qa: (body) => post('/api/qa', body),
  extract: (body) => post('/api/extract', body),
  compare: (body) => post('/api/compare', body),
  graphBuild: (body) => post('/api/graph/build', body),
  graph: (docIds) => get(`/api/graph?doc_ids=${encodeURIComponent((docIds || []).join(','))}`),
  graphCitations: (docId) => get(`/api/graph/citations?doc_id=${encodeURIComponent(docId)}`),
  survey: (docIds, topic) => get(`/api/survey?doc_ids=${encodeURIComponent((docIds || []).join(','))}&topic=${encodeURIComponent(topic || '')}`),
  futureWork: (docIds, check = true) => get(`/api/future-work?doc_ids=${encodeURIComponent((docIds || []).join(','))}&check=${check ? 'true' : 'false'}`),
  discoverSearch: (query, limit = 20) => post('/api/discover/search', { query, limit }),
  discoverIngest: (identifier, source = 'arxiv') => post('/api/discover/ingest', { identifier, source }),
  discoverReferences: (docId) => post('/api/discover/references', { doc_id: docId }),
  recommend: (docId, days = 30) => get(`/api/recommend?doc_id=${encodeURIComponent(docId || '')}&days=${days}`),
  feedback: (docId, action) => post('/api/feedback', { doc_id: docId, action }),
  trackStatus: () => get('/api/track/status'),
  trackRun: (topics, days = 7) => post('/api/track/run', { topics, since_days: days }),
  outline: (idea, docIds, venue = 'generic') => post('/api/writing/outline', { idea, doc_ids: docIds, venue }),
  diagram: (kind, spec, extra = {}) => post('/api/writing/diagram', { kind, spec, ...extra }),
  figure: (body) => post('/api/writing/figure', body),
  artifacts: () => get('/api/writing/artifacts'),
  agentTools: () => get('/api/agent/tools'),
  agentPlan: (goal, docIds) => post('/api/agent/plan', { goal, doc_ids: docIds }),
  agentRun: (body, onEvent, signal) => stream('/api/agent/run', body, onEvent, signal),
};
