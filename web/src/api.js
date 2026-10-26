export class ApiError extends Error {
  constructor(message, status, data) {
    super(message);
    this.status = status;
    this.data = data;
  }
}

let onUnauthorized = () => {};
export const setUnauthorizedHandler = (fn) => {
  onUnauthorized = fn;
};

async function request(path, { method = 'GET', json, form, signal } = {}) {
  const headers = { 'X-Requested-With': 'cutis-web' };
  let body;
  if (json !== undefined) {
    headers['Content-Type'] = 'application/json';
    body = JSON.stringify(json);
  } else if (form) {
    body = form;
  }
  let res;
  try {
    res = await fetch(`/api${path}`, { method, headers, body, credentials: 'same-origin', signal });
  } catch (e) {
    if (e?.name === 'AbortError') throw e;
    throw new ApiError('Cannot reach the server. Please check your connection and try again.', 0);
  }
  let data = null;
  try {
    data = await res.json();
  } catch {
    /* empty or non-JSON body */
  }
  if (!res.ok) {
    if (res.status === 401 && !path.startsWith('/auth/')) onUnauthorized();
    throw new ApiError(data?.message || `Something went wrong (${res.status}).`, res.status, data);
  }
  return data;
}

export const api = {
  get: (p, o) => request(p, o),
  post: (p, json, o) => request(p, { method: 'POST', json: json ?? {}, ...o }),
  upload: (p, form) => request(p, { method: 'POST', form }),
  del: (p) => request(p, { method: 'DELETE' }),
};
