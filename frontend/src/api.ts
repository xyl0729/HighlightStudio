import type {
  AnalysisMode,
  AspectRatio,
  HealthResponse,
  JobData,
  Segment,
} from './types';

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, '') ?? '';

export function resolveApiUrl(path: string | null | undefined): string {
  if (!path) return '';
  if (/^https?:\/\//i.test(path)) return path;
  return `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`;
}

async function parseError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (typeof data?.detail === 'string') return data.detail;
    if (Array.isArray(data?.detail) && data.detail[0]?.msg) {
      return String(data.detail[0].msg);
    }
    if (typeof data?.error?.message === 'string') return data.error.message;
  } catch {
    // ignore JSON parse failure
  }
  return `请求失败 (HTTP ${res.status})`;
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const res = await fetch(resolveApiUrl('/api/health'), { signal });
  if (!res.ok) {
    throw new Error(await parseError(res));
  }
  return (await res.json()) as HealthResponse;
}

export function createJob(
  file: File,
  aspectRatio: AspectRatio,
  mode: AnalysisMode,
  onUploadProgress?: (ratio: number) => void,
): Promise<JobData> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', resolveApiUrl('/api/jobs'));

    xhr.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable && event.total > 0 && onUploadProgress) {
        onUploadProgress(Math.min(1, event.loaded / event.total));
      }
    });

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as JobData);
        } catch {
          reject(new Error('后端返回的数据格式无效。'));
        }
        return;
      }
      try {
        const body = JSON.parse(xhr.responseText);
        const msg =
          typeof body?.detail === 'string'
            ? body.detail
            : typeof body?.error?.message === 'string'
              ? body.error.message
              : `上传失败 (HTTP ${xhr.status})`;
        reject(new Error(msg));
      } catch {
        reject(new Error(`上传失败 (HTTP ${xhr.status})`));
      }
    });

    xhr.addEventListener('error', () => {
      reject(new Error('无法连接到本地后端服务，请确认 127.0.0.1:8000 已启动。'));
    });

    xhr.addEventListener('abort', () => {
      reject(new Error('上传已取消。'));
    });

    const form = new FormData();
    form.append('file', file);
    form.append('aspect_ratio', aspectRatio);
    form.append('mode', mode);
    xhr.send(form);
  });
}

export async function listJobs(signal?: AbortSignal): Promise<JobData[]> {
  const res = await fetch(resolveApiUrl('/api/jobs'), { signal, cache: 'no-store' });
  if (!res.ok) throw new Error(await parseError(res));
  return (await res.json()) as JobData[];
}

export async function getJob(jobId: string, signal?: AbortSignal): Promise<JobData> {
  const res = await fetch(resolveApiUrl(`/api/jobs/${jobId}`), {
    signal,
    cache: 'no-store',
  });
  if (!res.ok) {
    throw new Error(await parseError(res));
  }
  return (await res.json()) as JobData;
}

export async function updateSegments(jobId: string, segments: Segment[]): Promise<JobData> {
  const res = await fetch(resolveApiUrl(`/api/jobs/${jobId}/segments`), {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ segments }),
  });
  if (!res.ok) {
    throw new Error(await parseError(res));
  }
  return (await res.json()) as JobData;
}

export async function requestExport(jobId: string): Promise<JobData> {
  const res = await fetch(resolveApiUrl(`/api/jobs/${jobId}/export`), {
    method: 'POST',
  });
  if (!res.ok) {
    throw new Error(await parseError(res));
  }
  return (await res.json()) as JobData;
}

export async function deleteJob(jobId: string): Promise<void> {
  const res = await fetch(resolveApiUrl(`/api/jobs/${jobId}`), {
    method: 'DELETE',
  });
  if (res.status === 204 || res.status === 404) return;
  throw new Error(await parseError(res));
}
