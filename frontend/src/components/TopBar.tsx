import { RefreshCw, Trash2 } from 'lucide-react';
import type { ReactNode } from 'react';
import type { HealthResponse, JobData } from '../types';

interface TopBarProps {
  health: HealthResponse | null;
  healthError: string | null;
  onRefreshHealth: () => void;
  job: JobData | null;
  fileName: string | null;
  busyCleaning: boolean;
  queue: ReactNode;
  onDeleteJob: () => void;
}

export function TopBar({
  health,
  healthError,
  onRefreshHealth,
  job,
  fileName,
  busyCleaning,
  queue,
  onDeleteJob,
}: TopBarProps) {
  const isJobActive =
    job !== null &&
    (job.status === 'queued' ||
      job.status === 'preparing' ||
      job.status === 'analyzing' ||
      job.status === 'exporting');

  const engineLabel =
    job?.analysis_engine === 'multimodal-v1' ? '多模态高光引擎' : '本地推理';
  const finalReady = health?.final_engine_configured;

  return (
    <header className="topbar">
      <div className="topbar-left">
        <div className="brand">
          <span className="brand-glyph" aria-hidden="true" />
          <span>HIGHLIGHT STUDIO</span>
        </div>

        <span className="v-sep" />

        <div
          className="health-inline"
          title={
            health
              ? (finalReady ? '多模态高光引擎已就绪' : '推理模型未配置')
              : healthError || '正在检测本地服务'
          }
        >
          <span
            className={`status-dot ${
              health ? (finalReady ? 'ok' : 'warn') : 'err'
            }`}
          />
          <span>
            {health
              ? finalReady
                ? '本地服务在线 · 多模态引擎就绪'
                : '本地服务在线 · 模型未配置'
              : '未连接后端'}
          </span>
          {!health && (
            <button
              type="button"
              className="btn-ghost btn-icon"
              style={{ width: 20, height: 20 }}
              onClick={onRefreshHealth}
              title="重新检测"
            >
              <RefreshCw size={11} />
            </button>
          )}
        </div>
      </div>

      <div className="topbar-center">
        {job && (
          <div className="job-meta-strip">
            <span className="job-meta-name" title={fileName ?? job.id}>
              {fileName ?? job.id.slice(0, 8)}
            </span>
            {job.media && (
              <>
                <span className="v-sep" />
                <span className="meta-datum mono">
                  {job.media.width}×{job.media.height} · {job.media.fps}fps
                </span>
              </>
            )}
            <span className="v-sep" />
            <span className="meta-datum mono">{job.aspect_ratio}</span>
            <span className="v-sep" />
            <span className="meta-datum accent">{engineLabel}</span>
          </div>
        )}
      </div>

      <div className="topbar-right">
        {queue}
        {job && (
          <button
            type="button"
            className="btn btn-sm btn-danger-ghost"
            disabled={isJobActive || busyCleaning}
            onClick={onDeleteJob}
            title="清理当前任务缓存"
          >
            <Trash2 size={13} />
            <span>{busyCleaning ? '清理中...' : '清理缓存'}</span>
          </button>
        )}
      </div>
    </header>
  );
}
