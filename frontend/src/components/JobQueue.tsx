import { useRef } from 'react';
import { FolderPlus, ListOrdered } from 'lucide-react';
import type { JobStatus, QueuedVideo } from '../types';

const statusLabels: Record<JobStatus, string> = {
  queued: '排队中',
  preparing: '转码中',
  analyzing: '分析中',
  ready: '可编辑',
  exporting: '导出中',
  exported: '已导出',
  failed: '分析失败',
};

interface JobQueueProps {
  items: QueuedVideo[];
  selectedId: string | null;
  busy: boolean;
  hasUnsavedEdits: boolean;
  onSelect: (id: string) => void;
  onAdd: () => void;
}

export function JobQueue({ items, selectedId, busy, hasUnsavedEdits, onSelect, onAdd }: JobQueueProps) {
  const details = useRef<HTMLDetailsElement>(null);
  if (!items.length) return null;
  const disabled = busy || hasUnsavedEdits;
  const reason = hasUnsavedEdits ? '先保存当前片段再切换任务' : undefined;

  return (
    <>
      <details className="job-queue" ref={details}>
        <summary className="btn btn-sm btn-ghost">
          <ListOrdered size={13} />
          <span>任务队列 ({items.length})</span>
        </summary>
        <div className="job-queue-panel">
          {reason && <p className="job-queue-note">{reason}</p>}
          <ul aria-label="任务队列">
            {items.map(({ job, name }) => (
              <li key={job.id}>
                <button
                  type="button"
                  className="job-queue-row"
                  aria-current={job.id === selectedId ? 'true' : undefined}
                  disabled={disabled}
                  onClick={() => {
                    if (details.current) details.current.open = false;
                    onSelect(job.id);
                  }}
                >
                  <span className="queue-file-name" title={name}>{name}</span>
                  <span className="mono">{job.aspect_ratio}</span>
                  <span className="job-queue-status" data-status={job.status}>
                    {statusLabels[job.status]}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      </details>
      {selectedId && (
        <button type="button" className="btn btn-sm btn-ghost" disabled={disabled} onClick={onAdd} title={reason}>
          <FolderPlus size={13} />
          <span>添加视频</span>
        </button>
      )}
    </>
  );
}
