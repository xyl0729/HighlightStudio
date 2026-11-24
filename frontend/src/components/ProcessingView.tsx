import { useEffect, useState } from 'react';
import { AlertCircle, Check, Loader2, RotateCcw } from 'lucide-react';
import type { JobData } from '../types';
import { formatClock } from '../utils/time';

interface ProcessingViewProps {
  job: JobData;
  fileName: string | null;
  startedAt: number;
  onBack: () => void;
}

const STEPS = [
  { key: 'queued', title: '任务排队', desc: '单进程工作队列已接收素材' },
  { key: 'preparing', title: '标准化转码', desc: '30 FPS · 最长边 ≤ 1280px · 对齐音轨' },
  { key: 'analyzing', title: '高光与裁剪轨迹', desc: '在原视频上运行多模态推理与构图后处理' },
];

function stepIndex(status: JobData['status']): number {
  if (status === 'queued') return 0;
  if (status === 'preparing') return 1;
  if (status === 'analyzing') return 2;
  return 3;
}

export function ProcessingView({ job, fileName, startedAt, onBack }: ProcessingViewProps) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (job.status === 'failed') return;
    const update = () => setElapsed(Math.max(0, (Date.now() - startedAt) / 1000));
    update();
    const timer = window.setInterval(update, 500);
    return () => window.clearInterval(timer);
  }, [job.status, startedAt]);

  const activeIdx = stepIndex(job.status);
  const failed = job.status === 'failed';

  return (
    <main className="processing-page">
      <div className="processing-panel">
        <div className="processing-head">
          <div style={{ minWidth: 0 }}>
            <h2 className="processing-title">{failed ? '分析失败' : '正在处理素材'}</h2>
            <p className="processing-sub mono">
              {fileName ?? job.id.slice(0, 8)} · {job.aspect_ratio} ·{' '}
              多模态高光引擎
            </p>
          </div>
          {!failed && (
            <span className="mono" style={{ fontSize: 12, color: 'var(--ink-secondary)' }}>
              {formatClock(elapsed)}
            </span>
          )}
        </div>

        {failed ? (
          <div style={{ padding: 18, display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div className="notice-bar error">
              <AlertCircle size={15} style={{ flexShrink: 0, marginTop: 1 }} />
              <div>
                {job.error?.code ? `[${job.error.code}] ` : ''}
                {job.error?.message || '处理中断，请查看本地后端日志。'}
              </div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
              <button type="button" className="btn btn-primary" onClick={onBack}>
                <RotateCcw size={13} />
                <span>返回重新导入</span>
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className="step-list">
              {STEPS.map((step, idx) => {
                const done = idx < activeIdx;
                const active = idx === activeIdx;
                return (
                  <div
                    key={step.key}
                    className={`step-item ${done ? 'done' : ''} ${active ? 'active' : ''}`}
                  >
                    <span className="step-num mono">
                      {done ? (
                        <Check size={13} strokeWidth={2.4} />
                      ) : active ? (
                        <Loader2 size={13} className="spin" />
                      ) : (
                        `0${idx + 1}`
                      )}
                    </span>
                    <div>
                      <div style={{ fontWeight: 500, fontSize: 12.5 }}>{step.title}</div>
                      <div style={{ fontSize: 11.5, color: 'var(--ink-secondary)' }}>
                        {step.desc}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="processing-foot">
              <span>
                完整推理可能需要几分钟，完成后自动进入剪辑台
              </span>
              <span className="mono">POLLING 1s</span>
            </div>
          </>
        )}
      </div>
    </main>
  );
}
