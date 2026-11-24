import { useEffect, useRef, useState } from 'react';
import { AlertCircle, ArrowRight, FolderOpen, X } from 'lucide-react';
import type { AspectRatio, HealthResponse, VideoUpload } from '../types';
import { formatBytes, formatClock } from '../utils/time';

interface ImportViewProps {
  health: HealthResponse | null;
  healthError: string | null;
  aspectRatio: AspectRatio;
  onChangeAspectRatio: (ratio: AspectRatio) => void;
  submitting: boolean;
  uploadProgress: number;
  submitError: string | null;
  onSubmit: (uploads: VideoUpload[]) => Promise<VideoUpload[]>;
}

interface LocalVideoMeta {
  width: number;
  height: number;
  duration: number;
}

export function ImportView({
  health,
  healthError,
  aspectRatio,
  onChangeAspectRatio,
  submitting,
  uploadProgress,
  submitError,
  onSubmit,
}: ImportViewProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragging, setDragging] = useState(false);
  const [uploads, setUploads] = useState<VideoUpload[]>([]);
  const file = uploads.length === 1 ? uploads[0].file : null;
  const firstRatio = uploads[0]?.aspectRatio ?? aspectRatio;
  const selectedRatio = uploads.every((item) => item.aspectRatio === firstRatio) ? firstRatio : null;
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [localMeta, setLocalMeta] = useState<LocalVideoMeta | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      setLocalMeta(null);
      return;
    }
    setLocalMeta(null);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => {
      URL.revokeObjectURL(url);
    };
  }, [file]);

  const maxBytes = health?.max_upload_bytes ?? 512 * 1024 * 1024;
  const modelReady = health?.final_engine_configured ?? false;

  function pickFiles(candidates: FileList | null) {
    if (!candidates?.length || submitting) return;
    const additions = [...uploads];
    const errors: string[] = [];
    for (const candidate of Array.from(candidates)) {
      if (!candidate.size) {
        errors.push(candidate.name + ' · 文件为空');
      } else if (candidate.size > maxBytes) {
        errors.push(candidate.name + ' · 超过 ' + formatBytes(maxBytes));
      } else if (!additions.some(({ file }) =>
        file.name === candidate.name && file.size === candidate.size &&
        file.lastModified === candidate.lastModified
      )) {
        const ratio = /^输出\s*16\s*[:：]\s*9(?=\D|$)/.test(candidate.name)
          ? '16:9'
          : /^输出\s*9\s*[:：]\s*16(?=\D|$)/.test(candidate.name)
            ? '9:16'
            : aspectRatio;
        additions.push({ file: candidate, aspectRatio: ratio });
      }
    }
    setUploads(additions);
    setLocalError(errors.join('\n') || null);
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);
    pickFiles(e.dataTransfer.files);
  }

  function changeRatio(ratio: AspectRatio) {
    onChangeAspectRatio(ratio);
    setUploads((current) => current.map((item) => ({ ...item, aspectRatio: ratio })));
  }

  function removeFile(file: File) {
    setUploads((current) => current.filter((item) => item.file !== file));
    setLocalError(null);
  }

  async function submit() {
    setLocalError(null);
    const failed = await onSubmit(uploads);
    setUploads(failed);
  }

  const canStart = uploads.length > 0 && !submitting && modelReady;

  return (
    <main className="import-workbench">
      {/* Left Full-Bleed Media Stage */}
      <section className="import-stage">
        <div className="import-stage-bar">
          <span>素材导入</span>
          <span className="mono" style={{ color: 'var(--ink-muted)' }}>
            MP4 / MOV / WebM / MKV / AVI · ≤ {formatBytes(maxBytes)} · ≤ 10 min
          </span>
        </div>

        <div
          className={`import-canvas ${dragging ? 'dragging' : ''}`}
          onDragOver={(e) => {
            e.preventDefault();
            if (!submitting) setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
        >
          <input
            ref={inputRef}
            type="file"
            multiple
            accept="video/*,.mp4,.mov,.mkv,.webm,.avi"
            style={{ display: 'none' }}
            onChange={(e) => {
              pickFiles(e.target.files);
              e.target.value = '';
            }}
          />

          {uploads.length === 0 ? (
            <div
              className="viewfinder-drop"
              role="button"
              tabIndex={0}
              onClick={() => !submitting && inputRef.current?.click()}
              onKeyDown={(e) => {
                if ((e.key === 'Enter' || e.key === ' ') && !submitting) {
                  e.preventDefault();
                  inputRef.current?.click();
                }
              }}
            >
              <span className="viewfinder-kicker mono">SOURCE MEDIA</span>
              <h1 className="viewfinder-title">拖入或选取一个或多个视频</h1>
              <p className="viewfinder-sub">
                本地后端分析原视频，生成高光片段与逐帧构图
              </p>
              <button
                type="button"
                className="btn"
                disabled={submitting}
                onClick={(e) => {
                  e.stopPropagation();
                  inputRef.current?.click();
                }}
              >
                <FolderOpen size={13} />
                <span>打开本地文件</span>
              </button>
            </div>
          ) : file ? (
            <div className="import-preview-stage">
              {previewUrl && (
                <video
                  src={previewUrl}
                  controls
                  muted
                  preload="metadata"
                  onLoadedMetadata={(e) => {
                    const v = e.currentTarget;
                    setLocalMeta({
                      width: v.videoWidth,
                      height: v.videoHeight,
                      duration: v.duration,
                    });
                  }}
                />
              )}
            </div>
          ) : (
            <ul className="import-batch-list" aria-label="待上传视频">
              {uploads.map((item) => (
                <li key={item.file.name + ':' + item.file.size + ':' + item.file.lastModified}>
                  <span className="queue-file-name" title={item.file.name}>{item.file.name}</span>
                  <span className="mono import-batch-size">{formatBytes(item.file.size)}</span>
                  <select
                    aria-label={item.file.name + ' 的输出画幅'}
                    value={item.aspectRatio}
                    disabled={submitting}
                    onChange={(e) => {
                      const ratio = e.target.value as AspectRatio;
                      setUploads((current) => current.map((other) =>
                        other.file === item.file ? { ...other, aspectRatio: ratio } : other,
                      ));
                    }}
                  >
                    <option value="9:16">9:16</option>
                    <option value="16:9">16:9</option>
                  </select>
                  <button
                    type="button"
                    className="btn btn-icon btn-ghost"
                    disabled={submitting}
                    onClick={() => removeFile(item.file)}
                    aria-label={'移除 ' + item.file.name}
                  >
                    <X size={13} />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="import-file-strip">
          {file ? (
            <>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  minWidth: 0,
                  fontSize: 12,
                }}
              >
                <span
                  style={{
                    color: 'var(--ink-primary)',
                    fontWeight: 500,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                  title={file.name}
                >
                  {file.name}
                </span>
                <span className="v-sep" />
                <span className="mono" style={{ color: 'var(--ink-secondary)', fontSize: 11.5 }}>
                  {formatBytes(file.size)}
                  {localMeta
                    ? ` · ${localMeta.width}×${localMeta.height} · ${formatClock(localMeta.duration)}`
                    : ''}
                </span>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 4, flexShrink: 0 }}>
                <button
                  type="button"
                  className="btn btn-sm btn-ghost"
                  disabled={submitting}
                  onClick={() => inputRef.current?.click()}
                >
                  添加视频
                </button>
                <button
                  type="button"
                  className="btn btn-icon btn-ghost"
                  disabled={submitting}
                  onClick={() => removeFile(file)}
                  title="移除文件"
                >
                  <X size={13} />
                </button>
              </div>
            </>
          ) : uploads.length ? (
            <>
              <span>{uploads.length} 条待上传</span>
              <button type="button" className="btn btn-sm btn-ghost" disabled={submitting} onClick={() => inputRef.current?.click()}>
                添加视频
              </button>
            </>
          ) : (
            <span style={{ fontSize: 11.5, color: 'var(--ink-muted)' }}>
              尚未选择视频 · 本地自托管运行，仅处理自有或授权素材
            </span>
          )}
        </div>
      </section>

      {/* Right Parameter Rail (flush 340px inspector) */}
      <aside className="import-rail" aria-label="分析参数设置">
        {healthError && (
          <div className="rail-section">
            <div className="notice-bar error">
              <AlertCircle size={14} style={{ flexShrink: 0, marginTop: 1 }} />
              <div>
                未检测到后端服务（127.0.0.1:8000）。请先运行 <code className="mono">.\start-local.ps1</code>。
              </div>
            </div>
          </div>
        )}

        <div className="rail-section">
          <div className="rail-heading">{uploads.length > 1 ? "统一设置输出画幅" : "输出画幅"}</div>
          <div className="ratio-switch" role="radiogroup" aria-label="输出画幅">
            <button
              type="button"
              role="radio"
              aria-checked={selectedRatio === '9:16'}
              className={`ratio-btn ${selectedRatio === '9:16' ? 'selected' : ''}`}
              disabled={submitting}
              onClick={() => changeRatio('9:16')}
            >
              <span className="ratio-wire portrait" />
              <span className="mono" style={{ fontWeight: 500 }}>9:16</span>
              <span style={{ fontSize: 11.5, color: 'var(--ink-secondary)' }}>竖屏</span>
            </button>

            <button
              type="button"
              role="radio"
              aria-checked={selectedRatio === '16:9'}
              className={`ratio-btn ${selectedRatio === '16:9' ? 'selected' : ''}`}
              disabled={submitting}
              onClick={() => changeRatio('16:9')}
            >
              <span className="ratio-wire landscape" />
              <span className="mono" style={{ fontWeight: 500 }}>16:9</span>
              <span style={{ fontSize: 11.5, color: 'var(--ink-secondary)' }}>横屏</span>
            </button>
          </div>
        </div>

        <div className="rail-section">
          <div className="rail-heading">高光识别引擎</div>
          <div className="mode-rows">
            <div className="mode-row selected">
              <div className="mode-row-body">
                <div className="mode-row-top">
                  <span className="mode-row-name">多模态高光引擎</span>
                  <span className={`mode-row-state ${modelReady ? 'ok' : 'warn'}`}>
                    {modelReady ? '已就绪' : '未配置'}
                  </span>
                </div>
                <div className="mode-row-spec">原视频多模态高光定位、主体追踪构图与平滑后处理</div>
              </div>
            </div>
          </div>

          {health && !modelReady && (
            <div className="notice-bar warn">
              <span>
                请配置本地模型目录（HIGHLIGHT_FINAL_MODELS）后重启本地服务。
              </span>
            </div>
          )}
        </div>

        <div className="rail-section">
          <div className="rail-heading">管线规格</div>
          <div className="spec-table">
            <div className="spec-row">
              <span className="spec-key">预览基准</span>
              <span className="spec-val mono">30 FPS · ≤1280px</span>
            </div>
            <div className="spec-row">
              <span className="spec-key">构图重构</span>
              <span className="spec-val">逐帧推荐构图</span>
            </div>
            <div className="spec-row">
              <span className="spec-key">音频导出</span>
              <span className="spec-val">按选中区间裁剪原音轨</span>
            </div>
            <div className="spec-row">
              <span className="spec-key">切换画幅</span>
              <span className="spec-val">需新建分析任务</span>
            </div>
          </div>
        </div>

        <div className="rail-footer">
          {(localError || submitError) && (
            <div className="notice-bar error">
              <AlertCircle size={14} style={{ flexShrink: 0, marginTop: 1 }} />
              <span className="upload-errors">{[localError, submitError].filter(Boolean).join('\n')}</span>
            </div>
          )}

          {submitting && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  fontSize: 11.5,
                  color: 'var(--ink-secondary)',
                }}
              >
                <span>正在上传并排队</span>
                <span className="mono">{Math.round(uploadProgress * 100)}%</span>
              </div>
              <div className="progress-track">
                <div
                  className="progress-fill"
                  style={{ width: `${Math.round(uploadProgress * 100)}%` }}
                />
              </div>
            </div>
          )}

          <button
            type="button"
            className="btn btn-primary btn-lg"
            disabled={!canStart}
            onClick={submit}
          >
            <span>{submitting ? '正在提交...' : '加入队列'}</span>
            <ArrowRight size={14} />
          </button>
        </div>
      </aside>
    </main>
  );
}
