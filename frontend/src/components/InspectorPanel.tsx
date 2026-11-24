import {
  AlertTriangle,
  Check,
  Download,
  Loader2,
  Play,
  Plus,
  RotateCcw,
  Save,
  Trash2,
} from 'lucide-react';
import type { JobData, MediaInfo, Segment } from '../types';
import {
  clamp,
  formatTimecode,
  frameToSeconds,
  MAX_SEGMENTS,
  totalSelectedFrames,
} from '../utils/time';

interface InspectorPanelProps {
  job: JobData;
  media: MediaInfo;
  segments: Segment[];
  selectedIndex: number;
  currentFrame: number;
  isDirty: boolean;
  canResetRecommended: boolean;
  saving: boolean;
  actionError: string | null;
  downloadUrl: string | null;
  onSelectSegment: (index: number, seekToStart?: boolean) => void;
  onChangeSegments: (next: Segment[], nextSelectedIndex?: number) => void;
  onAddSegment: () => void;
  onResetRecommended: () => void;
  onSaveSegments: () => void;
  onExport: () => void;
  onOpenExportPreview: () => void;
}

export function InspectorPanel({
  job,
  media,
  segments,
  selectedIndex,
  currentFrame,
  isDirty,
  canResetRecommended,
  saving,
  actionError,
  downloadUrl,
  onSelectSegment,
  onChangeSegments,
  onAddSegment,
  onResetRecommended,
  onSaveSegments,
  onExport,
  onOpenExportPreview,
}: InspectorPanelProps) {
  const isExporting = job.status === 'exporting';
  const isExportedAndClean = job.status === 'exported' && !isDirty && Boolean(downloadUrl);
  const selectedFrames = totalSelectedFrames(segments);
  const selectedSec = frameToSeconds(selectedFrames, media.fps);

  function updateBoundary(index: number, field: 'start_frame' | 'end_frame', rawValue: number) {
    if (!Number.isFinite(rawValue)) return;
    const minStart = index > 0 ? segments[index - 1].end_frame : 0;
    const maxEnd = index < segments.length - 1 ? segments[index + 1].start_frame : media.frame_count;
    const next = segments.map((s) => ({ ...s }));
    const target = next[index];
    if (!target) return;

    if (field === 'start_frame') {
      target.start_frame = clamp(Math.round(rawValue), minStart, target.end_frame - 1);
    } else {
      target.end_frame = clamp(Math.round(rawValue), target.start_frame + 1, maxEnd);
    }
    onChangeSegments(next, index);
  }

  function removeSegment(index: number) {
    if (segments.length <= 1) return;
    const next = segments.filter((_, i) => i !== index);
    const nextSel = clamp(selectedIndex >= index ? selectedIndex - 1 : selectedIndex, 0, next.length - 1);
    onChangeSegments(next, nextSel);
  }

  return (
    <aside className="inspector-column" aria-label="片段清单与导出">
      {/* Header */}
      <div className="inspector-header">
        <span className="inspector-title">
          片段清单 <span className="mono" style={{ color: 'var(--ink-muted)', fontWeight: 400 }}>{segments.length}/{MAX_SEGMENTS}</span>
        </span>
        <div style={{ display: 'flex', gap: 2 }}>
          {canResetRecommended && (
            <button
              type="button"
              className="btn btn-sm btn-ghost"
              disabled={isExporting || saving}
              onClick={onResetRecommended}
              title="恢复初始推荐"
            >
              <RotateCcw size={11} />
              <span>恢复</span>
            </button>
          )}
          <button
            type="button"
            className="btn btn-icon btn-ghost"
            disabled={isExporting || saving || segments.length >= MAX_SEGMENTS}
            onClick={onAddSegment}
            title="新增片段"
          >
            <Plus size={13} />
          </button>
        </div>
      </div>

      {/* Summary sub-bar */}
      <div className="inspector-subbar">
        <span>成片总长</span>
        <span className="mono" style={{ color: 'var(--ink-primary)' }}>
          {formatTimecode(selectedSec)} ({selectedFrames} 帧)
        </span>
      </div>

      {/* Flat Hairline Segment List */}
      <div className="inspector-body">
        {(job.warnings.length > 0 || actionError || job.error) && (
          <div className="inspector-alerts">
            {job.warnings.length > 0 && (
              <div className="notice-bar warn">
                <AlertTriangle size={13} style={{ flexShrink: 0, marginTop: 1 }} />
                <div>
                  {job.warnings.map((w, i) => (
                    <div key={i}>{w}</div>
                  ))}
                </div>
              </div>
            )}
            {(actionError || job.error) && (
              <div className="notice-bar error">
                <AlertTriangle size={13} style={{ flexShrink: 0, marginTop: 1 }} />
                <span>{actionError || job.error?.message}</span>
              </div>
            )}
          </div>
        )}

        {segments.map((seg, idx) => {
          const isSelected = idx === selectedIndex;
          const segFrames = seg.end_frame - seg.start_frame;
          const startSec = frameToSeconds(seg.start_frame, media.fps);
          const endSec = frameToSeconds(seg.end_frame, media.fps);
          const durSec = frameToSeconds(segFrames, media.fps);

          return (
            <div
              key={idx}
              className={`segment-row ${isSelected ? 'selected' : ''}`}
              onClick={() => onSelectSegment(idx, false)}
            >
              <div className="segment-row-head">
                <button
                  type="button"
                  className="segment-jump-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelectSegment(idx, true);
                  }}
                  title="跳至片段起点"
                >
                  <span className="segment-num mono">#{String(idx + 1).padStart(2, '0')}</span>
                  <span className="mono" style={{ fontSize: 11.5, color: 'var(--ink-secondary)' }}>
                    {formatTimecode(startSec)} – {formatTimecode(endSec)}
                  </span>
                </button>

                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span className="mono" style={{ fontSize: 11, color: 'var(--ink-muted)' }}>
                    {durSec.toFixed(2)}s
                  </span>
                  <button
                    type="button"
                    className="btn btn-icon btn-danger-ghost"
                    style={{ width: 22, height: 22 }}
                    disabled={isExporting || saving || segments.length <= 1}
                    onClick={(e) => {
                      e.stopPropagation();
                      removeSegment(idx);
                    }}
                    title={segments.length <= 1 ? '至少保留 1 段' : '删除此片段'}
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              </div>

              <div className="segment-inputs">
                <div className="frame-field">
                  <div className="frame-field-label">
                    <span>起点</span>
                    <button
                      type="button"
                      className="snap-link"
                      disabled={isExporting || saving}
                      onClick={(e) => {
                        e.stopPropagation();
                        updateBoundary(idx, 'start_frame', currentFrame);
                      }}
                    >
                      当前帧
                    </button>
                  </div>
                  <div className="frame-input-row">
                    <input
                      type="number"
                      className="mono"
                      disabled={isExporting || saving}
                      value={seg.start_frame}
                      min={idx > 0 ? segments[idx - 1].end_frame : 0}
                      max={seg.end_frame - 1}
                      onChange={(e) => updateBoundary(idx, 'start_frame', Number(e.target.value))}
                    />
                    <span className="frame-input-unit mono">f</span>
                  </div>
                </div>

                <div className="frame-field">
                  <div className="frame-field-label">
                    <span>终点 (不含)</span>
                    <button
                      type="button"
                      className="snap-link"
                      disabled={isExporting || saving}
                      onClick={(e) => {
                        e.stopPropagation();
                        updateBoundary(idx, 'end_frame', currentFrame + 1);
                      }}
                    >
                      当前帧
                    </button>
                  </div>
                  <div className="frame-input-row">
                    <input
                      type="number"
                      className="mono"
                      disabled={isExporting || saving}
                      value={seg.end_frame}
                      min={seg.start_frame + 1}
                      max={idx < segments.length - 1 ? segments[idx + 1].start_frame : media.frame_count}
                      onChange={(e) => updateBoundary(idx, 'end_frame', Number(e.target.value))}
                    />
                    <span className="frame-input-unit mono">f</span>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Pinned Save & Export Footer */}
      <div className="inspector-footer">
        {isDirty && (
          <div className="notice-bar warn">
            <span>
              {job.status === 'exported'
                ? '片段已变更，原导出已失效，需重新导出'
                : '存在未保存的片段修改'}
            </span>
          </div>
        )}

        <div className="export-actions">
          <button
            type="button"
            className="btn"
            disabled={!isDirty || saving || isExporting}
            onClick={onSaveSegments}
          >
            {saving ? <Loader2 size={12} className="spin" /> : <Save size={12} />}
            <span>{saving ? '保存中' : '保存片段'}</span>
          </button>

          <button
            type="button"
            className="btn btn-primary"
            disabled={saving || isExporting}
            onClick={onExport}
          >
            {isExporting ? (
              <>
                <Loader2 size={13} className="spin" />
                <span>正在导出 MP4...</span>
              </>
            ) : (
              <span>{isDirty ? '保存并导出 MP4' : '导出成片 MP4'}</span>
            )}
          </button>
        </div>

        {isExportedAndClean && downloadUrl && (
          <div className="exported-strip">
            <div className="exported-strip-head">
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                <Check size={13} />
                成片已就绪
              </span>
              <span className="mono" style={{ color: 'var(--ink-secondary)', fontSize: 11 }}>
                {job.aspect_ratio} · {media.has_audio ? '含原音轨' : '无音轨'}
              </span>
            </div>
            <div className="exported-strip-actions">
              <button type="button" className="btn btn-sm" onClick={onOpenExportPreview}>
                <Play size={11} />
                <span>预览成片</span>
              </button>
              <a
                className="btn btn-sm btn-accent"
                href={downloadUrl}
                download="highlights.mp4"
                style={{ textDecoration: 'none' }}
              >
                <Download size={11} />
                <span>下载 MP4</span>
              </a>
            </div>
          </div>
        )}

        <div className="footer-specs mono">
          <span>30 FPS · 逐帧构图</span>
          <span>{media.has_audio ? '保留音轨' : '无音轨'}</span>
        </div>
      </div>
    </aside>
  );
}
