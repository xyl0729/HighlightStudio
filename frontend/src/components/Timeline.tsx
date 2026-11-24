import { useMemo, useRef, useState } from 'react';
import {
  ArrowLeftToLine,
  ArrowRightToLine,
  Plus,
  Scissors,
  Trash2,
} from 'lucide-react';
import type { AspectRatio, CropBox, MediaInfo, Segment } from '../types';
import {
  clamp,
  computeExportCropRect,
  formatClock,
  formatTimecode,
  frameToSeconds,
  MAX_SEGMENTS,
} from '../utils/time';

interface TimelineProps {
  media: MediaInfo;
  aspectRatio: AspectRatio;
  cropBoxes: CropBox[];
  segments: Segment[];
  selectedIndex: number;
  currentFrame: number;
  disabled: boolean;
  onSelectSegment: (index: number, seekToStart?: boolean) => void;
  onSeekFrame: (frame: number) => void;
  onChangeSegments: (next: Segment[], nextSelectedIndex?: number) => void;
  onAddSegment: () => void;
  onSplitAtPlayhead: () => void;
  onMarkIn: () => void;
  onMarkOut: () => void;
  onDeleteSelected: () => void;
}

type DragState =
  | { type: 'scrub' }
  | {
      type: 'left' | 'right' | 'move';
      index: number;
      originFrame: number;
      initialStart: number;
      initialEnd: number;
      minStart: number;
      maxEnd: number;
    };

export function Timeline({
  media,
  aspectRatio,
  cropBoxes,
  segments,
  selectedIndex,
  currentFrame,
  disabled,
  onSelectSegment,
  onSeekFrame,
  onChangeSegments,
  onAddSegment,
  onSplitAtPlayhead,
  onMarkIn,
  onMarkOut,
  onDeleteSelected,
}: TimelineProps) {
  const trackAreaRef = useRef<HTMLDivElement | null>(null);
  const [dragState, setDragState] = useState<DragState | null>(null);

  const totalFrames = Math.max(1, media.frame_count);

  const clientXToFrame = (clientX: number, allowEnd = false): number => {
    const el = trackAreaRef.current;
    if (!el) return 0;
    const rect = el.getBoundingClientRect();
    const ratio = clamp((clientX - rect.left) / Math.max(1, rect.width), 0, 1);
    const maxFrame = allowEnd ? totalFrames : Math.max(0, totalFrames - 1);
    return clamp(Math.round(ratio * totalFrames), 0, maxFrame);
  };

  const { majorTicks, minorTicks } = useMemo(() => {
    const dur = Math.max(0.1, media.duration);
    const candidates = [0.5, 1, 2, 5, 10, 15, 30, 60];
    const step = candidates.find((s) => dur / s <= 12) ?? 60;
    const major: Array<{ sec: number; pct: number }> = [];
    const minor: number[] = [];

    for (let t = 0; t <= dur + 0.0001; t += step) {
      major.push({ sec: t, pct: Math.min(100, (t / dur) * 100) });
      const mid = t + step / 2;
      if (mid < dur - 0.001) {
        minor.push((mid / dur) * 100);
      }
    }
    return { majorTicks: major, minorTicks: minor };
  }, [media.duration]);

  const motionPath = useMemo(() => {
    if (!cropBoxes.length) return { d: '', axisLabel: '轨迹' };
    const sampleCount = Math.min(cropBoxes.length, 180);
    const step = Math.max(1, Math.floor(cropBoxes.length / sampleCount));
    const centers: Array<{ nx: number; ny: number }> = [];

    let minX = 1;
    let maxX = 0;
    let minY = 1;
    let maxY = 0;

    for (let i = 0; i < cropBoxes.length; i += step) {
      const r = computeExportCropRect(cropBoxes[i], media, aspectRatio);
      const availX = Math.max(1, media.width - r.w);
      const availY = Math.max(1, media.height - r.h);
      const nx = availX > 1 ? r.x / availX : 0.5;
      const ny = availY > 1 ? r.y / availY : 0.5;
      minX = Math.min(minX, nx);
      maxX = Math.max(maxX, nx);
      minY = Math.min(minY, ny);
      maxY = Math.max(maxY, ny);
      centers.push({ nx, ny });
    }

    const useHorizontal = maxX - minX >= maxY - minY;
    const pts = centers.map((pt, idx) => {
      const x = (idx / Math.max(1, centers.length - 1)) * 1000;
      const val = useHorizontal ? pt.nx : pt.ny;
      const y = 35 - val * 26;
      return `${idx === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
    });

    return {
      d: pts.join(' '),
      axisLabel: useHorizontal ? 'X 轴平移' : 'Y 轴平移',
    };
  }, [cropBoxes, media, aspectRatio]);

  function startPointerTracking(initial: DragState, e: React.PointerEvent) {
    if (disabled && initial.type !== 'scrub') return;
    e.stopPropagation();
    setDragState(initial);

    const onMove = (ev: PointerEvent) => {
      if (initial.type === 'scrub') {
        onSeekFrame(clientXToFrame(ev.clientX, false));
        return;
      }

      const pointerFrame = clientXToFrame(ev.clientX, true);
      const next = segments.map((s) => ({ ...s }));
      const target = next[initial.index];
      if (!target) return;

      if (initial.type === 'left') {
        const clampedStart = clamp(pointerFrame, initial.minStart, initial.initialEnd - 1);
        target.start_frame = clampedStart;
        onChangeSegments(next, initial.index);
        onSeekFrame(clampedStart);
      } else if (initial.type === 'right') {
        const clampedEnd = clamp(pointerFrame, initial.initialStart + 1, initial.maxEnd);
        target.end_frame = clampedEnd;
        onChangeSegments(next, initial.index);
        onSeekFrame(Math.max(0, clampedEnd - 1));
      } else if (initial.type === 'move') {
        const delta = pointerFrame - initial.originFrame;
        const len = initial.initialEnd - initial.initialStart;
        const clampedStart = clamp(
          initial.initialStart + delta,
          initial.minStart,
          initial.maxEnd - len,
        );
        target.start_frame = clampedStart;
        target.end_frame = clampedStart + len;
        onChangeSegments(next, initial.index);
      }
    };

    const onUp = () => {
      setDragState(null);
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };

    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  }

  const playheadPct = clamp((currentFrame / totalFrames) * 100, 0, 100);
  const canSplit =
    !disabled &&
    segments.length < MAX_SEGMENTS &&
    segments.some((s) => currentFrame > s.start_frame && currentFrame < s.end_frame);

  return (
    <section className="timeline-dock" aria-label="高光时间轴">
      <div className="timeline-toolbar">
        <div className="timeline-toolbar-left">
          <button
            type="button"
            className="btn btn-sm btn-ghost"
            disabled={disabled || segments.length >= MAX_SEGMENTS}
            onClick={onAddSegment}
            title="在播放头或空白处添加高光片段"
          >
            <Plus size={12} />
            <span>添加片段</span>
          </button>

          <button
            type="button"
            className="btn btn-sm btn-ghost"
            disabled={!canSplit}
            onClick={onSplitAtPlayhead}
            title="在播放头位置切分片段 (S)"
          >
            <Scissors size={12} />
            <span>切分</span>
          </button>

          <span className="v-sep" style={{ margin: '0 4px' }} />

          <button
            type="button"
            className="btn btn-sm btn-ghost"
            disabled={disabled || !segments[selectedIndex]}
            onClick={onMarkIn}
            title="将选中片段起点设为当前帧 (I)"
          >
            <ArrowLeftToLine size={12} />
            <span>入点</span>
          </button>

          <button
            type="button"
            className="btn btn-sm btn-ghost"
            disabled={disabled || !segments[selectedIndex]}
            onClick={onMarkOut}
            title="将选中片段终点设为当前帧 (O)"
          >
            <ArrowRightToLine size={12} />
            <span>出点</span>
          </button>

          <button
            type="button"
            className="btn btn-sm btn-danger-ghost"
            disabled={disabled || segments.length <= 1 || !segments[selectedIndex]}
            onClick={onDeleteSelected}
            title="删除选中片段"
          >
            <Trash2 size={12} />
            <span>删除</span>
          </button>
        </div>

        <div className="timeline-toolbar-right">
          <div className="shortcut-hints">
            <span><span className="kbd">Space</span> 播放</span>
            <span><span className="kbd">←/→</span> 逐帧</span>
            <span><span className="kbd">I/O</span> 起止</span>
            <span><span className="kbd">S</span> 切分</span>
          </div>
        </div>
      </div>

      {/* Authentic NLE Track Layout: Fixed Header Column + Scrubbable Canvas */}
      <div className="timeline-grid">
        <div className="timeline-headers">
          <div className="track-hdr-ruler mono">TC</div>
          <div className="track-hdr-clips">
            <span className="track-code mono">V1</span>
            <span className="track-name">高光片段</span>
          </div>
          <div className="track-hdr-motion">
            <span className="track-code mono">TR</span>
            <span className="track-name">裁剪轨迹</span>
          </div>
        </div>

        <div ref={trackAreaRef} className="timeline-canvas">
          {/* Ruler */}
          <div
            className="timeline-ruler"
            onPointerDown={(e) => {
              onSeekFrame(clientXToFrame(e.clientX, false));
              startPointerTracking({ type: 'scrub' }, e);
            }}
          >
            {minorTicks.map((pct) => (
              <span key={pct} className="ruler-subtick" style={{ left: `${pct}%` }} />
            ))}

            {majorTicks.map((tick) => {
              const alignStyle =
                tick.pct <= 0.1
                  ? { left: '4px', transform: 'translateX(0)', alignItems: 'flex-start' as const }
                  : tick.pct >= 99.9
                    ? {
                        left: 'calc(100% - 4px)',
                        transform: 'translateX(-100%)',
                        alignItems: 'flex-end' as const,
                      }
                    : { left: `${tick.pct}%` };
              return (
                <div key={tick.sec} className="ruler-tick" style={alignStyle}>
                  <span className="ruler-tick-label mono">
                    {media.duration < 10 ? `${tick.sec.toFixed(1)}s` : formatClock(tick.sec)}
                  </span>
                  <span className="ruler-tick-mark" />
                </div>
              );
            })}
          </div>

          {/* Highlight Clip Lane */}
          <div
            className="clip-lane"
            onPointerDown={(e) => {
              onSeekFrame(clientXToFrame(e.clientX, false));
              startPointerTracking({ type: 'scrub' }, e);
            }}
          >
            {segments.map((seg, idx) => {
              const leftPct = (seg.start_frame / totalFrames) * 100;
              const widthPct = ((seg.end_frame - seg.start_frame) / totalFrames) * 100;
              const isSelected = idx === selectedIndex;
              const isDraggingThis =
                dragState && dragState.type !== 'scrub' && dragState.index === idx;

              const minStart = idx > 0 ? segments[idx - 1].end_frame : 0;
              const maxEnd =
                idx < segments.length - 1 ? segments[idx + 1].start_frame : totalFrames;
              const durSec = frameToSeconds(seg.end_frame - seg.start_frame, media.fps);

              return (
                <div
                  key={`${idx}-${seg.start_frame}-${seg.end_frame}`}
                  className={`clip-block ${isSelected ? 'selected' : ''} ${
                    isDraggingThis ? 'dragging' : ''
                  }`}
                  style={{
                    left: `${leftPct}%`,
                    width: `${Math.max(widthPct, 0.8)}%`,
                  }}
                  title={`#${idx + 1} ${formatTimecode(
                    frameToSeconds(seg.start_frame, media.fps),
                  )} – ${formatTimecode(frameToSeconds(seg.end_frame, media.fps))} (${durSec.toFixed(
                    2,
                  )}s)`}
                  onPointerDown={(e) => {
                    e.stopPropagation();
                    onSelectSegment(idx, true);
                    if (disabled) return;
                    startPointerTracking(
                      {
                        type: 'move',
                        index: idx,
                        originFrame: clientXToFrame(e.clientX, true),
                        initialStart: seg.start_frame,
                        initialEnd: seg.end_frame,
                        minStart,
                        maxEnd,
                      },
                      e,
                    );
                  }}
                >
                  <div
                    className="clip-handle left"
                    title="拖动调整起点"
                    onPointerDown={(e) => {
                      e.stopPropagation();
                      onSelectSegment(idx, false);
                      startPointerTracking(
                        {
                          type: 'left',
                          index: idx,
                          originFrame: seg.start_frame,
                          initialStart: seg.start_frame,
                          initialEnd: seg.end_frame,
                          minStart,
                          maxEnd,
                        },
                        e,
                      );
                    }}
                  />

                  <div className="clip-label mono">
                    #{String(idx + 1).padStart(2, '0')} · {durSec.toFixed(2)}s
                  </div>

                  <div
                    className="clip-handle right"
                    title="拖动调整终点"
                    onPointerDown={(e) => {
                      e.stopPropagation();
                      onSelectSegment(idx, false);
                      startPointerTracking(
                        {
                          type: 'right',
                          index: idx,
                          originFrame: seg.end_frame,
                          initialStart: seg.start_frame,
                          initialEnd: seg.end_frame,
                          minStart,
                          maxEnd,
                        },
                        e,
                      );
                    }}
                  />
                </div>
              );
            })}
          </div>

          {/* Real Crop Motion Track */}
          <div
            className="motion-lane"
            onPointerDown={(e) => {
              onSeekFrame(clientXToFrame(e.clientX, false));
              startPointerTracking({ type: 'scrub' }, e);
            }}
          >
            <span className="motion-axis-tag mono">{motionPath.axisLabel}</span>
            <svg
              viewBox="0 0 1000 44"
              preserveAspectRatio="none"
              style={{ width: '100%', height: '100%', display: 'block' }}
            >
              <line
                x1="0"
                y1="22"
                x2="1000"
                y2="22"
                stroke="rgba(255,255,255,0.06)"
                strokeDasharray="2 4"
              />
              {motionPath.d && (
                <path
                  d={motionPath.d}
                  fill="none"
                  stroke="#7e7c76"
                  strokeWidth="1.4"
                  vectorEffect="non-scaling-stroke"
                />
              )}
            </svg>
          </div>

          {/* Global Playhead */}
          <div className="playhead-line" style={{ left: `${playheadPct}%` }}>
            <div className="playhead-cap" />
          </div>
        </div>
      </div>
    </section>
  );
}
