import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ChevronLeft,
  ChevronRight,
  Columns2,
  Crop,
  Grid3x3,
  Pause,
  Play,
  Scan,
  Scissors,
  SkipBack,
  SkipForward,
  Volume2,
  VolumeX,
} from 'lucide-react';
import type {
  AspectRatio,
  CropBox,
  MediaInfo,
  MonitorViewMode,
  Segment,
} from '../types';
import {
  clamp,
  computeContainedVideoRect,
  computeExportCropRect,
  findSegmentAtFrame,
  formatTimecode,
  frameToSeconds,
  getAspectDimensions,
  secondsToFrame,
} from '../utils/time';

interface VideoStageProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  sourceUrl: string;
  media: MediaInfo;
  aspectRatio: AspectRatio;
  cropBoxes: CropBox[];
  segments: Segment[];
  currentFrame: number;
  isPlaying: boolean;
  highlightOnly: boolean;
  onHighlightOnlyChange: (value: boolean) => void;
  onFrameChange: (frame: number) => void;
  onPlayingChange: (playing: boolean) => void;
  onSeekFrame: (frame: number) => void;
  onStepFrame: (delta: number) => void;
  onJumpSegment: (direction: -1 | 1) => void;
  onTogglePlay: () => void;
}

export function VideoStage({
  videoRef,
  sourceUrl,
  media,
  aspectRatio,
  cropBoxes,
  segments,
  currentFrame,
  isPlaying,
  highlightOnly,
  onHighlightOnlyChange,
  onFrameChange,
  onPlayingChange,
  onSeekFrame,
  onStepFrame,
  onJumpSegment,
  onTogglePlay,
}: VideoStageProps) {
  const sourceWrapRef = useRef<HTMLDivElement | null>(null);
  const cropWrapRef = useRef<HTMLDivElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const cropCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const [viewMode, setViewMode] = useState<MonitorViewMode>('split');
  const [showGrid, setShowGrid] = useState(true);
  const [muted, setMuted] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);

  const latestRef = useRef({
    media,
    aspectRatio,
    cropBoxes,
    segments,
    showGrid,
    viewMode,
    highlightOnly,
    currentFrame,
  });
  latestRef.current = {
    media,
    aspectRatio,
    cropBoxes,
    segments,
    showGrid,
    viewMode,
    highlightOnly,
    currentFrame,
  };

  const drawCanvases = useCallback((frameIdx: number) => {
    const state = latestRef.current;
    const video = videoRef.current;
    const safeFrame = clamp(frameIdx, 0, Math.max(0, state.media.frame_count - 1));
    const box = state.cropBoxes[safeFrame];
    const exportRect = computeExportCropRect(box, state.media, state.aspectRatio);

    // 1. Draw Source Overlay Reticle inside .monitor-canvas-wrap (accounting for 10px padding)
    const overlayCanvas = overlayCanvasRef.current;
    const sourceWrap = sourceWrapRef.current;
    if (overlayCanvas && sourceWrap && state.viewMode !== 'crop') {
      const dpr = window.devicePixelRatio || 1;
      const wrapW = sourceWrap.clientWidth;
      const wrapH = sourceWrap.clientHeight;
      if (wrapW > 0 && wrapH > 0) {
        const targetW = Math.round(wrapW * dpr);
        const targetH = Math.round(wrapH * dpr);
        if (overlayCanvas.width !== targetW || overlayCanvas.height !== targetH) {
          overlayCanvas.width = targetW;
          overlayCanvas.height = targetH;
        }
        const ctx = overlayCanvas.getContext('2d');
        if (ctx) {
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          ctx.clearRect(0, 0, wrapW, wrapH);

          // Video sits inside 10px padding of sourceWrap
          const pad = 10;
          const innerW = Math.max(0, wrapW - pad * 2);
          const innerH = Math.max(0, wrapH - pad * 2);
          const vRect = computeContainedVideoRect(
            innerW,
            innerH,
            state.media.width,
            state.media.height,
          );
          vRect.x += pad;
          vRect.y += pad;

          if (vRect.w > 0 && vRect.h > 0) {
            const scaleX = vRect.w / state.media.width;
            const scaleY = vRect.h / state.media.height;

            const rx = vRect.x + exportRect.x * scaleX;
            const ry = vRect.y + exportRect.y * scaleY;
            const rw = exportRect.w * scaleX;
            const rh = exportRect.h * scaleY;

            // Darken area outside crop box (strictly within actual video rect)
            ctx.save();
            ctx.beginPath();
            ctx.rect(vRect.x, vRect.y, vRect.w, vRect.h);
            ctx.rect(rx, ry, rw, rh);
            ctx.fillStyle = 'rgba(0, 0, 0, 0.58)';
            ctx.fill('evenodd');
            ctx.restore();

            // Subtle rule-of-thirds grid
            if (state.showGrid) {
              ctx.save();
              ctx.strokeStyle = 'rgba(255, 255, 255, 0.19)';
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.moveTo(rx + rw / 3, ry);
              ctx.lineTo(rx + rw / 3, ry + rh);
              ctx.moveTo(rx + (rw * 2) / 3, ry);
              ctx.lineTo(rx + (rw * 2) / 3, ry + rh);
              ctx.moveTo(rx, ry + rh / 3);
              ctx.lineTo(rx + rw, ry + rh / 3);
              ctx.moveTo(rx, ry + (rh * 2) / 3);
              ctx.lineTo(rx + rw, ry + (rh * 2) / 3);
              ctx.stroke();
              ctx.restore();
            }

            // Crisp viewfinder outline & brackets
            const inHighlight = findSegmentAtFrame(state.segments, safeFrame) !== -1;
            ctx.save();
            ctx.strokeStyle = inHighlight ? '#d69c36' : 'rgba(236, 235, 232, 0.55)';
            ctx.lineWidth = 1.5;
            ctx.strokeRect(rx + 0.5, ry + 0.5, rw - 1, rh - 1);

            const bLen = Math.min(12, rw * 0.18, rh * 0.18);
            ctx.strokeStyle = inHighlight ? '#f6d186' : '#ecebe8';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(rx, ry + bLen);
            ctx.lineTo(rx, ry);
            ctx.lineTo(rx + bLen, ry);
            ctx.moveTo(rx + rw - bLen, ry);
            ctx.lineTo(rx + rw, ry);
            ctx.lineTo(rx + rw, ry + bLen);
            ctx.moveTo(rx, ry + rh - bLen);
            ctx.lineTo(rx, ry + rh);
            ctx.lineTo(rx + bLen, ry + rh);
            ctx.moveTo(rx + rw - bLen, ry + rh);
            ctx.lineTo(rx + rw, ry + rh);
            ctx.lineTo(rx + rw, ry + rh - bLen);
            ctx.stroke();
            ctx.restore();
          }
        }
      }
    }

    // 2. Draw Live Cropped Target Monitor
    const cropCanvas = cropCanvasRef.current;
    const cropWrap = cropWrapRef.current;
    if (
      cropCanvas &&
      cropWrap &&
      video &&
      video.readyState >= 2 &&
      (state.viewMode === 'split' || state.viewMode === 'crop')
    ) {
      const { rw, rh } = getAspectDimensions(state.aspectRatio);
      const pad = 20;
      const availW = Math.max(40, cropWrap.clientWidth - pad);
      const availH = Math.max(40, cropWrap.clientHeight - pad);
      const fit = computeContainedVideoRect(availW, availH, rw, rh);
      const dpr = window.devicePixelRatio || 1;
      const cssW = Math.max(2, Math.round(fit.w));
      const cssH = Math.max(2, Math.round(fit.h));
      const pxW = Math.round(cssW * dpr);
      const pxH = Math.round(cssH * dpr);

      if (cropCanvas.width !== pxW || cropCanvas.height !== pxH) {
        cropCanvas.width = pxW;
        cropCanvas.height = pxH;
        cropCanvas.style.width = `${cssW}px`;
        cropCanvas.style.height = `${cssH}px`;
      }

      const ctx = cropCanvas.getContext('2d');
      if (ctx) {
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, pxW, pxH);
        const sxRatio = video.videoWidth > 0 ? video.videoWidth / state.media.width : 1;
        const syRatio = video.videoHeight > 0 ? video.videoHeight / state.media.height : 1;

        ctx.drawImage(
          video,
          exportRect.x * sxRatio,
          exportRect.y * syRatio,
          exportRect.w * sxRatio,
          exportRect.h * syRatio,
          0,
          0,
          pxW,
          pxH,
        );

        if (state.showGrid) {
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.16)';
          ctx.lineWidth = dpr;
          ctx.beginPath();
          ctx.moveTo(pxW / 3, 0);
          ctx.lineTo(pxW / 3, pxH);
          ctx.moveTo((pxW * 2) / 3, 0);
          ctx.lineTo((pxW * 2) / 3, pxH);
          ctx.moveTo(0, pxH / 3);
          ctx.lineTo(pxW, pxH / 3);
          ctx.moveTo(0, (pxH * 2) / 3);
          ctx.lineTo(pxW, (pxH * 2) / 3);
          ctx.stroke();
        }
      }
    }
  }, [videoRef]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    let rafId = 0;
    let rvfcId = 0;
    let active = true;

    const handleTick = (mediaTime: number) => {
      if (!active) return;
      const state = latestRef.current;
      const frame = secondsToFrame(mediaTime, state.media.fps, state.media.frame_count);

      if (state.highlightOnly && !video.paused && state.segments.length > 0) {
        const inIdx = findSegmentAtFrame(state.segments, frame);
        if (inIdx === -1) {
          const nextSeg = state.segments.find((s) => s.start_frame > frame);
          if (nextSeg) {
            video.currentTime = frameToSeconds(nextSeg.start_frame, state.media.fps);
            onFrameChange(nextSeg.start_frame);
            drawCanvases(nextSeg.start_frame);
            return;
          } else {
            const lastEnd = state.segments[state.segments.length - 1].end_frame;
            if (frame >= lastEnd) {
              video.pause();
              const stopFrame = Math.max(0, lastEnd - 1);
              video.currentTime = frameToSeconds(stopFrame, state.media.fps);
              onFrameChange(stopFrame);
              drawCanvases(stopFrame);
              return;
            }
          }
        }
      }

      onFrameChange(frame);
      drawCanvases(frame);
    };

    const videoWithRvfc = video as HTMLVideoElement & {
      requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime: number }) => void) => number;
      cancelVideoFrameCallback?: (id: number) => void;
    };

    if (typeof videoWithRvfc.requestVideoFrameCallback === 'function') {
      const onVideoFrame = (_now: number, meta: { mediaTime: number }) => {
        handleTick(meta.mediaTime);
        if (active && typeof videoWithRvfc.requestVideoFrameCallback === 'function') {
          rvfcId = videoWithRvfc.requestVideoFrameCallback(onVideoFrame);
        }
      };
      rvfcId = videoWithRvfc.requestVideoFrameCallback(onVideoFrame);
    } else {
      const loop = () => {
        if (!active) return;
        if (!video.paused) handleTick(video.currentTime);
        rafId = requestAnimationFrame(loop);
      };
      rafId = requestAnimationFrame(loop);
    }

    return () => {
      active = false;
      if (rafId) cancelAnimationFrame(rafId);
      if (rvfcId && typeof videoWithRvfc.cancelVideoFrameCallback === 'function') {
        videoWithRvfc.cancelVideoFrameCallback(rvfcId);
      }
    };
  }, [videoRef, drawCanvases, onFrameChange]);

  useEffect(() => {
    drawCanvases(currentFrame);
  }, [currentFrame, viewMode, showGrid, segments, cropBoxes, aspectRatio, drawCanvases]);

  useEffect(() => {
    const observer = new ResizeObserver(() => {
      drawCanvases(latestRef.current.currentFrame);
    });
    if (sourceWrapRef.current) observer.observe(sourceWrapRef.current);
    if (cropWrapRef.current) observer.observe(cropWrapRef.current);
    return () => observer.disconnect();
  }, [viewMode, drawCanvases]);

  useEffect(() => {
    const video = videoRef.current;
    if (video) video.muted = muted;
  }, [muted, videoRef]);

  useEffect(() => {
    const video = videoRef.current;
    if (video) video.playbackRate = playbackRate;
  }, [playbackRate, videoRef]);

  const activeSegIndex = findSegmentAtFrame(segments, currentFrame);
  const currentCrop = computeExportCropRect(cropBoxes[currentFrame], media, aspectRatio);
  const currentSec = frameToSeconds(currentFrame, media.fps);

  return (
    <section className="stage-column">
      {/* Stage Top Bar */}
      <div className="stage-toolbar">
        <div className="stage-toolbar-group">
          <div className="segmented" role="tablist" aria-label="监看模式">
            <button
              type="button"
              className={viewMode === 'overlay' ? 'active' : ''}
              onClick={() => setViewMode('overlay')}
            >
              <Scan size={12} />
              <span>原片叠加</span>
            </button>
            <button
              type="button"
              className={viewMode === 'split' ? 'active' : ''}
              onClick={() => setViewMode('split')}
            >
              <Columns2 size={12} />
              <span>双屏对比</span>
            </button>
            <button
              type="button"
              className={viewMode === 'crop' ? 'active' : ''}
              onClick={() => setViewMode('crop')}
            >
              <Crop size={12} />
              <span>成片构图</span>
            </button>
          </div>
        </div>

        <div className="stage-toolbar-group">
          <button
            type="button"
            className={`toggle-btn ${highlightOnly ? 'active' : ''}`}
            onClick={() => onHighlightOnlyChange(!highlightOnly)}
            title="播放时自动跳过未入选空白，直接试看拼接节奏"
          >
            <Scissors size={12} />
            <span>仅播高光</span>
          </button>

          <button
            type="button"
            className={`toggle-btn ${showGrid ? 'active' : ''}`}
            onClick={() => setShowGrid((v) => !v)}
            title="在裁剪框内显示三分线"
          >
            <Grid3x3 size={12} />
            <span>三分线</span>
          </button>
        </div>
      </div>

      {/* Monitor Viewport */}
      <div className="monitor-viewport">
        {/* Source Pane (kept in compositor even in crop-only mode) */}
        <div
          className="monitor-pane"
          style={
            viewMode === 'crop'
              ? {
                  position: 'absolute',
                  width: 2,
                  height: 2,
                  opacity: 0.001,
                  pointerEvents: 'none',
                  overflow: 'hidden',
                }
              : undefined
          }
        >
          <div className="monitor-strip">
            <div className="monitor-strip-left">
              <span>原视频 · 推荐裁剪框</span>
              <span className="mono" style={{ color: 'var(--ink-muted)' }}>
                ({currentCrop.x}, {currentCrop.y}) {currentCrop.w}×{currentCrop.h}
              </span>
            </div>
            <div className="monitor-strip-right">
              <span className={`clip-indicator mono ${activeSegIndex !== -1 ? 'in-clip' : ''}`}>
                {activeSegIndex !== -1
                  ? `保留 · #${String(activeSegIndex + 1).padStart(2, '0')}`
                  : '未入选'}
              </span>
            </div>
          </div>

          <div ref={sourceWrapRef} className="monitor-canvas-wrap">
            <video
              ref={videoRef}
              className="monitor-video"
              src={sourceUrl}
              playsInline
              preload="auto"
              onPlay={() => onPlayingChange(true)}
              onPause={() => onPlayingChange(false)}
              onEnded={() => onPlayingChange(false)}
              onLoadedData={() => drawCanvases(latestRef.current.currentFrame)}
              onSeeked={() => {
                const v = videoRef.current;
                if (!v) return;
                const f = secondsToFrame(v.currentTime, media.fps, media.frame_count);
                onFrameChange(f);
                drawCanvases(f);
              }}
              onClick={onTogglePlay}
            />
            <canvas ref={overlayCanvasRef} className="overlay-canvas" />
          </div>
        </div>

        {/* Live Cropped Target Monitor */}
        {(viewMode === 'split' || viewMode === 'crop') && (
          <div className={`monitor-pane crop-pane ${viewMode === 'crop' ? 'solo' : ''}`}>
            <div className="monitor-strip">
              <div className="monitor-strip-left">
                <span>成片构图</span>
                <span className="mono" style={{ color: 'var(--clip-gold)' }}>
                  {aspectRatio}
                </span>
              </div>
              <div className="monitor-strip-right">
                <span className="mono" style={{ color: 'var(--ink-muted)' }}>
                  {currentCrop.w}×{currentCrop.h}
                </span>
                {viewMode === 'crop' && (
                  <span className={`clip-indicator mono ${activeSegIndex !== -1 ? 'in-clip' : ''}`}>
                    {activeSegIndex !== -1
                      ? `保留 · #${String(activeSegIndex + 1).padStart(2, '0')}`
                      : '未入选'}
                  </span>
                )}
              </div>
            </div>

            <div ref={cropWrapRef} className="monitor-canvas-wrap">
              <canvas
                ref={cropCanvasRef}
                className="crop-canvas"
                onClick={onTogglePlay}
                style={{ cursor: 'pointer' }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Transport Controls */}
      <div className="transport-bar">
        <div className="transport-group">
          <div className="timecode-readout">
            <span className="timecode-main mono">{formatTimecode(currentSec)}</span>
            <span className="timecode-sub mono">/ {formatTimecode(media.duration)}</span>
          </div>
          <span className="v-sep" style={{ margin: '0 4px' }} />
          <span className="mono" style={{ fontSize: 11.5, color: 'var(--ink-muted)' }}>
            F {String(currentFrame).padStart(4, '0')} / {media.frame_count}
          </span>
        </div>

        <div className="transport-group">
          <button
            type="button"
            className="btn btn-icon btn-ghost"
            onClick={() => onJumpSegment(-1)}
            title="上一段 ([)"
          >
            <SkipBack size={13} />
          </button>
          <button
            type="button"
            className="btn btn-icon btn-ghost"
            onClick={() => onStepFrame(-1)}
            title="上一帧 (← / ,)"
          >
            <ChevronLeft size={14} />
          </button>
          <button
            type="button"
            className="play-btn"
            onClick={onTogglePlay}
            title="播放 / 暂停 (Space)"
          >
            {isPlaying ? <Pause size={13} /> : <Play size={13} style={{ marginLeft: 1 }} />}
          </button>
          <button
            type="button"
            className="btn btn-icon btn-ghost"
            onClick={() => onStepFrame(1)}
            title="下一帧 (→ / .)"
          >
            <ChevronRight size={14} />
          </button>
          <button
            type="button"
            className="btn btn-icon btn-ghost"
            onClick={() => onJumpSegment(1)}
            title="下一段 (])"
          >
            <SkipForward size={13} />
          </button>
        </div>

        <div className="transport-group">
          <button
            type="button"
            className="btn btn-sm btn-ghost mono"
            onClick={() => onSeekFrame(0)}
            title="回到起点"
          >
            00:00
          </button>

          <select
            className="speed-select mono"
            aria-label="播放倍速"
            value={playbackRate}
            onChange={(e) => setPlaybackRate(Number(e.target.value))}
          >
            <option value={0.5}>0.5×</option>
            <option value={1}>1.0×</option>
            <option value={1.5}>1.5×</option>
            <option value={2}>2.0×</option>
          </select>

          <button
            type="button"
            className="btn btn-icon btn-ghost"
            disabled={!media.has_audio}
            onClick={() => setMuted((m) => !m)}
            title={!media.has_audio ? '源视频无音轨' : muted ? '取消静音' : '静音'}
          >
            {!media.has_audio || muted ? <VolumeX size={13} /> : <Volume2 size={13} />}
          </button>
        </div>
      </div>
    </section>
  );
}
