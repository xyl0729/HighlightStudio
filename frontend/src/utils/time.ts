import type { AspectRatio, CropBox, MediaInfo, Segment } from '../types';

export const MAX_SEGMENTS = 50;

export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function frameToSeconds(frame: number, fps: number): number {
  if (fps <= 0) return 0;
  return frame / fps;
}

export function secondsToFrame(seconds: number, fps: number, frameCount: number): number {
  const raw = Math.round(seconds * fps);
  return clamp(raw, 0, Math.max(0, frameCount - 1));
}

export function formatTimecode(seconds: number): string {
  const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const mins = Math.floor(safe / 60);
  const secs = Math.floor(safe % 60);
  const centis = Math.floor((safe - Math.floor(safe)) * 100);
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(centis).padStart(2, '0')}`;
}

export function formatClock(seconds: number): string {
  const safe = Math.max(0, Math.floor(seconds));
  const mins = Math.floor(safe / 60);
  const secs = safe % 60;
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export function totalSelectedFrames(segments: Segment[]): number {
  return segments.reduce((sum, seg) => sum + Math.max(0, seg.end_frame - seg.start_frame), 0);
}

export function segmentsEqual(a: Segment[], b: Segment[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i].start_frame !== b[i].start_frame || a[i].end_frame !== b[i].end_frame) {
      return false;
    }
  }
  return true;
}

export function findSegmentAtFrame(segments: Segment[], frame: number): number {
  for (let i = 0; i < segments.length; i++) {
    if (frame >= segments[i].start_frame && frame < segments[i].end_frame) {
      return i;
    }
  }
  return -1;
}

/**
 * Insert a new non-overlapping segment near `anchorFrame` (or in the largest available gap).
 * Returns `{ segments, insertedIndex }` or `null` if no free space or at max segment limit.
 */
export function insertSegmentAtFrame(
  segments: Segment[],
  anchorFrame: number,
  frameCount: number,
  fps: number,
): { segments: Segment[]; insertedIndex: number } | null {
  if (segments.length >= MAX_SEGMENTS || frameCount <= 0) return null;

  const sorted = [...segments].sort((a, b) => a.start_frame - b.start_frame);
  const defaultLen = Math.max(1, Math.min(Math.round(fps * 1.5), Math.max(1, Math.floor(frameCount / 4))));

  // Collect all free gaps [gapStart, gapEnd)
  const gaps: Array<{ start: number; end: number }> = [];
  let cursor = 0;
  for (const seg of sorted) {
    if (seg.start_frame > cursor) {
      gaps.push({ start: cursor, end: seg.start_frame });
    }
    cursor = Math.max(cursor, seg.end_frame);
  }
  if (cursor < frameCount) {
    gaps.push({ start: cursor, end: frameCount });
  }
  if (gaps.length === 0) return null;

  // Prefer the gap containing anchorFrame, else the closest gap to anchorFrame, else largest gap
  let targetGap = gaps.find((g) => anchorFrame >= g.start && anchorFrame < g.end);
  if (!targetGap) {
    // Pick largest gap if anchor is inside an existing segment, preferring one immediately after anchor
    const afterAnchor = gaps.find((g) => g.start >= anchorFrame && g.end - g.start >= 2);
    targetGap =
      afterAnchor ??
      gaps.reduce((best, g) => (g.end - g.start > best.end - best.start ? g : best), gaps[0]);
  }

  const gapLen = targetGap.end - targetGap.start;
  const segLen = Math.min(defaultLen, gapLen);
  let start = targetGap.start;
  if (anchorFrame >= targetGap.start && anchorFrame < targetGap.end) {
    start = clamp(anchorFrame, targetGap.start, targetGap.end - segLen);
  }
  const newSeg: Segment = {
    start_frame: start,
    end_frame: start + segLen,
  };

  const next = [...sorted, newSeg].sort((a, b) => a.start_frame - b.start_frame);
  const insertedIndex = next.findIndex(
    (s) => s.start_frame === newSeg.start_frame && s.end_frame === newSeg.end_frame,
  );
  return { segments: next, insertedIndex };
}

/**
 * Split the segment containing `splitFrame` into two adjacent segments [start, split) and [split, end).
 */
export function splitSegmentAtFrame(
  segments: Segment[],
  splitFrame: number,
): { segments: Segment[]; rightIndex: number } | null {
  if (segments.length >= MAX_SEGMENTS) return null;
  const idx = segments.findIndex(
    (seg) => splitFrame > seg.start_frame && splitFrame < seg.end_frame,
  );
  if (idx === -1) return null;

  const target = segments[idx];
  const left: Segment = { start_frame: target.start_frame, end_frame: splitFrame };
  const right: Segment = { start_frame: splitFrame, end_frame: target.end_frame };
  const next = [...segments.slice(0, idx), left, right, ...segments.slice(idx + 1)];
  return { segments: next, rightIndex: idx + 1 };
}

export function getAspectDimensions(ratio: AspectRatio): { rw: number; rh: number } {
  const [rw, rh] = ratio.split(':').map(Number);
  return { rw, rh };
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/**
 * Compute the rendered video content box inside an `object-fit: contain` container.
 */
export function computeContainedVideoRect(
  containerW: number,
  containerH: number,
  videoW: number,
  videoH: number,
): Rect {
  if (containerW <= 0 || containerH <= 0 || videoW <= 0 || videoH <= 0) {
    return { x: 0, y: 0, w: 0, h: 0 };
  }
  const containerRatio = containerW / containerH;
  const videoRatio = videoW / videoH;
  let w = containerW;
  let h = containerH;
  let x = 0;
  let y = 0;

  if (videoRatio > containerRatio) {
    h = containerW / videoRatio;
    y = (containerH - h) / 2;
  } else {
    w = containerH * videoRatio;
    x = (containerW - w) / 2;
  }
  return { x, y, w, h };
}

/**
 * Exact export crop rectangle matching `studio/media.py` lines 63-65 & 99-103.
 */
export function computeExportCropRect(
  box: CropBox | undefined,
  media: MediaInfo,
  ratio: AspectRatio,
): Rect {
  const { rw, rh } = getAspectDimensions(ratio);
  const unit = Math.floor(Math.min(Math.floor(media.width / rw), Math.floor(media.height / rh)) / 2) * 2;
  const outW = Math.max(2, unit * rw);
  const outH = Math.max(2, unit * rh);

  if (!box) {
    return {
      x: Math.round((media.width - outW) / 2),
      y: Math.round((media.height - outH) / 2),
      w: outW,
      h: outH,
    };
  }

  const [bx, by, bw] = box;
  const bh = (bw * rh) / rw;
  const left = clamp(Math.round(bx + (bw - outW) / 2), 0, Math.max(0, media.width - outW));
  const top = clamp(Math.round(by + (bh - outH) / 2), 0, Math.max(0, media.height - outH));
  return { x: left, y: top, w: outW, h: outH };
}

/**
 * Raw recommended crop box in source pixel space: [x, y, w] -> {x, y, w, h}.
 */
export function computeRawCropRect(
  box: CropBox | undefined,
  media: MediaInfo,
  ratio: AspectRatio,
): Rect {
  if (!box) return computeExportCropRect(undefined, media, ratio);
  const { rw, rh } = getAspectDimensions(ratio);
  const [x, y, w] = box;
  const h = (w * rh) / rw;
  return { x, y, w, h };
}
