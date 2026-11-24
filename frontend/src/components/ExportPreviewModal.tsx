import { Download, X } from 'lucide-react';
import type { AspectRatio } from '../types';
import { formatTimecode } from '../utils/time';

interface ExportPreviewModalProps {
  downloadUrl: string;
  aspectRatio: AspectRatio;
  segmentCount: number;
  totalDurationSec: number;
  hasAudio: boolean;
  onClose: () => void;
}

export function ExportPreviewModal({
  downloadUrl,
  aspectRatio,
  segmentCount,
  totalDurationSec,
  hasAudio,
  onClose,
}: ExportPreviewModalProps) {
  return (
    <div
      className="modal-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label="成片预览"
      onClick={onClose}
    >
      <div className="modal-window" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontWeight: 600, fontSize: 12.5 }}>成片预览</span>
            <span className="v-sep" />
            <span className="mono" style={{ fontSize: 11.5, color: 'var(--clip-gold)' }}>
              {aspectRatio}
            </span>
            <span className="mono" style={{ fontSize: 11.5, color: 'var(--ink-secondary)' }}>
              {segmentCount} 段 · {formatTimecode(totalDurationSec)} ·{' '}
              {hasAudio ? '含原音轨' : '无音轨'}
            </span>
          </div>
          <button
            type="button"
            className="btn btn-icon btn-ghost"
            onClick={onClose}
            aria-label="关闭预览"
          >
            <X size={14} />
          </button>
        </div>

        <div className="modal-body">
          <video src={downloadUrl} controls autoPlay playsInline />
        </div>

        <div className="modal-footer">
          <span className="mono" style={{ fontSize: 11, color: 'var(--ink-muted)' }}>
            highlights.mp4 · 30 FPS · 偶数像素对齐
          </span>
          <div style={{ display: 'flex', gap: 8 }}>
            <button type="button" className="btn btn-sm" onClick={onClose}>
              返回编辑
            </button>
            <a
              className="btn btn-sm btn-primary"
              href={downloadUrl}
              download="highlights.mp4"
              style={{ textDecoration: 'none' }}
            >
              <Download size={12} />
              <span>下载 MP4</span>
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
