import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import {
  createJob,
  deleteJob,
  getHealth,
  getJob,
  listJobs,
  requestExport,
  resolveApiUrl,
  updateSegments,
} from './api';
import { ExportPreviewModal } from './components/ExportPreviewModal';
import { ImportView } from './components/ImportView';
import { JobQueue } from './components/JobQueue';
import { InspectorPanel } from './components/InspectorPanel';
import { ProcessingView } from './components/ProcessingView';
import { Timeline } from './components/Timeline';
import { TopBar } from './components/TopBar';
import { VideoStage } from './components/VideoStage';
import type {
  AspectRatio,
  HealthResponse,
  JobData,
  QueuedVideo,
  Segment,
  VideoUpload,
} from './types';
import {
  clamp,
  findSegmentAtFrame,
  frameToSeconds,
  insertSegmentAtFrame,
  segmentsEqual,
  splitSegmentAtFrame,
  totalSelectedFrames,
} from './utils/time';

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  const [aspectRatio, setAspectRatio] = useState<AspectRatio>('9:16');

  const [submitting, setSubmitting] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const [queue, setQueue] = useState<QueuedVideo[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [openingJob, setOpeningJob] = useState(false);
  const selected = queue.find(({ job }) => job.id === selectedJobId);
  const job = selected?.job ?? null;
  const fileName = selected?.name ?? null;

  // Editor state
  const [initialSegments, setInitialSegments] = useState<Segment[]>([]);
  const [draftSegments, setDraftSegments] = useState<Segment[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [highlightOnly, setHighlightOnly] = useState(false);

  const [saving, setSaving] = useState(false);
  const [busyCleaning, setBusyCleaning] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [showExportModal, setShowExportModal] = useState(false);
  const [exportVersion, setExportVersion] = useState(0);

  const checkHealth = useCallback(async () => {
    try {
      const data = await getHealth();
      setHealth(data);
      setHealthError(null);
    } catch (err) {
      setHealth(null);
      setHealthError(err instanceof Error ? err.message : '后端连接失败');
    }
  }, []);

  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  useEffect(() => {
    const controller = new AbortController();
    listJobs(controller.signal).then((jobs) => {
      setQueue((current) => {
        const existing = new Map(current.map((entry) => [entry.job.id, entry]));
        return [
          ...jobs.map((job) => existing.get(job.id) ??
            { job, name: job.name, submittedAt: job.submitted_at }),
          ...current.filter((entry) => !jobs.some((job) => job.id === entry.job.id)),
        ];
      });
    }).catch((error) => {
      if (!controller.signal.aborted) {
        setSubmitError(error instanceof Error ? error.message : '无法读取任务队列。');
      }
    });
    return () => controller.abort();
  }, []);

  const pendingIds = queue
    .filter(({ job }) => ['queued', 'preparing', 'analyzing', 'exporting'].includes(job.status))
    .map(({ job }) => job.id)
    .join(',');

  useEffect(() => {
    if (!pendingIds) return;

    const ids = pendingIds.split(',');
    const controller = new AbortController();
    let timer: number;

    async function poll() {
      const results = await Promise.allSettled(ids.map((id) => getJob(id, controller.signal)));
      if (controller.signal.aborted) return;

      const upcoming = new Map(results.flatMap((result) =>
        result.status === 'fulfilled' ? [[result.value.id, result.value] as const] : [],
      ));
      setQueue((current) => current.map((entry) => {
        const latest = upcoming.get(entry.job.id);
        return latest ? {
          ...entry,
          job: latest,
          recommendations: latest.recommended_segments?.length
            ? latest.recommended_segments
            : entry.recommendations ?? (latest.status === 'ready' ? latest.segments : undefined),
        } : entry;
      }));
      if ([...upcoming.values()].some((latest) => latest.status === 'exported')) {
        setExportVersion((version) => version + 1);
      }
      timer = window.setTimeout(poll, 1000);
    }

    timer = window.setTimeout(poll, 1000);
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [pendingIds]);

  function setJob(updated: JobData) {
    setQueue((current) => current.map((entry) =>
      entry.job.id === updated.id ? { ...entry, job: updated } : entry,
    ));
  }

  // Seek <video> to frame
  const seekFrame = useCallback(
    (frame: number) => {
      if (!job?.media) return;
      const clamped = clamp(Math.round(frame), 0, Math.max(0, job.media.frame_count - 1));
      setCurrentFrame(clamped);
      const video = videoRef.current;
      if (video && video.readyState >= 1) {
        video.currentTime = frameToSeconds(clamped, job.media.fps);
      }
    },
    [job?.media],
  );

  // When entering editor for the first time, jump video to first recommended highlight once loaded
  useEffect(() => {
    if (!job?.media || job.status !== 'ready') return;
    const video = videoRef.current;
    if (!video) return;
    const firstStart = initialSegments[0]?.start_frame ?? 0;
    const onMeta = () => {
      video.currentTime = frameToSeconds(firstStart, job.media!.fps);
      setCurrentFrame(firstStart);
    };
    if (video.readyState >= 1) {
      return;
    }
    video.addEventListener('loadedmetadata', onMeta, { once: true });
    return () => video.removeEventListener('loadedmetadata', onMeta);
  }, [job?.id, job?.media, job?.status, initialSegments]);

  const togglePlay = useCallback(() => {
    const video = videoRef.current;
    if (!video || !job?.media) return;
    if (video.paused) {
      // If highlight-only mode is on and playhead is outside or at the very end, jump to first valid highlight
      if (highlightOnly && draftSegments.length > 0) {
        const inIdx = findSegmentAtFrame(draftSegments, currentFrame);
        if (inIdx === -1) {
          const nextSeg =
            draftSegments.find((s) => s.start_frame > currentFrame) ?? draftSegments[0];
          video.currentTime = frameToSeconds(nextSeg.start_frame, job.media.fps);
          setCurrentFrame(nextSeg.start_frame);
        }
      }
      video.play().catch(() => {});
    } else {
      video.pause();
    }
  }, [highlightOnly, draftSegments, currentFrame, job?.media]);

  const stepFrame = useCallback(
    (delta: number) => {
      const video = videoRef.current;
      if (video && !video.paused) video.pause();
      seekFrame(currentFrame + delta);
    },
    [currentFrame, seekFrame],
  );

  const jumpSegment = useCallback(
    (direction: -1 | 1) => {
      if (draftSegments.length === 0) return;
      if (direction === 1) {
        const nextIdx = draftSegments.findIndex((s) => s.start_frame > currentFrame + 1);
        const idx = nextIdx !== -1 ? nextIdx : 0;
        setSelectedIndex(idx);
        seekFrame(draftSegments[idx].start_frame);
      } else {
        let prevIdx = -1;
        for (let i = draftSegments.length - 1; i >= 0; i--) {
          if (draftSegments[i].start_frame < currentFrame - 1) {
            prevIdx = i;
            break;
          }
        }
        const idx = prevIdx !== -1 ? prevIdx : draftSegments.length - 1;
        setSelectedIndex(idx);
        seekFrame(draftSegments[idx].start_frame);
      }
    },
    [draftSegments, currentFrame, seekFrame],
  );

  const handleSelectSegment = useCallback(
    (index: number, seekToStart = false) => {
      setSelectedIndex(index);
      const seg = draftSegments[index];
      if (seekToStart && seg) {
        seekFrame(seg.start_frame);
      }
    },
    [draftSegments, seekFrame],
  );

  const handleChangeSegments = useCallback((next: Segment[], nextSel?: number) => {
    setDraftSegments(next);
    if (typeof nextSel === 'number') {
      setSelectedIndex(clamp(nextSel, 0, Math.max(0, next.length - 1)));
    }
    setActionError(null);
  }, []);

  const handleAddSegment = useCallback(() => {
    if (!job?.media) return;
    const res = insertSegmentAtFrame(
      draftSegments,
      currentFrame,
      job.media.frame_count,
      job.media.fps,
    );
    if (!res) {
      setActionError('没有可用的空白区间或已达到 50 段上限。');
      return;
    }
    setDraftSegments(res.segments);
    setSelectedIndex(res.insertedIndex);
    seekFrame(res.segments[res.insertedIndex].start_frame);
    setActionError(null);
  }, [job?.media, draftSegments, currentFrame, seekFrame]);

  const handleSplitAtPlayhead = useCallback(() => {
    const res = splitSegmentAtFrame(draftSegments, currentFrame);
    if (!res) return;
    setDraftSegments(res.segments);
    setSelectedIndex(res.rightIndex);
    setActionError(null);
  }, [draftSegments, currentFrame]);

  const handleMarkIn = useCallback(() => {
    const target = draftSegments[selectedIndex];
    if (!target) return;
    const minStart = selectedIndex > 0 ? draftSegments[selectedIndex - 1].end_frame : 0;
    if (currentFrame >= target.end_frame || currentFrame < minStart) {
      setActionError('当前播放头超出该片段允许的起点范围。');
      return;
    }
    const next = draftSegments.map((s, i) =>
      i === selectedIndex ? { ...s, start_frame: currentFrame } : s,
    );
    setDraftSegments(next);
    setActionError(null);
  }, [draftSegments, selectedIndex, currentFrame]);

  const handleMarkOut = useCallback(() => {
    if (!job?.media) return;
    const target = draftSegments[selectedIndex];
    if (!target) return;
    const maxEnd =
      selectedIndex < draftSegments.length - 1
        ? draftSegments[selectedIndex + 1].start_frame
        : job.media.frame_count;
    const nextEnd = currentFrame + 1;
    if (nextEnd <= target.start_frame || nextEnd > maxEnd) {
      setActionError('当前播放头超出该片段允许的终点范围。');
      return;
    }
    const next = draftSegments.map((s, i) =>
      i === selectedIndex ? { ...s, end_frame: nextEnd } : s,
    );
    setDraftSegments(next);
    setActionError(null);
  }, [draftSegments, selectedIndex, currentFrame, job?.media]);

  const handleDeleteSelected = useCallback(() => {
    if (draftSegments.length <= 1) return;
    const next = draftSegments.filter((_, i) => i !== selectedIndex);
    setDraftSegments(next);
    setSelectedIndex(clamp(selectedIndex - 1, 0, next.length - 1));
    setActionError(null);
  }, [draftSegments, selectedIndex]);

  // Keyboard shortcuts when inside editor
  const inEditor =
    job !== null &&
    job.media !== null &&
    (job.status === 'ready' || job.status === 'exporting' || job.status === 'exported');

  useEffect(() => {
    if (!inEditor || showExportModal) return;
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable)
      ) {
        return;
      }

      if (e.code === 'Space') {
        e.preventDefault();
        togglePlay();
      } else if (e.key === 'ArrowLeft' || e.key === ',') {
        e.preventDefault();
        stepFrame(-1);
      } else if (e.key === 'ArrowRight' || e.key === '.') {
        e.preventDefault();
        stepFrame(1);
      } else if (e.key === '[') {
        e.preventDefault();
        jumpSegment(-1);
      } else if (e.key === ']') {
        e.preventDefault();
        jumpSegment(1);
      } else if (e.key === 'i' || e.key === 'I') {
        e.preventDefault();
        handleMarkIn();
      } else if (e.key === 'o' || e.key === 'O') {
        e.preventDefault();
        handleMarkOut();
      } else if (e.key === 's' || e.key === 'S') {
        if (!e.ctrlKey && !e.metaKey) {
          e.preventDefault();
          handleSplitAtPlayhead();
        }
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [
    inEditor,
    showExportModal,
    togglePlay,
    stepFrame,
    jumpSegment,
    handleMarkIn,
    handleMarkOut,
    handleSplitAtPlayhead,
  ]);

  useLayoutEffect(() => {
    const segments = inEditor ? job?.segments ?? [] : [];
    setInitialSegments(selected?.recommendations ?? segments);
    setDraftSegments(segments);
    setSelectedIndex(0);
    setCurrentFrame(segments[0]?.start_frame ?? 0);
    setIsPlaying(false);
    setActionError(null);
    setShowExportModal(false);
  }, [job?.id, inEditor]);

  async function handleSubmitFiles(uploads: VideoUpload[]): Promise<VideoUpload[]> {
    setSubmitting(true);
    setUploadProgress(0);
    setSubmitError(null);
    const accepted: QueuedVideo[] = [];
    const failed: VideoUpload[] = [];
    const errors: string[] = [];

    try {
      for (const [index, upload] of uploads.entries()) {
        try {
          const created = await createJob(upload.file, upload.aspectRatio, 'ai', (progress) => {
            setUploadProgress((index + progress) / uploads.length);
          });
          accepted.push({ job: created, name: upload.file.name, submittedAt: Date.now() });
        } catch (error) {
          failed.push(upload);
          errors.push(upload.file.name + ' · ' +
            (error instanceof Error ? error.message : '提交任务失败'));
        }
        setUploadProgress((index + 1) / uploads.length);
      }
      setQueue((current) => [...current, ...accepted]);
      if (failed.length) {
        setSubmitError(errors.join('\n'));
      } else if (accepted.length) {
        setSelectedJobId(accepted[0].job.id);
      }
      return failed;
    } finally {
      setSubmitting(false);
    }
  }

  const isDirty = job !== null && !segmentsEqual(draftSegments, job.segments);
  const canResetRecommended =
    initialSegments.length > 0 && !segmentsEqual(draftSegments, initialSegments);

  async function handleSaveSegments(): Promise<JobData | null> {
    if (!job) return null;
    setSaving(true);
    setActionError(null);
    try {
      const updated = await updateSegments(job.id, draftSegments);
      setJob(updated);
      setDraftSegments(updated.segments);
      return updated;
    } catch (err) {
      setActionError(err instanceof Error ? err.message : '保存片段失败');
      return null;
    } finally {
      setSaving(false);
    }
  }

  async function handleExport() {
    if (!job) return;
    setActionError(null);
    if (isDirty) {
      const saved = await handleSaveSegments();
      if (!saved) return;
    }
    try {
      const updated = await requestExport(job.id);
      setJob(updated);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : '发起导出失败');
    }
  }

  // Safely unload <video> before calling DELETE so Windows releases file handles
  async function releaseAndDeleteJob() {
    if (!job) return;
    setBusyCleaning(true);
    setActionError(null);
    setShowExportModal(false);

    const video = videoRef.current;
    if (video) {
      video.pause();
      video.removeAttribute('src');
      video.load();
    }

    // Give browser & OS a moment to close the media stream before rmtree
    await new Promise((r) => window.setTimeout(r, 180));

    try {
      try {
        await deleteJob(job.id);
      } catch {
        // Retry once after a transient Windows file handle release.
        await new Promise((resolve) => window.setTimeout(resolve, 350));
        await deleteJob(job.id);
      }
      setQueue((current) => current.filter((entry) => entry.job.id !== job.id));
      setSelectedJobId(null);
      setSubmitError(null);
    } catch (error) {
      setActionError(error instanceof Error ? error.message : '清理任务失败，请稍后重试。');
    } finally {
      setBusyCleaning(false);
    }
  }

  async function openJob(id: string) {
    setOpeningJob(true);
    try {
      const job = await getJob(id);
      setQueue((current) => current.map((entry) => entry.job.id === id ? {
        ...entry,
        job,
        recommendations: job.recommended_segments?.length ? job.recommended_segments : entry.recommendations,
      } : entry));
      setSelectedJobId(id);
    } catch (error) {
      const message = error instanceof Error ? error.message : '无法打开任务。';
      if (selectedJobId) setActionError(message);
      else setSubmitError(message);
    } finally {
      setOpeningJob(false);
    }
  }

  function openImport() {
    setSubmitError(null);
    setSelectedJobId(null);
  }

  const sourceUrl = job?.source_url ? resolveApiUrl(job.source_url) : '';
  const downloadUrl =
    job?.download_url && job.status === 'exported'
      ? `${resolveApiUrl(job.download_url)}?v=${exportVersion}`
      : null;

  return (
    <div className="app-shell">
      <TopBar
        health={health}
        healthError={healthError}
        onRefreshHealth={checkHealth}
        job={job}
        fileName={fileName}
        busyCleaning={busyCleaning}
        onDeleteJob={releaseAndDeleteJob}
        queue={
          <JobQueue
            items={queue}
            selectedId={selectedJobId}
            busy={submitting || busyCleaning || saving || openingJob}
            hasUnsavedEdits={isDirty}
            onSelect={openJob}
            onAdd={openImport}
          />
        }
      />

      {!selected || !job ? (
        <ImportView
          health={health}
          healthError={healthError}
          aspectRatio={aspectRatio}
          onChangeAspectRatio={setAspectRatio}
          submitting={submitting}
          uploadProgress={uploadProgress}
          submitError={submitError}
          onSubmit={handleSubmitFiles}
        />
      ) : !inEditor ? (
        <ProcessingView
          job={job}
          fileName={fileName}
          startedAt={selected.submittedAt}
          onBack={releaseAndDeleteJob}
        />
      ) : (
        <div className="workspace">
          <div className="workspace-upper">
            <VideoStage
              videoRef={videoRef}
              sourceUrl={sourceUrl}
              media={job.media!}
              aspectRatio={job.aspect_ratio}
              cropBoxes={job.crop_boxes}
              segments={draftSegments}
              currentFrame={currentFrame}
              isPlaying={isPlaying}
              highlightOnly={highlightOnly}
              onHighlightOnlyChange={setHighlightOnly}
              onFrameChange={setCurrentFrame}
              onPlayingChange={setIsPlaying}
              onSeekFrame={seekFrame}
              onStepFrame={stepFrame}
              onJumpSegment={jumpSegment}
              onTogglePlay={togglePlay}
            />

            <InspectorPanel
              job={job}
              media={job.media!}
              segments={draftSegments}
              selectedIndex={selectedIndex}
              currentFrame={currentFrame}
              isDirty={isDirty}
              canResetRecommended={canResetRecommended}
              saving={saving}
              actionError={actionError ?? (job.error?.code === 'export_failed' ? job.error.message : null)}
              downloadUrl={downloadUrl}
              onSelectSegment={handleSelectSegment}
              onChangeSegments={handleChangeSegments}
              onAddSegment={handleAddSegment}
              onResetRecommended={() => {
                setDraftSegments(initialSegments);
                setSelectedIndex(0);
                setActionError(null);
              }}
              onSaveSegments={handleSaveSegments}
              onExport={handleExport}
              onOpenExportPreview={() => setShowExportModal(true)}
            />
          </div>

          <Timeline
            media={job.media!}
            aspectRatio={job.aspect_ratio}
            cropBoxes={job.crop_boxes}
            segments={draftSegments}
            selectedIndex={selectedIndex}
            currentFrame={currentFrame}
            disabled={job.status === 'exporting' || saving}
            onSelectSegment={handleSelectSegment}
            onSeekFrame={seekFrame}
            onChangeSegments={handleChangeSegments}
            onAddSegment={handleAddSegment}
            onSplitAtPlayhead={handleSplitAtPlayhead}
            onMarkIn={handleMarkIn}
            onMarkOut={handleMarkOut}
            onDeleteSelected={handleDeleteSelected}
          />
        </div>
      )}

      {showExportModal && downloadUrl && job?.media && (
        <ExportPreviewModal
          downloadUrl={downloadUrl}
          aspectRatio={job.aspect_ratio}
          segmentCount={job.segments.length}
          totalDurationSec={frameToSeconds(totalSelectedFrames(job.segments), job.media.fps)}
          hasAudio={job.media.has_audio}
          onClose={() => setShowExportModal(false)}
        />
      )}
    </div>
  );
}
