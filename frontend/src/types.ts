export type AspectRatio = '9:16' | '16:9';
export type AnalysisMode = 'ai' | 'quick';

export type JobStatus =
  | 'queued'
  | 'preparing'
  | 'analyzing'
  | 'ready'
  | 'exporting'
  | 'exported'
  | 'failed';

export interface HealthResponse {
  status: string;
  model_configured: boolean;
  final_engine_configured?: boolean;
  max_upload_bytes: number;
}

export interface MediaInfo {
  width: number;
  height: number;
  fps: number;
  frame_count: number;
  duration: number;
  has_audio: boolean;
}

export interface Segment {
  start_frame: number;
  end_frame: number;
}

export type CropBox = [number, number, number];

export interface JobError {
  code: string;
  message: string;
}

export interface JobData {
  id: string;
  name: string;
  submitted_at: number;
  status: JobStatus;
  aspect_ratio: AspectRatio;
  mode: AnalysisMode;
  media: MediaInfo | null;
  segments: Segment[];
  recommended_segments: Segment[];
  crop_boxes: CropBox[];
  analysis_engine: 'video-ora-4b' | 'motion' | string | null;
  warnings: string[];
  error: JobError | null;
  source_url: string | null;
  download_url: string | null;
}

export type MonitorViewMode = 'overlay' | 'split' | 'crop';

export interface VideoUpload {
  file: File;
  aspectRatio: AspectRatio;
}

export interface QueuedVideo {
  job: JobData;
  name: string;
  submittedAt: number;
  recommendations?: Segment[];
}
