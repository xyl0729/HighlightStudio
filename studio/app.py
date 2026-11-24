"""Single-user FastAPI service for a self-hosted highlight editor."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field, replace
import json
import logging
import os
from pathlib import Path
import shutil
from threading import RLock
from time import time_ns
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from .analysis import Analyzer
from .final_engine import FinalEngine
from .media import Media, MediaError, normalize, render

logger = logging.getLogger(__name__)
ACTIVE = {"queued", "preparing", "analyzing", "exporting"}


class Segment(BaseModel):
    start_frame: Annotated[int, Field(strict=True, ge=0)]
    end_frame: Annotated[int, Field(strict=True, gt=0)]


class Edit(BaseModel):
    segments: Annotated[list[Segment], Field(min_length=1, max_length=50)]


@dataclass
class Job:
    id: str
    folder: Path
    ratio: str
    mode: str
    name: str
    submitted_at: int
    status: str = "queued"
    media: Media | None = None
    segments: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    boxes: list = field(default_factory=list)
    engine: str | None = None
    crop_tracking: str | None = None
    warnings: list = field(default_factory=list)
    error: dict | None = None
    export: Path | None = None

    @property
    def source(self) -> Path:
        return self.folder / "source.mp4"

    def save(self, **changes) -> None:
        updated = replace(self, **changes)
        data = vars(updated).copy()
        del data["folder"]
        data["media"] = asdict(updated.media) if updated.media else None
        data["export"] = updated.export.name if updated.export else None
        temporary = self.folder / "job.json.tmp"
        temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.folder / "job.json")
        vars(self).update(changes)

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        if str(UUID(path.parent.name)) != data["id"]:
            raise ValueError("Job ID does not match its folder.")
        export = data.pop("export")
        if export and (Path(export).name != export or not export.startswith("export-")
                       or not export.endswith(".mp4")):
            raise ValueError("Invalid export filename.")
        media = data.pop("media")
        return cls(folder=path.parent, media=Media(**media) if media else None,
                   export=path.parent / export if export else None, **data)

    def public(self, *, detail: bool = True) -> dict:
        base = f"/api/jobs/{self.id}"
        return {
            "id": self.id, "name": self.name, "submitted_at": self.submitted_at,
            "status": self.status, "aspect_ratio": self.ratio, "mode": self.mode,
            "media": asdict(self.media) if detail and self.media else None,
            "segments": [{"start_frame": a, "end_frame": b} for a, b in self.segments] if detail else [],
            "recommended_segments": [
                {"start_frame": a, "end_frame": b} for a, b in self.recommendations
            ] if detail else [],
            "crop_boxes": self.boxes if detail else [],
            "analysis_engine": self.engine, "crop_tracking": self.crop_tracking,
            "warnings": self.warnings,
            "error": self.error, "source_url": f"{base}/source" if detail and self.media else None,
            "download_url": f"{base}/download" if detail and self.status == "exported" else None,
        }


def create_app(work: Path | None = None, max_upload_bytes: int = 512 * 1024 ** 2,
               dev_preview: bool = False) -> FastAPI:
    work = (work or Path(os.getenv("HIGHLIGHT_WORK_DIR", ".work"))).resolve()
    model = os.getenv("HIGHLIGHT_MODEL_DIR")
    weights = os.getenv("HIGHLIGHT_YOLO_WEIGHTS")
    analyzer = Analyzer(Path(model) if model else None, Path(weights) if weights else None)
    final_engine = FinalEngine(os.getenv("HIGHLIGHT_FINAL_CODE"), os.getenv("HIGHLIGHT_FINAL_MODELS"))
    origins = [origin.strip() for origin in os.getenv(
        "HIGHLIGHT_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",") if origin.strip()]
    lock = RLock()
    jobs: dict[str, Job] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        work.mkdir(parents=True, exist_ok=True)
        jobs.clear()
        for path in work.glob("*/job.json"):
            try:
                job = Job.load(path)
                if job.status in {"queued", "preparing", "analyzing"}:
                    if (job.folder / "upload").is_file():
                        job.status = "queued"
                    else:
                        job.save(status="failed", error={
                            "code": "analysis_interrupted", "message": "上次分析已中断，请重新导入原视频。",
                        })
                elif job.status == "exporting" or (job.status == "exported" and
                                                   (job.export is None or not job.export.is_file())):
                    job.save(status="ready", error={
                        "code": "export_failed", "message": "上次导出未保留，请重新导出。",
                    })
                jobs[job.id] = job
            except (OSError, ValueError, TypeError, KeyError):
                logger.exception("Could not restore job metadata %s", path)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-worker") as worker:
            app.state.worker = worker
            for job in sorted(jobs.values(), key=lambda item: item.submitted_at):
                if job.status == "queued":
                    worker.submit(prepare, job)
            yield

    app = FastAPI(title="Highlight Studio API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=origins,
                       allow_methods=["GET", "POST", "PUT", "DELETE"], allow_headers=["Content-Type", "Range"])

    @app.middleware("http")
    async def browser_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        same_origin = f"{request.url.scheme}://{request.url.netloc}"
        if request.method in {"POST", "PUT", "DELETE"} and origin and origin not in [same_origin, *origins]:
            return JSONResponse({"detail": "Origin is not permitted."}, status_code=403)
        return await call_next(request)

    def get_job(job_id: UUID) -> Job:
        job = jobs.get(str(job_id))
        if job is None:
            raise HTTPException(404, "Job not found.")
        return job

    def prepare(job: Job):
        upload = job.folder / "upload"
        try:
            with lock:
                job.status = "preparing"
            media = normalize(upload, job.source)
            with lock:
                job.media, job.status = media, "analyzing"
            if job.mode == "ai" and final_engine.configured:
                segments, boxes = final_engine.analyze(upload, media, job.ratio)
                engine, tracking, warnings = "multimodal-v1", "multimodal-v1", []
            elif dev_preview:
                segments, boxes, engine, warnings, tracking = analyzer.analyze(
                    job.source, media, job.ratio, job.mode)
            else:
                raise ValueError("多模态推理引擎未配置，请检查模型目录设置。")
            result = dict(segments=segments, recommendations=segments, boxes=boxes,
                          engine=engine, warnings=warnings, crop_tracking=tracking, status="ready")
        except MediaError as error:
            result = dict(error={"code": "invalid_video", "message": str(error)}, status="failed")
        except Exception as error:
            logger.exception("Analysis failed for %s", job.id)
            message = str(error) if type(error) is ValueError else "Analysis failed. Check the server log."
            result = dict(error={"code": "analysis_failed", "message": message}, status="failed")
        with lock:
            try:
                job.save(**result)
            except OSError:
                logger.exception("Could not save analysis for %s", job.id)
                job.status = "failed"
                job.error = {"code": "storage_failed", "message": "无法保存分析结果，请检查本机存储。"}
                return
        upload.unlink(missing_ok=True)

    def export(job: Job, output: Path):
        try:
            render(job.source, output, job.media, job.ratio, job.segments, job.boxes)
            with lock:
                job.save(export=output, status="exported")
        except Exception:
            logger.exception("Export failed for %s", job.id)
            output.unlink(missing_ok=True)
            error = {"code": "export_failed", "message": "MP4 export failed. Check the server log."}
            with lock:
                try:
                    job.save(error=error, status="ready")
                except OSError:
                    logger.exception("Could not save export status for %s", job.id)
                    job.error, job.status = error, "ready"

    @app.get("/api/health")
    def health():
        return {"status": "ok", "model_configured": final_engine.configured,
                "final_engine_configured": final_engine.configured,
                "subject_detector_configured": bool(weights and Path(weights).is_file()),
                "max_upload_bytes": max_upload_bytes}

    @app.post("/api/jobs", status_code=202)
    def create_job(
        file: Annotated[UploadFile, File()],
        aspect_ratio: Annotated[Literal["9:16", "16:9"], Form()] = "9:16",
        mode: Annotated[Literal["ai", "quick"], Form()] = "ai",
    ):
        if mode == "quick" and not dev_preview:
            raise HTTPException(422, "快速模式已从正式界面移除。")
        if mode == "ai" and not (final_engine.configured or dev_preview):
            raise HTTPException(503, "多模态推理引擎尚未配置，请配置本机模型目录后重启服务。")
        job_id = str(uuid4())
        folder = work / job_id
        folder.mkdir()
        try:
            with (folder / "upload").open("xb") as target:
                size = 0
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_upload_bytes:
                        raise HTTPException(413, "Video exceeds the upload limit.")
                    target.write(chunk)
            if not size:
                raise HTTPException(422, "Video file is empty.")
            name = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1] or "视频"
            job = Job(job_id, folder, aspect_ratio, mode, name, time_ns() // 1_000_000)
            job.save()
        except Exception:
            shutil.rmtree(folder)
            raise
        with lock:
            jobs[job_id] = job
            response = job.public()
        app.state.worker.submit(prepare, job)
        return response

    @app.get("/api/jobs")
    def list_jobs():
        with lock:
            return [job.public(detail=False) for job in
                    sorted(jobs.values(), key=lambda item: item.submitted_at)]

    @app.get("/api/jobs/{job_id}")
    def read_job(job_id: UUID):
        with lock:
            return get_job(job_id).public()

    @app.get("/api/jobs/{job_id}/source", response_class=FileResponse)
    def source(job_id: UUID):
        with lock:
            job = get_job(job_id)
            if job.media is None:
                raise HTTPException(409, "Video preparation has not completed.")
            return FileResponse(job.source, media_type="video/mp4", headers={"Cache-Control": "no-store"})

    @app.put("/api/jobs/{job_id}/segments")
    def edit(job_id: UUID, data: Edit):
        with lock:
            job = get_job(job_id)
            if job.status not in {"ready", "exported"}:
                raise HTTPException(409, "Job is not ready for editing.")
            segments = [(part.start_frame, part.end_frame) for part in data.segments]
            previous_end = 0
            for start, end in segments:
                if not previous_end <= start < end <= job.media.frame_count:
                    raise HTTPException(422, "Segments must be sorted, non-overlapping and within the video.")
                previous_end = end
            job.save(segments=segments, status="ready", error=None)
            return job.public()

    @app.post("/api/jobs/{job_id}/export", status_code=202)
    def request_export(job_id: UUID):
        with lock:
            job = get_job(job_id)
            if job.status not in {"ready", "exported"}:
                raise HTTPException(409, "Job is not ready for export.")
            job.save(status="exporting", error=None)
            response = job.public()
        app.state.worker.submit(export, job, job.folder / f"export-{uuid4().hex}.mp4")
        return response

    @app.get("/api/jobs/{job_id}/download", response_class=FileResponse)
    def download(job_id: UUID):
        with lock:
            job = get_job(job_id)
            if job.status != "exported":
                raise HTTPException(409, "The current edit has no exported video.")
            return FileResponse(job.export, media_type="video/mp4", filename="highlights.mp4",
                                headers={"Cache-Control": "no-store"})

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete(job_id: UUID):
        with lock:
            job = get_job(job_id)
            if job.status in ACTIVE:
                raise HTTPException(409, "Wait until the job has finished.")
            try:
                shutil.rmtree(job.folder)
            except OSError as error:
                raise HTTPException(409, "Stop playing or downloading this video and try again.") from error
            jobs.pop(job.id)
        return Response(status_code=204)

    return app


app = create_app()
