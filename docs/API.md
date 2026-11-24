# Highlight Studio HTTP API 与前后端协议

后端默认监听 `http://127.0.0.1:8000`，启动后可在 `/docs` 查看 OpenAPI 交互式文档。前端开发服务器默认在 `http://127.0.0.1:5173` 通过 Vite 代理 `/api` 转发到后端；如需其他跨域来源，可通过环境变量 `HIGHLIGHT_ORIGINS` 配置（逗号分隔）。

## 核心接口

| 动作 | 方法与路径 | 说明 |
|---|---|---|
| 健康检查 | `GET /api/health` | 返回 `status`、`model_configured`、`final_engine_configured`、`max_upload_bytes` |
| 获取任务列表 | `GET /api/jobs` | 返回工作目录下已持久化的所有任务摘要（用于刷新/重启恢复队列） |
| 提交视频任务 | `POST /api/jobs` | `multipart/form-data`，字段：`file`（视频文件）、`aspect_ratio`（`9:16` \| `16:9`）、`mode`（`ai`）。返回 HTTP 202 与任务对象 |
| 查询任务详情 | `GET /api/jobs/{id}` | 建议约 1 秒轮询一次，在 `ready`、`exported`、`failed` 时停止 |
| 原视频预览流 | `GET /api/jobs/{id}/source` | 标准化后的 30 FPS MP4 流，支持 HTTP `Range` 请求，可直接填入 `<video src>` |
| 保存时间段 | `PUT /api/jobs/{id}/segments` | 提交编辑后的片段列表，返回更新后的 `ready` 任务（原导出作废） |
| 发起导出 | `POST /api/jobs/{id}/export` | 无请求体，返回 HTTP 202 与 `exporting` 状态，恢复轮询直至 `exported` |
| 下载成片 | `GET /api/jobs/{id}/download` | 下载带原音轨的拼接成片 `highlights.mp4` |
| 删除任务缓存 | `DELETE /api/jobs/{id}` | 清理该任务的预览、状态与成片，返回 HTTP 204（处理中或文件占用时返回 409） |

## 片段数据结构

`PUT /api/jobs/{id}/segments` 请求体示例：

```json
{
  "segments": [
    { "start_frame": 15, "end_frame": 60 },
    { "start_frame": 96, "end_frame": 138 }
  ]
}
```

- `start_frame` 从 `0` 开始，`end_frame` 为不包含的结束帧；时间换算为 `frame / media.fps`。
- 每段长度至少 1 帧，按起点升序排列，区间可相邻但不可重叠，`end_frame <= media.frame_count`，最多 50 段。

## 任务状态流转

- 分析阶段：`queued → preparing → analyzing → ready`（失败时为 `failed` 并携带 `error.message`）
- 导出阶段：`exporting → exported`（失败时回到 `ready` 并携带 `error`，允许重试）

## 裁剪框坐标系

任务就绪后，`media` 包含 `width`、`height`、`fps`、`frame_count`、`duration` 和 `has_audio`。`crop_boxes[frame]` 返回 `[x, y, w]`，以标准化预览视频像素为基准，高度按目标画幅换算 `h = w × rh / rw`。导出时按目标画幅对齐偶数像素并对源视频对应帧区间进行精确裁剪与音轨拼接。
