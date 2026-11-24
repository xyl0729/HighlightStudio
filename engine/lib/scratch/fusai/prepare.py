"""Local-only safe import of the official per-video ZIP; no video imagery output."""
import argparse
import hashlib
import json
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def archive_pairs(z):
    groups = {'video': {}, 'jsonl': {}}
    for info in z.infolist():
        name = info.filename
        if info.is_dir():
            continue
        parts = PurePosixPath(name).parts
        if '__MACOSX' in parts or any(p.startswith('._') or p == '.DS_Store' for p in parts):
            continue
        if '\\' in name or ':' in name or name.startswith('/') or '..' in parts:
            raise ValueError('Unsafe ZIP path')
        suffix = PurePosixPath(name).suffix.lower()
        kind = {'.mp4': 'video', '.jsonl': 'jsonl'}.get(suffix)
        if kind is None or len(parts) < 2 or parts[-2] != kind:
            raise ValueError('Unexpected archive entry')
        vid = PurePosixPath(name).stem
        if not re.fullmatch(r'[0-9]+', vid) or vid in groups[kind]:
            raise ValueError('Unsafe or duplicate sample identifier')
        groups[kind][vid] = info
    if not groups['video'] or groups['video'].keys() != groups['jsonl'].keys():
        raise ValueError('Archive does not contain one-to-one video/JSONL pairs')
    return [(vid, groups['video'][vid], groups['jsonl'][vid])
            for vid in sorted(groups['video'], key=lambda s: (int(s), s))]


def write_once(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Existing generated file differs; refusing overwrite')
    else:
        with path.open('xb') as f:
            f.write(data)


def prepare(archive, destination, inspect_videos=True):
    archive, destination = Path(archive).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('video', 'jsonl'):
        folder = destination / name
        folder.mkdir(exist_ok=True)
        if not folder.resolve().is_relative_to(destination):
            raise ValueError('Destination directory escapes data root')
    index, inventory, files = [], [], []
    with zipfile.ZipFile(archive) as z:
        pairs = archive_pairs(z)
        # Validate all metadata before writing any archive payload.
        for vid, video, meta in pairs:
            rows = [json.loads(s) for s in z.read(meta).decode('utf-8-sig').splitlines() if s.strip()]
            if len(rows) != 1 or not isinstance(rows[0], dict):
                raise ValueError('Expected one metadata object per sample')
            ratio = rows[0].get('targetRatioWH')
            if ratio not in ([16, 9], [9, 16]):
                raise ValueError('Unsupported or invalid target ratio')
            if set(rows[0]) != {'targetRatioWH'}:
                raise ValueError('Unexpected metadata schema; review before import')
            index.append({'video_id': vid, 'video': vid+'.mp4', 'targetRatioWH': ratio})
        for i, (vid, video, meta) in enumerate(pairs, 1):
            for info, folder, ext in ((video, 'video', '.mp4'), (meta, 'jsonl', '.jsonl')):
                out = destination / folder / (vid + ext)
                if not out.resolve().is_relative_to(destination):
                    raise ValueError('Output path escapes data root')
                h = hashlib.sha256()
                if out.exists():
                    with z.open(info) as src:
                        for block in iter(lambda: src.read(4*1024*1024), b''):
                            h.update(block)
                    if out.stat().st_size != info.file_size or digest(out) != h.hexdigest():
                        raise ValueError('Existing extracted file differs; refusing overwrite')
                else:
                    tmp = out.with_suffix(ext+'.partial')
                    if tmp.exists():
                        raise ValueError('Incomplete extraction found; inspect before retry')
                    with z.open(info) as src, tmp.open('xb') as dst:
                        for block in iter(lambda: src.read(4*1024*1024), b''):
                            h.update(block)
                            dst.write(block)
                    if tmp.stat().st_size != info.file_size:
                        raise ValueError('Extracted size mismatch')
                    tmp.rename(out)
                files.append({'path': str(out.relative_to(destination)).replace('\\','/'),
                              'bytes': info.file_size, 'sha256': h.hexdigest()})
            if inspect_videos:
                from hlcut.io_utils import video_meta
                n, fps, W, H = video_meta(str(destination/'video'/(vid+'.mp4')))
                if n <= 0 or fps <= 0 or W <= 0 or H <= 0:
                    raise ValueError('Video metadata invalid')
                inventory.append({'video_id': vid, 'n': n, 'fps': fps, 'W': W, 'H': H})
            if i % 50 == 0:
                print(json.dumps({'stage':'import','completed':i,'total':len(pairs)}),flush=True)
    payload = lambda obj: (json.dumps(obj, ensure_ascii=False, indent=2)+'\n').encode('utf-8')
    write_once(destination/'index.json', payload(index))
    write_once(destination/'smoke_index.json', payload(index[:2]))
    if inspect_videos:
        write_once(destination/'inventory.json', payload(inventory))
    manifest = {'archive_sha256': digest(archive), 'archive_bytes':archive.stat().st_size,
                'videos':len(index), 'ratios':dict(Counter(':'.join(map(str,x['targetRatioWH'])) for x in index)),
                'files':files}
    write_once(destination/'manifest.json', payload(manifest))
    summary = {k:v for k,v in manifest.items() if k != 'files'}
    if inventory:
        import numpy as np
        d = np.array([r['n']/r['fps'] for r in inventory])
        summary['duration_sec_quantiles'] = dict(zip(['min','p50','p90','max'],np.quantile(d,[0,.5,.9,1]).tolist()))
        summary['total_video_hours'] = float(d.sum()/3600)
    return summary


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive',required=True);ap.add_argument('--destination',required=True)
    args=ap.parse_args()
    print(json.dumps(prepare(args.archive,args.destination),ensure_ascii=False,indent=2))
