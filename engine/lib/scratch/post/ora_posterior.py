"""Full answer distribution of Video-ORA-4B for the generic highlight query.

Instead of the single greedy "<answer> S to E </answer>", score every integer answer
(S, E), 0 <= S < E <= ceil(duration), by teacher forcing on a cloned KV cache.
Writes one line per video: {video_id, dur, greedy, S: [..], E: [..], logp: [..]}.

  python scratch/post/ora_posterior.py --index data/train/../scratch/train_index.json --video-dir data/train/video --out X.jsonl --fps 2.5
"""
import argparse, copy, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hlcut.io_utils import load_index, resolve_video_path, video_meta
from hlcut.vlm import VideoLLM, TEMPORAL_GROUNDING_PROMPT, GENERIC_HIGHLIGHT_QUERY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=2.5)
    ap.add_argument("--model", default="models/Video-ORA-4B")
    ap.add_argument("--max-s", type=int, default=400)
    ap.add_argument("--top-s", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--offset", type=float, default=0.0, help="shift of the frame grid, in sampling steps")
    ap.add_argument("--max-px", type=int, default=256 * 448, help="per-frame pixel cap (token budget still applies)")
    ap.add_argument("--query", default="", help="grounding query (default: the generic highlight query)")
    args = ap.parse_args()
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["video_id"] for l in open(args.out, encoding="utf-8")}
    vlm = VideoLLM(args.model, quant="4bit")
    torch, model, proc = vlm.torch, vlm.model, vlm.processor
    tok = proc.tokenizer
    dig = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
    ids_of = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    t_to, t_sp, t_close = ids_of(" to")[0], ids_of(" ")[0], ids_of(" </")[0]
    from transformers.video_utils import VideoMetadata
    index = load_index(args.index)
    fo = open(args.out, "a", encoding="utf-8")
    for it in index:
        vid = str(it["video_id"])
        if vid in done:
            continue
        t0 = time.time()
        path = resolve_video_path(it, args.video_dir)
        n, fps, W, H = video_meta(path)
        dur = n / fps
        frames, fids = vlm.sample_frames(path, n, fps, args.fps, 128, args.max_px, 3000, args.offset)
        prompt = TEMPORAL_GROUNDING_PROMPT.format(args.query or GENERIC_HIGHLIGHT_QUERY)
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        h, w = frames[0].shape[:2]
        meta = VideoMetadata(total_num_frames=n, fps=fps, width=w, height=h, duration=dur, frames_indices=list(fids))
        inputs = proc(text=[text], videos=[np.stack(frames)], video_metadata=[meta],
                      do_sample_frames=False, do_resize=False, return_tensors="pt").to(model.device)
        with torch.no_grad():
            g = model.generate(**inputs, max_new_tokens=24, do_sample=False)
        gen = g[0, inputs["input_ids"].shape[1]:].tolist()
        greedy = tok.decode(gen, skip_special_tokens=True)
        k = next((i for i, t in enumerate(gen) if t in dig), None)
        if k is None:
            fo.write(json.dumps({"video_id": vid, "dur": dur, "greedy": greedy, "S": [], "E": [], "logp": []}) + "\n"); fo.flush()
            continue
        pre_ids = torch.cat([inputs["input_ids"], torch.tensor([gen[:k]], device=model.device)], 1)
        kw = {kk: v for kk, v in inputs.items() if kk not in ("input_ids", "attention_mask", "mm_token_type_ids")}
        if "mm_token_type_ids" in inputs:
            mt = inputs["mm_token_type_ids"]
            kw["mm_token_type_ids"] = torch.cat([mt, torch.zeros((1, k), dtype=mt.dtype, device=mt.device)], 1)
        with torch.no_grad():
            out = model(input_ids=pre_ids, attention_mask=torch.ones_like(pre_ids), use_cache=True, **kw)
        base_cache = out.past_key_values
        base_lp = torch.log_softmax(out.logits[0, -1].float(), -1)

        def expanded(B):
            c = copy.deepcopy(base_cache)
            for layer in c.layers:
                if hasattr(layer, "keys") and isinstance(getattr(layer, "keys", None), torch.Tensor):
                    layer.keys = layer.keys.repeat(B, *[1] * (layer.keys.dim() - 1))
                    layer.values = layer.values.repeat(B, *[1] * (layer.values.dim() - 1))
                for name in ("conv_states", "recurrent_states"):
                    d = getattr(layer, name, None)
                    if isinstance(d, dict):
                        for kk, v in d.items():
                            if isinstance(v, torch.Tensor):
                                d[kk] = v.repeat(B, *[1] * (v.dim() - 1))
            return c

        def score(seqs):
            """log P(seq | prefix) for token lists, batched on a repeated prefix cache."""
            res = []
            for i in range(0, len(seqs), args.batch):
                chunk = seqs[i:i + args.batch]
                B, T = len(chunk), max(len(q) for q in chunk)
                ids = torch.tensor([q + [t_close] * (T - len(q)) for q in chunk], device=model.device)
                with torch.no_grad():
                    o = model(input_ids=ids, past_key_values=expanded(B), use_cache=True)
                lps = torch.log_softmax(o.logits.float(), -1)
                for r, q in enumerate(chunk):
                    v = float(base_lp[q[0]])
                    for j in range(1, len(q)):
                        v += float(lps[r, j - 1, q[j]])
                    res.append(v)
                del o, lps
            return res

        Emax = int(math.ceil(dur))
        digs = lambda x: [dig[int(ch)] for ch in str(x)]
        Ss = list(range(0, min(Emax, args.max_s)))
        lpS = score([digs(S) + [t_to, t_sp] for S in Ss])
        top = max(lpS)
        keep = [S for S, v in sorted(zip(Ss, lpS), key=lambda z: -z[1]) if v > top - 10.0][:args.top_s]
        pairs = [(S, E) for S in keep for E in range(S + 1, Emax + 1)]
        lpP = score([digs(S) + [t_to, t_sp] + digs(E) + [t_close] for S, E in pairs])
        S_l = [S for S, _ in pairs]; E_l = [E for _, E in pairs]; P_l = lpP
        fo.write(json.dumps({"video_id": vid, "dur": dur, "greedy": greedy, "S": S_l, "E": E_l,
                             "logp": [round(x, 4) for x in P_l]}) + "\n")
        fo.flush()
        print(vid, "dur %.1f" % dur, greedy, "n=%d" % len(P_l), "%.1fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
