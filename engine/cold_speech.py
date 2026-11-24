import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_LAUNCH_BLOCKING='1')
from cold_runtime import *
import av,numpy as np,torch
from transformers import WhisperProcessor,WhisperForConditionalGeneration,CLIPTokenizerFast,CLIPTextConfig,CLIPTextModelWithProjection
from scratch.speech_text_source_oof_20261012_v1.core import uniform16,native_segments,pool_segments
from scratch.speech_transcript_feasibility_20261012_v1.core import place_samples,clip_segments
from scratch.speech_transcript_feasibility_20261012_v1.legacy_state import text_state
from scratch.speech_transcript_guard_recovery_20261012_v1.guard import digital_zero
ASR=dict(task='translate',temperature=0.,num_beams=1,do_sample=False,condition_on_prev_tokens=False,return_timestamps=True,return_segments=True,return_dict_in_generate=True,logprob_threshold=-1.,no_speech_threshold=.6,compression_ratio_threshold=None,max_new_tokens=440)

def extract(c):
    sr=16000;duration=c['n']/c['fps'];pcm=np.zeros(round(duration*sr),np.float32);covered=np.zeros(len(pcm),bool)
    with av.open(c['source']) as container:
        if not container.streams.audio:return pcm,covered,dict(status='missing_audio',start=0.,duration=duration)
        stream=container.streams.audio[0];resample=av.AudioResampler(format='fltp',layout='mono',rate=sr);count=0
        def consume(frames):
            for frame in frames:
                if frame.pts is None:raise ValueError('missing native PTS')
                place_samples(pcm,covered,frame.to_ndarray().reshape(-1),float(frame.pts*frame.time_base),0.,sr)
        for frame in container.decode(stream):
            count+=1;consume(resample.resample(frame))
            if frame.time is not None and frame.time>duration+1:break
        consume(resample.resample(None))
    assert covered.mean()>=.99 and np.isfinite(pcm).all()
    return pcm,covered,dict(status='ok',coverage=float(covered.mean()),start=0.,duration=duration,decoded_audio_frames=count,sample_count=len(pcm))

def transcript(pcm,processor,model):
    chunks=[];segments=[];errors=0
    for offset in range(0,len(pcm),480000):
        audio=pcm[offset:offset+480000];duration=len(audio)/16000;row=dict(start=offset/16000,duration=duration,digital_zero=digital_zero(audio),tokens=[],decoded=dict(text='',offsets=[]),placement_error=None)
        if row['digital_zero']:chunks.append(row);continue
        batch=processor(audio,sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
        with torch.inference_mode():gen=model.generate(**{k:v.cuda() for k,v in batch.items()},**ASR)
        torch.cuda.synchronize();row['tokens']=gen['sequences'][0].cpu().tolist()
        try:
            row['decoded']=processor.tokenizer.decode(row['tokens'],skip_special_tokens=True,output_offsets=True)
            if not isinstance(row['decoded'],dict) or 'offsets' not in row['decoded']:raise TypeError('missing timestamp offsets')
            placed,report=clip_segments(row['decoded']['offsets'],offset/16000,duration);row['placement']=report;segments.extend(placed)
        except (TypeError,ValueError,KeyError) as e:errors+=1;row['placement_error']=repr(e)
        chunks.append(row)
    return dict(chunks=chunks,segments=segments,parse_failed_chunks=errors,empty_abstention=not bool(segments))

def run():
    install_inference_IO_guard();torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=asset('CLIP-ViT-B-32');cfg=CLIPTextConfig.from_dict(read(root/'config.json')['text_config']);cfg._attn_implementation='eager';text=CLIPTextModelWithProjection(cfg)
    state=torch.load(root/'pytorch_model.bin',map_location='cpu',weights_only=True);selected,proof=text_state(state,set(text.state_dict()),text.text_model.embeddings.position_ids);text.load_state_dict(selected,strict=True);del state,selected;text.cuda().eval()
    tokenizer=CLIPTokenizerFast.from_pretrained(root,local_files_only=True);processor=WhisperProcessor.from_pretrained(asset('Whisper-small'),local_files_only=True);whisper=WhisperForConditionalGeneration.from_pretrained(asset('Whisper-small'),local_files_only=True,attn_implementation='eager',dtype=torch.float32).cuda().eval()
    cs=[c for c in cases() if (SIGNALS/'scratch/spatial_object_ranker/final_v1/test_clips'/f"{c['video_id']}.json").exists()];files={}
    for i,c in enumerate(cs):
        validate_source(c);pcm,covered,audio=extract(c);raw=transcript(pcm,processor,whisper) if audio['status']=='ok' else dict(chunks=[],segments=[],parse_failed_chunks=0,empty_abstention=True,missing_audio_abstention=True)
        segments,trimmed=native_segments(raw['segments'],c['n'],c['fps']);emb=[];trunc=[]
        for seg in segments:
            full=tokenizer(seg['text'],truncation=False)['input_ids'];batch=tokenizer(seg['text'],truncation=True,max_length=77,return_tensors='pt')
            with torch.inference_mode():a=text(**{k:v.cuda() for k,v in batch.items()}).text_embeds;a=torch.nn.functional.normalize(a,p=2,dim=-1)
            emb.append(a[0].cpu().numpy());trunc.append(dict(untruncated_tokens=len(full),truncated=len(full)>77))
        emb=np.stack(emb).astype(np.float32) if emb else np.empty((0,512),np.float32);ids=uniform16(c['n']);local,topic,mask=pool_segments(segments,emb,ids,c['fps'],c['n'],radius=3.)
        p=SIGNALS/'scratch/audiovisual_exploratory_delivery_20261013_v1/features'/c['video_id'];p.parent.mkdir(parents=True,exist_ok=True)
        with p.with_suffix('.npz').open('xb') as f:np.savez_compressed(f,ids=ids,local=local.astype(np.float32),topic=topic.astype(np.float32),mask=mask,embeddings=emb)
        write(p.with_suffix('.json'),dict(case=c,audio=audio,raw_transcript=raw,native_segments=segments,trimmed=trimmed,truncations=trunc,buffer_check=proof,pcm_SHA256=hashlib.sha256(pcm.tobytes()).hexdigest(),new_text_not_used_for_training=True))
        for q in (p.with_suffix('.npz'),p.with_suffix('.json')):files[str(q)]=sha(q)
        progress('speech',completed_cases=i+1,total_cases=len(cs),parse_failed_chunks=raw['parse_failed_chunks'])
    write(WORK/'speech_seal.json',dict(files=files,Whisper_parameters=sum(p.numel() for p in whisper.parameters()),CLIP_text_parameters=sum(p.numel() for p in text.parameters())))
    return dict(records=len(cs),Whisper_parameters=sum(p.numel() for p in whisper.parameters()),CLIP_text_parameters=sum(p.numel() for p in text.parameters()))

if __name__=='__main__':raise SystemExit(finish('speech',run))
