"""Portable integrity helpers; no media/model execution or score claims."""
def cutoff(member,anchor,cut0,cut1):
    if member not in (0,1) or anchor not in ('T35','op10','op20'):
        raise ValueError('unknown member or anchor')
    return cut1 if member==1 or anchor in ('T35','op20') else cut0

def remove(row,gap):
    fs=[p['frame'] for p in row['predictions']];a,b=map(int,gap)
    if not fs or fs!=list(range(fs[0],fs[-1]+1)) or not fs[0]<a<b<=fs[-1]:
        raise ValueError('strict internal half-open gap required')
    return dict(row,predictions=[p for p in row['predictions'] if not a<=p['frame']<b])

def validate(rows,metadata):
    seen=set()
    for r in rows:
        v=str(r['video_id']);m=metadata[v];tw,th=r['targetRatioWH'];prev=-1
        if v in seen or not r['predictions'] or not 0<float(r['model_params_b'])<=9:
            raise ValueError('duplicate/empty/parameter declaration')
        seen.add(v)
        for p in r['predictions']:
            f=p['frame'];x,y,w=p['bboxes']
            if (type(f)!=int or not prev<f<m['n'] or any(type(z)!=int for z in (x,y,w))
                or min(x,y)<0 or w<=0 or x+w>m['W'] or y*tw+w*th>m['H']*tw):
                raise ValueError('invalid frame/continuous geometry')
            prev=f
    if seen!=set(metadata):raise ValueError('incomplete metadata coverage')
