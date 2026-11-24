"""Fixed label-free temporal/positional explorations; no learned parameters."""
import math

def candidates(post,fps,n_use):
    s,e,lp=(post[k] for k in ('S','E','logp'))
    if not len(s)==len(e)==len(lp) or not math.isfinite(fps) or fps<=0 or n_use<1:
        raise ValueError('candidate shape/clock')
    result=[]
    for i,(a,b,l) in enumerate(zip(s,e,lp)):
        if not all(math.isfinite(float(v)) for v in (a,b,l)):raise ValueError('nonfinite candidate')
        if a<0 or b<a or a*fps>=n_use:continue
        result.append((int(a*fps),min(n_use-1,int(b*fps)),float(l),i))
    return sorted(result,key=lambda r:(-r[2],r[3]))

def choose(post,fps,n_use,arm):
    if arm not in ('A','B'):raise ValueError('unknown temporal arm')
    c=candidates(post,fps,n_use)
    if not c:return []
    a,b,lp,_=c[0];spans=[(a,b)]
    if arm=='B':
        for u,v,l,_ in c[1:]:
            if l-lp<math.log(.25):break
            if v<a or u>b:
                spans.append((u,v));break
    return sorted(spans)

def lead_x(xs,widths,frames,fps,W):
    if len(xs)!=len(widths) or not xs or fps<=0:raise ValueError('track shape/clock')
    offset=max(1,int(round(.20*fps)));out=[]
    for f in frames:
        x,w=xs[f],widths[f];future=xs[min(len(xs)-1,f+offset)]
        val=x+max(-.1*w,min(.1*w,future-x))
        out.append(int(max(0,min(W-w,round(val)))))
    return out

def center_x(xs,widths,W):
    if len(xs)!=len(widths):raise ValueError('track shape')
    result=[]
    for x,w in zip(xs,widths):
        q,r=divmod(18*x+W-w,20)
        value=q+int(2*r>20 or (2*r==20 and q%2))
        result.append(max(0,min(W-w,value)))
    return result
