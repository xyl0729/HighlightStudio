import json
import numpy as np


def digital_zero(pcm):
    pcm=np.asarray(pcm)
    if pcm.ndim!=1 or not len(pcm) or not np.isfinite(pcm).all():raise ValueError('finite nonempty one-dimensional PCM required')
    return not bool(np.any(pcm))


def canonical(value):
    return json.loads(json.dumps(value,ensure_ascii=False,allow_nan=False))
