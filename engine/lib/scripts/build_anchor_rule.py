"""Subject anchor calculation used by full-track reconstruction."""
from hlcut.spatial import SUBJECT_CLASS_WEIGHT, PERSON

def subject_anchor_v2(dets, axis, W, H, crop_len, headroom=0.03, head_align=True, short_beta=0.5, tall_omega=0.0, beta_t=0.0, weight_pow=2.0):
    """v2 of subject_anchor: parameterised short-person beta, tall-person scaling (omega),
    and weight exponent. Defaults reproduce the shipped `subject_anchor` exactly."""
    if dets is None or len(dets) == 0:
        return (None, 0.0)
    num, den = (0.0, 0.0)
    for x1, y1, x2, y2, conf, cls in dets:
        cw_ = SUBJECT_CLASS_WEIGHT.get(int(cls), 0.0)
        if cw_ <= 0:
            continue
        area = max(0.0, (x2 - x1) * (y2 - y1)) / float(W * H)
        wgt = (cw_ * conf * area) ** weight_pow
        if wgt <= 0:
            continue
        if axis == 'x':
            a = 0.5 * (x1 + x2)
        else:
            h = y2 - y1
            if head_align and int(cls) == PERSON and (h > 0.9 * crop_len):
                a = y1 + (1.0 - tall_omega) * (0.5 - headroom) * crop_len + tall_omega * beta_t * h
            elif int(cls) == PERSON and head_align:
                a = y1 + short_beta * h
            else:
                a = 0.5 * (y1 + y2)
        num += wgt * a
        den += wgt
    if den <= 0:
        return (None, 0.0)
    return (num / den, den)
