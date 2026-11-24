"""YOLO subject-track reconstruction using the delivered geometry and smoothing rules."""
from build_final import full_track
from hlcut.geometry import place_box, safe_crop_width

def own_track(sample, ratio):
    tw, th = map(float, ratio)
    if sample['axis'] in ('x', 'y'):
        return full_track(sample, tw, th, 12., 2., .03)
    box=list(place_box(sample['W']/2, sample['H']/2, sample['W'], sample['H'],
                       safe_crop_width(sample['W'],sample['H'],tw,th),tw,th))
    return [box]*sample['n']
