"""The named skeletal muscles on the skeleton he already has.

Six hundred and fifty, left and right counted apart, each one a whole
muscle. A ligament leaves the bone and meets the belly. No fibers.
No skin. Thickness and width are the muscle itself, in metres, on a
1.70 m man. His left is -x. Forward is +z.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REF_H = 1.70


def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _lerp(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def _centroid(pts):
    n = float(len(pts) or 1)
    return tuple(sum(p[i] for p in pts) / n for i in range(3))


def _along(seq, t):
    if len(seq) == 1:
        return seq[0]
    spans = [_dist(a, b) for a, b in zip(seq, seq[1:])]
    total = sum(spans)
    if total < 1e-8:
        return seq[0]
    want = max(0.0, min(1.0, t)) * total
    walked = 0.0
    for (a, b), span in zip(zip(seq, seq[1:]), spans):
        if walked + span >= want or span == 0.0:
            u = 0.0 if span == 0.0 else (want - walked) / span
            return _lerp(a, b, u)
        walked += span
    return seq[-1]


def _toward(origin, target, length):
    span = _dist(origin, target)
    if span < 1e-6:
        return origin
    step = min(length, span * 0.2)
    return _lerp(origin, target, step / span)


class Muscle:
    def __init__(self, name, radius, width, origins, insertions):
        self.name = name
        self.radius = float(radius)
        self.width = float(width)
        self.origins = tuple(origins)
        self.insertions = tuple(insertions)

    def place(self, world, side):
        sx = -1.0 if side == "l" else 1.0

        def bone_name(raw):
            return raw.format(side=side) if "{side}" in raw else raw

        def point(spec):
            bone, t, lat, up, fwd = spec
            name = bone_name(bone)
            if name not in world:
                raise KeyError(f"{self.name} has no bone {name}")
            p = _along(world[name], t)
            return (p[0] + lat * sx, p[1] + up, p[2] + fwd)

        starts = [point(s) for s in self.origins]
        ends = [point(s) for s in self.insertions]
        far_o = _centroid(ends)
        far_i = _centroid(starts)
        span = _dist(_centroid(starts), far_o)
        tendon = max(0.006, min(0.04, span * 0.1))
        ligaments = []
        near_o, near_i = [], []
        for at in starts:
            tip = _toward(at, far_o, tendon)
            ligaments.append((at, tip))
            near_o.append(tip)
        for at in ends:
            tip = _toward(at, far_i, tendon)
            ligaments.append((at, tip))
            near_i.append(tip)
        return {
            "name": self.name,
            "radius": self.radius,
            "width": max(self.width, self.radius),
            "ligaments": ligaments,
            "belly": (_centroid(near_o), _centroid(near_i)),
        }


def _catalog():
    """name, radius, width, paired, origins, insertions.

    An attachment is (bone, t, lateral, up, forward). Lateral is how
    far off the midline. The side flips it.
    """
    rows = []

    def add(name, radius, width, origins, insertions, paired=True):
        rows.append((name, radius, width, paired, origins, insertions))

    # Face. Expression, both sides.
    add("occipitalis", 0.004, 0.03, [("occipital", 0.7, 0.04, 0.0, -0.01)], [("parietal_{side}", 0.5, 0.02, 0.01, -0.01)])
    add("frontalis", 0.003, 0.035, [("frontal", 0.45, 0.025, 0.01, 0.01)], [("parietal_{side}", 0.35, 0.02, 0.02, 0.0)])
    add("orbicularis_oculi", 0.003, 0.012, [("frontal", 0.75, 0.02, -0.01, 0.02)], [("zygomatic_{side}", 0.3, 0.0, 0.01, 0.01)])
    add("corrugator_supercilii", 0.003, 0.008, [("frontal", 0.8, 0.01, -0.005, 0.015)], [("frontal", 0.55, 0.03, 0.01, 0.02)])
    add("depressor_supercilii", 0.002, 0.006, [("nasal_{side}", 0.4, 0.0, 0.005, 0.0)], [("frontal", 0.85, 0.015, -0.01, 0.01)])
    add("procerus", 0.002, 0.008, [("nasal_{side}", 0.3, 0.0, 0.0, 0.0)], [("frontal", 0.9, 0.008, -0.01, 0.01)])
    add("nasalis", 0.003, 0.01, [("maxilla_{side}", 0.4, 0.0, 0.005, 0.005)], [("nasal_{side}", 0.7, 0.0, -0.005, 0.005)])
    add("levator_labii_superioris_alaeque_nasi", 0.003, 0.006, [("frontal", 0.95, 0.008, -0.02, 0.01)], [("maxilla_{side}", 0.7, 0.0, -0.01, 0.01)])
    add("levator_labii_superioris", 0.004, 0.01, [("maxilla_{side}", 0.25, 0.005, 0.01, 0.01)], [("maxilla_{side}", 0.85, 0.01, -0.01, 0.015)])
    add("zygomaticus_minor", 0.003, 0.006, [("zygomatic_{side}", 0.4, 0.0, 0.01, 0.01)], [("maxilla_{side}", 0.8, 0.005, -0.005, 0.01)])
    add("zygomaticus_major", 0.004, 0.008, [("zygomatic_{side}", 0.7, 0.005, 0.0, 0.005)], [("mandible", 0.42, 0.03, 0.0, 0.02)])
    add("levator_anguli_oris", 0.003, 0.007, [("maxilla_{side}", 0.55, 0.01, 0.0, 0.01)], [("mandible", 0.4, 0.028, 0.0, 0.015)])
    add("risorius", 0.002, 0.01, [("zygomatic_{side}", 0.9, 0.01, -0.02, 0.0)], [("mandible", 0.38, 0.035, -0.005, 0.02)])
    add("buccinator", 0.004, 0.02, [("maxilla_{side}", 0.65, 0.015, -0.02, 0.0)], [("mandible", 0.5, 0.02, -0.01, 0.015)])
    add("depressor_anguli_oris", 0.003, 0.008, [("mandible", 0.7, 0.03, -0.02, 0.01)], [("mandible", 0.4, 0.032, 0.0, 0.02)])
    add("depressor_labii_inferioris", 0.003, 0.008, [("mandible", 0.65, 0.02, -0.015, 0.015)], [("mandible", 0.45, 0.015, 0.0, 0.02)])
    add("mentalis", 0.004, 0.008, [("mandible", 0.8, 0.008, -0.01, 0.015)], [("mandible", 0.55, 0.008, 0.005, 0.02)])
    add("orbicularis_oris", 0.004, 0.012, [("maxilla_{side}", 0.75, 0.012, -0.01, 0.015)], [("mandible", 0.48, 0.012, 0.0, 0.02)])
    add("auricularis_anterior", 0.002, 0.008, [("temporal_{side}", 0.3, 0.01, 0.02, 0.01)], [("temporal_{side}", 0.55, 0.02, 0.0, 0.0)])
    add("auricularis_superior", 0.002, 0.01, [("parietal_{side}", 0.6, 0.03, 0.0, 0.0)], [("temporal_{side}", 0.2, 0.015, 0.02, 0.0)])
    add("auricularis_posterior", 0.002, 0.008, [("temporal_{side}", 0.8, 0.0, 0.0, -0.01)], [("occipital", 0.4, 0.05, 0.0, 0.0)])
    add("platysma", 0.002, 0.04, [("clavicle_{side}", 0.4, 0.0, -0.01, 0.02)], [("mandible", 0.6, 0.04, -0.01, 0.01)])

    # Jaw.
    add("masseter", 0.012, 0.035, [("zygomatic_{side}", 0.55, 0.0, -0.005, 0.0)], [("mandible", 0.62, 0.045, -0.02, 0.0)])
    add("temporalis", 0.008, 0.05, [("temporal_{side}", 0.35, 0.0, 0.02, 0.0)], [("mandible", 0.25, 0.04, 0.01, 0.01)])
    add("medial_pterygoid", 0.008, 0.015, [("sphenoid", 0.6, 0.02, -0.01, 0.0)], [("mandible", 0.58, 0.03, -0.02, 0.0)])
    add("lateral_pterygoid", 0.006, 0.012, [("sphenoid", 0.4, 0.025, 0.0, 0.01)], [("mandible", 0.2, 0.04, 0.0, 0.015)])

    # Eye and ear.
    for name, up, fwd in (
        ("superior_rectus", 0.012, 0.02),
        ("inferior_rectus", -0.012, 0.02),
        ("medial_rectus", 0.0, 0.02),
        ("lateral_rectus", 0.0, 0.02),
        ("superior_oblique", 0.014, 0.015),
        ("inferior_oblique", -0.01, 0.02),
    ):
        lat = 0.012 if "lateral" in name else 0.006
        if "medial" in name:
            lat = 0.002
        add(name, 0.003, 0.004, [("sphenoid", 0.5, lat, up, fwd * 0.3)], [("zygomatic_{side}", 0.15, 0.0, up * 0.3, 0.02)])
    add("levator_palpebrae", 0.002, 0.01, [("sphenoid", 0.45, 0.01, 0.015, 0.02)], [("frontal", 0.8, 0.025, -0.005, 0.02)])
    add("tensor_tympani", 0.001, 0.002, [("temporal_{side}", 0.6, 0.0, 0.0, 0.0)], [("malleus_{side}", 0.5, 0.0, 0.0, 0.0)])
    add("stapedius", 0.001, 0.0015, [("temporal_{side}", 0.7, 0.0, -0.005, 0.0)], [("stapes_{side}", 0.5, 0.0, 0.0, 0.0)])

    # Tongue, palate, throat.
    add("genioglossus", 0.008, 0.015, [("mandible", 0.35, 0.008, 0.0, 0.01)], [("hyoid", 0.4, 0.008, 0.01, 0.0)])
    add("hyoglossus", 0.005, 0.01, [("hyoid", 0.7, 0.015, 0.0, 0.0)], [("mandible", 0.3, 0.015, 0.015, 0.01)])
    add("styloglossus", 0.004, 0.006, [("temporal_{side}", 0.85, 0.0, -0.02, 0.0)], [("hyoid", 0.3, 0.01, 0.01, 0.01)])
    add("palatoglossus", 0.003, 0.006, [("palatine_{side}", 0.5, 0.0, 0.01, 0.0)], [("hyoid", 0.25, 0.01, 0.015, 0.005)])
    add("superior_longitudinal", 0.004, 0.012, [("mandible", 0.32, 0.006, 0.01, 0.02)], [("hyoid", 0.2, 0.006, 0.02, 0.015)])
    add("inferior_longitudinal", 0.004, 0.01, [("mandible", 0.4, 0.008, -0.005, 0.01)], [("hyoid", 0.35, 0.008, 0.0, 0.005)])
    add("transverse_tongue", 0.004, 0.008, [("mandible", 0.36, 0.004, 0.0, 0.015)], [("mandible", 0.36, 0.02, 0.0, 0.015)])
    add("vertical_tongue", 0.003, 0.008, [("mandible", 0.34, 0.01, 0.012, 0.02)], [("mandible", 0.38, 0.01, -0.005, 0.012)])
    add("levator_veli_palatini", 0.004, 0.008, [("temporal_{side}", 0.5, 0.0, 0.0, 0.0)], [("palatine_{side}", 0.4, 0.0, 0.01, 0.0)])
    add("tensor_veli_palatini", 0.004, 0.008, [("sphenoid", 0.55, 0.02, 0.0, 0.0)], [("palatine_{side}", 0.6, 0.0, 0.0, 0.0)])
    add("palatopharyngeus", 0.004, 0.008, [("palatine_{side}", 0.5, 0.0, 0.0, 0.0)], [("hyoid", 0.55, 0.012, -0.01, 0.0)])
    add("musculus_uvulae", 0.003, 0.004, [("palatine_l", 0.8, 0.0, 0.0, 0.0)], [("palatine_r", 0.8, 0.0, 0.0, 0.0)], paired=False)
    add("superior_constrictor", 0.004, 0.02, [("sphenoid", 0.7, 0.02, -0.02, 0.0)], [("occipital", 0.2, 0.01, -0.04, -0.01)])
    add("middle_constrictor", 0.004, 0.018, [("hyoid", 0.8, 0.015, 0.0, 0.0)], [("C3", 0.5, 0.015, 0.0, -0.02)])
    add("inferior_constrictor", 0.005, 0.02, [("sternum", 0.15, 0.02, 0.02, 0.0)], [("C6", 0.5, 0.018, 0.0, -0.02)])
    add("stylopharyngeus", 0.003, 0.005, [("temporal_{side}", 0.9, 0.0, -0.015, 0.0)], [("hyoid", 0.45, 0.012, 0.0, 0.0)])
    add("salpingopharyngeus", 0.002, 0.004, [("temporal_{side}", 0.4, 0.0, 0.0, 0.0)], [("hyoid", 0.4, 0.01, 0.01, 0.0)])
    add("cricothyroid", 0.004, 0.01, [("sternum", 0.05, 0.012, 0.04, 0.02)], [("hyoid", 0.6, 0.01, -0.01, 0.01)])
    add("posterior_cricoarytenoid", 0.003, 0.006, [("sternum", 0.08, 0.01, 0.03, 0.0)], [("hyoid", 0.65, 0.008, 0.0, -0.005)])
    add("lateral_cricoarytenoid", 0.003, 0.006, [("sternum", 0.1, 0.016, 0.03, 0.015)], [("hyoid", 0.62, 0.012, 0.0, 0.01)])
    add("thyroarytenoid", 0.004, 0.008, [("hyoid", 0.5, 0.008, -0.005, 0.012)], [("sternum", 0.12, 0.008, 0.035, 0.02)])
    add("vocalis", 0.002, 0.004, [("hyoid", 0.48, 0.004, 0.0, 0.014)], [("sternum", 0.11, 0.004, 0.03, 0.02)])
    add("oblique_arytenoid", 0.002, 0.005, [("hyoid", 0.58, 0.006, -0.008, 0.008)], [("hyoid", 0.58, 0.014, -0.004, 0.006)])
    add("transverse_arytenoid", 0.003, 0.006, [("hyoid", 0.55, -0.008, -0.005, 0.01)], [("hyoid", 0.55, 0.008, -0.005, 0.01)], paired=False)

    # Hyoid straps and the neck.
    add("digastric", 0.005, 0.008, [("temporal_{side}", 0.75, 0.0, -0.02, 0.0)], [("mandible", 0.3, 0.02, -0.02, 0.02)])
    add("stylohyoid", 0.003, 0.005, [("temporal_{side}", 0.88, 0.0, -0.015, 0.0)], [("hyoid", 0.75, 0.012, 0.005, 0.0)])
    add("mylohyoid", 0.004, 0.02, [("mandible", 0.45, 0.03, -0.015, 0.0)], [("hyoid", 0.5, 0.0, 0.0, 0.008)])
    add("geniohyoid", 0.004, 0.008, [("mandible", 0.28, 0.006, -0.01, 0.012)], [("hyoid", 0.35, 0.0, 0.005, 0.01)])
    add("sternohyoid", 0.005, 0.012, [("sternum", 0.7, 0.015, 0.0, 0.01)], [("hyoid", 0.7, 0.012, -0.005, 0.008)])
    add("omohyoid", 0.004, 0.008, [("scapula_{side}", 0.8, 0.0, -0.01, 0.0)], [("hyoid", 0.8, 0.016, 0.0, 0.005)])
    add("sternothyroid", 0.005, 0.012, [("sternum", 0.55, 0.018, 0.0, 0.008)], [("hyoid", 0.85, 0.014, -0.015, 0.005)])
    add("thyrohyoid", 0.004, 0.01, [("hyoid", 0.8, 0.01, -0.012, 0.006)], [("hyoid", 0.45, 0.01, 0.0, 0.01)])
    add("sternocleidomastoid", 0.01, 0.025, [("sternum", 0.82, 0.02, 0.0, 0.015), ("clavicle_{side}", 0.35, 0.0, 0.0, 0.015)], [("temporal_{side}", 0.65, 0.0, -0.01, 0.0)])
    add("scalene_anterior", 0.006, 0.012, [("C4", 0.5, 0.02, 0.0, 0.005)], [("rib_1_{side}", 0.4, 0.0, 0.0, 0.01)])
    add("scalene_middle", 0.007, 0.014, [("C3", 0.5, 0.025, 0.0, 0.0)], [("rib_1_{side}", 0.65, 0.01, 0.0, 0.0)])
    add("scalene_posterior", 0.005, 0.01, [("C5", 0.5, 0.028, 0.0, -0.008)], [("rib_2_{side}", 0.5, 0.01, 0.0, 0.0)])
    add("longus_colli", 0.005, 0.01, [("C5", 0.5, 0.012, 0.0, 0.012)], [("T3", 0.5, 0.015, 0.0, 0.01)])
    add("longus_capitis", 0.005, 0.01, [("occipital", 0.3, 0.02, -0.01, 0.01)], [("C6", 0.5, 0.015, 0.0, 0.012)])
    add("rectus_capitis_anterior", 0.004, 0.008, [("occipital", 0.25, 0.015, -0.015, 0.01)], [("C1", 0.6, 0.012, 0.0, 0.01)])
    add("rectus_capitis_lateralis", 0.003, 0.006, [("occipital", 0.35, 0.03, -0.01, 0.0)], [("C1", 0.5, 0.02, 0.0, 0.0)])
    add("rectus_capitis_posterior_major", 0.005, 0.01, [("C2", 0.5, 0.008, 0.01, -0.015)], [("occipital", 0.55, 0.02, 0.0, -0.01)])
    add("rectus_capitis_posterior_minor", 0.004, 0.008, [("C1", 0.5, 0.006, 0.008, -0.012)], [("occipital", 0.5, 0.012, 0.005, -0.012)])
    add("obliquus_capitis_superior", 0.004, 0.008, [("C1", 0.5, 0.02, 0.0, 0.0)], [("occipital", 0.45, 0.03, 0.01, -0.005)])
    add("obliquus_capitis_inferior", 0.005, 0.008, [("C2", 0.5, 0.01, 0.0, -0.012)], [("C1", 0.5, 0.022, 0.0, 0.0)])

    # Back, the columns and the segments.
    add("splenius_capitis", 0.008, 0.02, [("C7", 0.5, 0.015, 0.0, -0.015)], [("occipital", 0.6, 0.035, 0.0, -0.008)])
    add("splenius_cervicis", 0.005, 0.01, [("T3", 0.5, 0.018, 0.0, -0.015)], [("C2", 0.5, 0.02, 0.0, -0.01)])
    add("semispinalis_capitis", 0.008, 0.025, [("C7", 0.5, 0.02, 0.0, -0.012)], [("occipital", 0.65, 0.02, 0.015, -0.01)])
    add("semispinalis_cervicis", 0.006, 0.012, [("T5", 0.5, 0.02, 0.0, -0.015)], [("C2", 0.5, 0.01, 0.0, -0.012)])
    add("semispinalis_thoracis", 0.006, 0.014, [("T11", 0.5, 0.02, 0.0, -0.015)], [("T2", 0.5, 0.015, 0.0, -0.015)])
    add("iliocostalis_cervicis", 0.006, 0.012, [("rib_6_{side}", 0.7, 0.0, 0.01, -0.01)], [("C4", 0.5, 0.02, 0.0, -0.01)])
    add("iliocostalis_thoracis", 0.008, 0.016, [("rib_12_{side}", 0.6, 0.0, 0.0, -0.01)], [("rib_1_{side}", 0.6, 0.0, 0.01, -0.01)])
    add("iliocostalis_lumborum", 0.01, 0.02, [("sacrum", 0.4, 0.04, 0.0, -0.01), ("hip_{side}", 0.5, 0.0, 0.02, -0.02)], [("rib_7_{side}", 0.55, 0.0, 0.0, -0.015)])
    add("longissimus_capitis", 0.006, 0.012, [("C5", 0.5, 0.02, 0.0, -0.01)], [("temporal_{side}", 0.7, 0.0, 0.0, -0.01)])
    add("longissimus_cervicis", 0.006, 0.012, [("T4", 0.5, 0.02, 0.0, -0.012)], [("C4", 0.5, 0.018, 0.0, -0.01)])
    add("longissimus_thoracis", 0.012, 0.025, [("sacrum", 0.5, 0.03, 0.0, -0.015)], [("T2", 0.5, 0.02, 0.0, -0.012)])
    add("spinalis_cervicis", 0.004, 0.008, [("C6", 0.5, 0.006, 0.0, -0.012)], [("C2", 0.5, 0.006, 0.0, -0.012)])
    add("spinalis_thoracis", 0.006, 0.01, [("T11", 0.5, 0.008, 0.0, -0.018)], [("T2", 0.5, 0.008, 0.0, -0.015)])
    add("trapezius", 0.008, 0.08, [("occipital", 0.5, 0.02, 0.02, -0.01), ("T8", 0.5, 0.015, 0.0, -0.02)], [("clavicle_{side}", 0.85, 0.0, 0.01, 0.0), ("scapula_{side}", 0.35, 0.0, 0.01, -0.01)])
    add("latissimus", 0.01, 0.1, [("sacrum", 0.55, 0.05, 0.0, -0.02), ("T7", 0.5, 0.02, 0.0, -0.02)], [("humerus_{side}", 0.18, 0.0, 0.0, -0.015)])
    add("rhomboid_major", 0.006, 0.03, [("T3", 0.5, 0.01, 0.0, -0.018)], [("scapula_{side}", 0.7, 0.0, -0.01, -0.01)])
    add("rhomboid_minor", 0.004, 0.015, [("C7", 0.5, 0.012, 0.0, -0.015)], [("scapula_{side}", 0.25, 0.0, 0.01, -0.01)])
    add("levator_scapulae", 0.006, 0.015, [("C2", 0.5, 0.02, 0.0, -0.008)], [("scapula_{side}", 0.15, 0.0, 0.02, 0.0)])
    add("serratus_posterior_superior", 0.003, 0.02, [("C7", 0.5, 0.015, 0.0, -0.012)], [("rib_3_{side}", 0.5, 0.0, 0.01, 0.0)])
    add("serratus_posterior_inferior", 0.003, 0.025, [("T11", 0.5, 0.02, 0.0, -0.015)], [("rib_10_{side}", 0.55, 0.0, 0.0, -0.01)])
    add("quadratus_lumborum", 0.01, 0.025, [("hip_{side}", 0.45, 0.0, 0.04, -0.02)], [("rib_12_{side}", 0.4, 0.0, 0.0, 0.0), ("L1", 0.5, 0.02, 0.0, -0.01)])

    spine = ["C1", "C2", "C3", "C4", "C5", "C6", "C7",
             "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12",
             "L1", "L2", "L3", "L4", "L5"]
    for i, name in enumerate(spine):
        if i == 0:
            continue
        below = spine[i - 1]
        add(f"multifidus_{name.lower()}", 0.004, 0.008,
            [(below, 0.5, 0.012, 0.0, -0.012)], [(name, 0.5, 0.006, 0.0, -0.016)])
        if name[0] in "TL":
            add(f"rotator_{name.lower()}", 0.003, 0.005,
                [(below, 0.5, 0.016, 0.0, -0.006)], [(name, 0.5, 0.004, 0.0, -0.016)])
    for a, b in (("C2", "C3"), ("C3", "C4"), ("C4", "C5"), ("C5", "C6"), ("C6", "C7"),
                 ("L1", "L2"), ("L2", "L3"), ("L3", "L4"), ("L4", "L5")):
        add(f"interspinalis_{a.lower()}", 0.003, 0.004,
            [(a, 0.6, 0.004, 0.0, -0.014)], [(b, 0.4, 0.004, 0.0, -0.014)])
        add(f"intertransversarius_{a.lower()}", 0.003, 0.005,
            [(a, 0.5, 0.018, 0.0, 0.0)], [(b, 0.5, 0.018, 0.0, 0.0)])

    # Ribs and the breathing muscles.
    for i in range(1, 12):
        add(f"external_intercostal_{i}", 0.003, 0.02,
            [(f"rib_{i}_{{side}}", 0.45, 0.0, 0.0, 0.012)], [(f"rib_{i + 1}_{{side}}", 0.7, 0.0, 0.0, 0.008)])
        add(f"internal_intercostal_{i}", 0.003, 0.018,
            [(f"rib_{i}_{{side}}", 0.6, 0.0, 0.0, 0.006)], [(f"rib_{i + 1}_{{side}}", 0.35, 0.0, 0.0, 0.01)])
    for i in range(3, 11):
        add(f"innermost_intercostal_{i}", 0.002, 0.012,
            [(f"rib_{i}_{{side}}", 0.5, 0.0, 0.0, 0.002)], [(f"rib_{i + 1}_{{side}}", 0.5, 0.0, 0.0, 0.002)])
    for i in range(1, 13):
        add(f"levator_costarum_{i}", 0.003, 0.006,
            [(f"T{i}" if i < 12 else "T12", 0.5, 0.016, 0.0, -0.008)], [(f"rib_{i}_{{side}}", 0.25, 0.0, 0.005, 0.0)])
    add("transversus_thoracis", 0.003, 0.02, [("sternum", 0.6, 0.015, 0.0, -0.005)], [("rib_4_{side}", 0.3, 0.0, 0.0, 0.01)])
    add("subcostalis", 0.003, 0.01, [("rib_10_{side}", 0.4, 0.0, 0.0, 0.0)], [("rib_8_{side}", 0.5, 0.0, 0.0, 0.0)])
    add("diaphragm_costal", 0.004, 0.04, [("rib_8_{side}", 0.3, 0.0, -0.01, 0.01)], [("sternum", 0.95, 0.0, -0.02, 0.0)])
    add("diaphragm_crus", 0.006, 0.012, [("L2", 0.5, 0.015, -0.01, 0.0)], [("sternum", 0.98, 0.0, -0.03, 0.0)])

    # Belly and the hip flexors.
    add("rectus_abdominis", 0.012, 0.05, [("sternum", 0.9, 0.02, -0.01, 0.02)], [("hip_{side}", 0.15, 0.0, -0.02, 0.03)])
    add("external_oblique", 0.008, 0.08, [("rib_8_{side}", 0.4, 0.0, 0.0, 0.02)], [("hip_{side}", 0.25, 0.02, 0.0, 0.04)])
    add("internal_oblique", 0.008, 0.07, [("hip_{side}", 0.4, 0.0, 0.02, 0.0)], [("rib_10_{side}", 0.35, 0.0, 0.0, 0.015)])
    add("transversus_abdominis", 0.005, 0.06, [("hip_{side}", 0.55, 0.0, 0.0, -0.01)], [("sternum", 0.95, 0.01, -0.02, 0.01)])
    add("pyramidalis", 0.003, 0.01, [("hip_{side}", 0.12, 0.0, -0.01, 0.03)], [("hip_{side}", 0.2, 0.01, 0.02, 0.03)])
    add("psoas_major", 0.014, 0.02, [("T12", 0.5, 0.02, -0.01, 0.01), ("L3", 0.5, 0.022, -0.01, 0.012)], [("femur_{side}", 0.08, 0.0, 0.0, 0.015)])
    add("psoas_minor", 0.004, 0.008, [("T12", 0.5, 0.018, 0.0, 0.015)], [("hip_{side}", 0.3, 0.0, 0.02, 0.02)])
    add("iliacus", 0.012, 0.03, [("hip_{side}", 0.5, 0.0, 0.03, 0.02)], [("femur_{side}", 0.1, 0.0, 0.005, 0.012)])

    # Shoulder and arm.
    add("pectoralis_major", 0.014, 0.12, [("clavicle_{side}", 0.6, 0.0, -0.01, 0.015), ("sternum", 0.5, 0.03, 0.0, 0.02)], [("humerus_{side}", 0.16, 0.0, 0.0, 0.02)])
    add("pectoralis_minor", 0.006, 0.03, [("rib_3_{side}", 0.25, 0.0, 0.0, 0.015)], [("scapula_{side}", 0.2, 0.0, 0.01, 0.02)])
    add("subclavius", 0.004, 0.008, [("rib_1_{side}", 0.2, 0.0, 0.01, 0.01)], [("clavicle_{side}", 0.5, 0.0, -0.008, 0.0)])
    add("serratus_anterior", 0.006, 0.05, [("rib_4_{side}", 0.35, 0.0, 0.0, 0.02)], [("scapula_{side}", 0.75, 0.0, 0.0, -0.01)])
    add("deltoid", 0.018, 0.06, [("clavicle_{side}", 0.95, 0.0, 0.01, 0.01), ("scapula_{side}", 0.3, 0.0, 0.015, -0.01)], [("humerus_{side}", 0.4, 0.012, 0.0, 0.0)])
    add("supraspinatus", 0.008, 0.02, [("scapula_{side}", 0.2, 0.0, 0.02, -0.01)], [("humerus_{side}", 0.06, 0.01, 0.015, 0.0)])
    add("infraspinatus", 0.01, 0.035, [("scapula_{side}", 0.6, 0.0, -0.005, -0.02)], [("humerus_{side}", 0.08, 0.015, 0.0, -0.01)])
    add("teres_minor", 0.006, 0.012, [("scapula_{side}", 0.85, 0.0, -0.01, -0.01)], [("humerus_{side}", 0.1, 0.012, 0.0, -0.012)])
    add("teres_major", 0.01, 0.02, [("scapula_{side}", 0.9, 0.0, -0.02, -0.01)], [("humerus_{side}", 0.14, 0.0, 0.0, -0.012)])
    add("subscapularis", 0.012, 0.04, [("scapula_{side}", 0.5, 0.0, 0.0, 0.015)], [("humerus_{side}", 0.07, 0.0, 0.0, 0.015)])
    add("coracobrachialis", 0.008, 0.012, [("scapula_{side}", 0.18, 0.0, 0.01, 0.02)], [("humerus_{side}", 0.55, 0.0, 0.0, 0.012)])
    add("biceps_brachii", 0.016, 0.032, [("scapula_{side}", 0.22, 0.0, 0.015, 0.015)], [("radius_{side}", 0.08, 0.0, 0.0, 0.012)])
    add("brachialis", 0.014, 0.025, [("humerus_{side}", 0.55, 0.0, 0.0, 0.012)], [("ulna_{side}", 0.06, 0.0, 0.0, 0.008)])
    add("triceps_brachii", 0.016, 0.035, [("scapula_{side}", 0.4, 0.0, 0.0, -0.015), ("humerus_{side}", 0.5, 0.0, 0.0, -0.012)], [("ulna_{side}", 0.04, 0.0, 0.0, -0.008)])
    add("anconeus", 0.005, 0.012, [("humerus_{side}", 0.92, 0.01, 0.0, -0.008)], [("ulna_{side}", 0.12, 0.0, 0.0, -0.006)])

    # Forearm.
    add("pronator_teres", 0.008, 0.014, [("humerus_{side}", 0.95, 0.0, 0.0, 0.01)], [("radius_{side}", 0.45, 0.008, 0.0, 0.006)])
    add("flexor_carpi_radialis", 0.007, 0.012, [("humerus_{side}", 0.96, 0.008, 0.0, 0.01)], [("metacarpal_2_{side}", 0.15, 0.0, 0.0, 0.004)])
    add("palmaris_longus", 0.004, 0.006, [("humerus_{side}", 0.96, 0.0, 0.0, 0.012)], [("metacarpal_3_{side}", 0.05, 0.0, -0.01, 0.006)])
    add("flexor_carpi_ulnaris", 0.008, 0.014, [("humerus_{side}", 0.96, -0.01, 0.0, 0.004)], [("metacarpal_5_{side}", 0.12, 0.0, 0.0, 0.0)])
    add("flexor_digitorum_superficialis", 0.008, 0.016, [("humerus_{side}", 0.94, 0.0, -0.005, 0.01)], [("ph_hand_3_mid_{side}", 0.3, 0.0, -0.004, 0.004)])
    add("flexor_digitorum_profundus", 0.008, 0.014, [("ulna_{side}", 0.4, 0.0, 0.0, 0.006)], [("ph_hand_3_dist_{side}", 0.3, 0.0, -0.003, 0.003)])
    add("flexor_pollicis_longus", 0.005, 0.008, [("radius_{side}", 0.5, 0.0, 0.0, 0.006)], [("ph_hand_1_dist_{side}", 0.4, 0.0, 0.0, 0.003)])
    add("pronator_quadratus", 0.004, 0.015, [("ulna_{side}", 0.9, 0.0, 0.0, 0.0)], [("radius_{side}", 0.92, 0.0, 0.0, 0.004)])
    add("brachioradialis", 0.008, 0.014, [("humerus_{side}", 0.85, 0.012, 0.0, 0.0)], [("radius_{side}", 0.95, 0.0, 0.0, 0.004)])
    add("extensor_carpi_radialis_longus", 0.006, 0.01, [("humerus_{side}", 0.9, 0.014, 0.005, -0.004)], [("metacarpal_2_{side}", 0.2, 0.0, 0.004, -0.002)])
    add("extensor_carpi_radialis_brevis", 0.006, 0.01, [("humerus_{side}", 0.96, 0.01, 0.0, -0.006)], [("metacarpal_3_{side}", 0.18, 0.0, 0.003, -0.002)])
    add("extensor_digitorum", 0.006, 0.014, [("humerus_{side}", 0.96, 0.0, 0.0, -0.008)], [("ph_hand_3_dist_{side}", 0.2, 0.0, 0.004, -0.002)])
    add("extensor_digiti_minimi", 0.003, 0.005, [("humerus_{side}", 0.96, -0.008, 0.0, -0.006)], [("ph_hand_5_mid_{side}", 0.3, 0.0, 0.002, 0.0)])
    add("extensor_carpi_ulnaris", 0.006, 0.01, [("humerus_{side}", 0.96, -0.012, 0.0, -0.004)], [("metacarpal_5_{side}", 0.2, 0.0, 0.0, -0.003)])
    add("supinator", 0.006, 0.015, [("ulna_{side}", 0.15, 0.0, 0.0, -0.006)], [("radius_{side}", 0.35, 0.006, 0.0, 0.0)])
    add("abductor_pollicis_longus", 0.005, 0.008, [("ulna_{side}", 0.45, 0.0, 0.0, -0.004)], [("metacarpal_1_{side}", 0.2, 0.0, 0.0, 0.0)])
    add("extensor_pollicis_brevis", 0.003, 0.005, [("radius_{side}", 0.55, 0.0, 0.0, -0.004)], [("ph_hand_1_prox_{side}", 0.4, 0.0, 0.0, 0.0)])
    add("extensor_pollicis_longus", 0.004, 0.006, [("ulna_{side}", 0.55, 0.0, 0.0, -0.005)], [("ph_hand_1_dist_{side}", 0.35, 0.0, 0.0, 0.0)])
    add("extensor_indicis", 0.003, 0.005, [("ulna_{side}", 0.7, 0.0, 0.0, -0.004)], [("ph_hand_2_dist_{side}", 0.25, 0.0, 0.002, 0.0)])

    # Hand.
    add("abductor_pollicis_brevis", 0.004, 0.008, [("scaphoid_{side}", 0.6, 0.0, 0.0, 0.004)], [("ph_hand_1_prox_{side}", 0.3, 0.0, 0.0, 0.003)])
    add("opponens_pollicis", 0.004, 0.008, [("trapezium_{side}", 0.5, 0.0, 0.0, 0.003)], [("metacarpal_1_{side}", 0.7, 0.0, 0.0, 0.0)])
    add("flexor_pollicis_brevis", 0.004, 0.008, [("trapezium_{side}", 0.4, 0.0, -0.003, 0.003)], [("ph_hand_1_prox_{side}", 0.35, 0.0, -0.002, 0.002)])
    add("adductor_pollicis", 0.005, 0.012, [("metacarpal_3_{side}", 0.4, 0.0, 0.0, 0.0)], [("ph_hand_1_prox_{side}", 0.4, 0.0, 0.0, 0.0)])
    add("abductor_digiti_minimi_hand", 0.003, 0.006, [("pisiform_{side}", 0.5, 0.0, 0.0, 0.0)], [("ph_hand_5_prox_{side}", 0.3, 0.0, 0.0, 0.0)])
    add("flexor_digiti_minimi_hand", 0.003, 0.005, [("hamate_{side}", 0.5, 0.0, -0.002, 0.002)], [("ph_hand_5_prox_{side}", 0.35, 0.0, -0.002, 0.0)])
    add("opponens_digiti_minimi", 0.003, 0.006, [("hamate_{side}", 0.6, 0.0, 0.0, 0.0)], [("metacarpal_5_{side}", 0.7, 0.0, 0.0, 0.0)])
    add("palmaris_brevis", 0.002, 0.01, [("metacarpal_5_{side}", 0.1, 0.0, -0.006, 0.004)], [("metacarpal_5_{side}", 0.35, 0.004, -0.004, 0.004)])
    for d in range(1, 5):
        add(f"lumbrical_hand_{d}", 0.002, 0.004,
            [(f"metacarpal_{d + 1}_{{side}}", 0.3, 0.0, -0.004, 0.004)], [(f"ph_hand_{d + 1}_prox_{{side}}", 0.2, 0.0, 0.003, 0.0)])
    for d in range(1, 5):
        add(f"dorsal_interosseous_hand_{d}", 0.003, 0.006,
            [(f"metacarpal_{d}_{{side}}", 0.5, 0.004, 0.0, 0.0)], [(f"ph_hand_{min(d, 4)}_prox_{{side}}", 0.25, 0.0, 0.003, 0.0)])
    for d in range(1, 4):
        add(f"palmar_interosseous_hand_{d}", 0.002, 0.004,
            [(f"metacarpal_{d + 1}_{{side}}", 0.55, 0.0, -0.003, 0.0)], [(f"ph_hand_{d + 1}_prox_{{side}}", 0.25, 0.0, 0.0, 0.003)])

    # Hip and thigh. Radii are the belly, so the thigh fills out to a man's.
    add("gluteus_maximus", 0.028, 0.12, [("sacrum", 0.45, 0.04, 0.0, -0.02), ("hip_{side}", 0.7, 0.0, 0.0, -0.03)], [("femur_{side}", 0.18, 0.02, 0.0, -0.02)])
    add("gluteus_medius", 0.018, 0.06, [("hip_{side}", 0.55, 0.02, 0.04, -0.01)], [("femur_{side}", 0.08, 0.025, 0.01, 0.0)])
    add("gluteus_minimus", 0.012, 0.035, [("hip_{side}", 0.5, 0.0, 0.03, 0.0)], [("femur_{side}", 0.07, 0.02, 0.012, 0.005)])
    add("tensor_fasciae_latae", 0.008, 0.02, [("hip_{side}", 0.35, 0.01, 0.04, 0.03)], [("tibia_{side}", 0.12, 0.02, 0.0, 0.0)])
    add("piriformis", 0.008, 0.016, [("sacrum", 0.6, 0.03, 0.0, -0.01)], [("femur_{side}", 0.09, 0.02, 0.0, -0.012)])
    add("gemellus_superior", 0.004, 0.006, [("hip_{side}", 0.8, 0.0, 0.0, -0.02)], [("femur_{side}", 0.1, 0.018, 0.0, -0.01)])
    add("gemellus_inferior", 0.004, 0.006, [("hip_{side}", 0.85, 0.0, -0.02, -0.015)], [("femur_{side}", 0.11, 0.016, 0.0, -0.01)])
    add("obturator_internus", 0.008, 0.015, [("hip_{side}", 0.6, 0.0, -0.01, 0.0)], [("femur_{side}", 0.1, 0.02, 0.0, -0.008)])
    add("obturator_externus", 0.007, 0.014, [("hip_{side}", 0.4, 0.0, -0.02, 0.02)], [("femur_{side}", 0.12, 0.015, 0.0, -0.008)])
    add("quadratus_femoris", 0.008, 0.02, [("hip_{side}", 0.9, 0.0, -0.02, -0.01)], [("femur_{side}", 0.14, 0.02, 0.0, -0.01)])
    add("sartorius", 0.008, 0.015, [("hip_{side}", 0.22, 0.0, 0.03, 0.03)], [("tibia_{side}", 0.08, 0.0, 0.0, 0.015)])
    add("rectus_femoris", 0.018, 0.035, [("hip_{side}", 0.3, 0.0, 0.02, 0.03)], [("patella_{side}", 0.4, 0.0, 0.0, 0.012)])
    add("vastus_lateralis", 0.024, 0.05, [("femur_{side}", 0.12, 0.02, 0.0, 0.0)], [("patella_{side}", 0.55, 0.015, 0.0, 0.008)])
    add("vastus_medialis", 0.02, 0.04, [("femur_{side}", 0.15, -0.01, 0.0, 0.008)], [("patella_{side}", 0.5, -0.012, 0.0, 0.008)])
    add("vastus_intermedius", 0.016, 0.03, [("femur_{side}", 0.25, 0.0, 0.0, 0.012)], [("patella_{side}", 0.45, 0.0, 0.0, 0.01)])
    add("articularis_genus", 0.004, 0.012, [("femur_{side}", 0.8, 0.0, 0.0, 0.01)], [("patella_{side}", 0.2, 0.0, 0.01, 0.006)])
    add("pectineus", 0.008, 0.015, [("hip_{side}", 0.2, 0.0, -0.01, 0.03)], [("femur_{side}", 0.22, -0.01, 0.0, 0.01)])
    add("adductor_longus", 0.012, 0.025, [("hip_{side}", 0.18, 0.0, -0.03, 0.02)], [("femur_{side}", 0.45, -0.02, 0.0, 0.0)])
    add("adductor_brevis", 0.01, 0.02, [("hip_{side}", 0.22, 0.0, -0.025, 0.015)], [("femur_{side}", 0.32, -0.018, 0.0, 0.0)])
    add("adductor_magnus", 0.02, 0.04, [("hip_{side}", 0.75, 0.0, -0.03, -0.01)], [("femur_{side}", 0.7, -0.015, 0.0, 0.0)])
    add("gracilis", 0.008, 0.014, [("hip_{side}", 0.16, 0.0, -0.04, 0.02)], [("tibia_{side}", 0.07, -0.015, 0.0, 0.01)])
    add("biceps_femoris", 0.016, 0.03, [("hip_{side}", 0.82, 0.0, -0.01, -0.025)], [("fibula_{side}", 0.06, 0.0, 0.0, -0.008)])
    add("semitendinosus", 0.01, 0.016, [("hip_{side}", 0.8, 0.0, -0.015, -0.02)], [("tibia_{side}", 0.08, -0.01, 0.0, -0.008)])
    add("semimembranosus", 0.014, 0.025, [("hip_{side}", 0.78, 0.0, -0.02, -0.018)], [("tibia_{side}", 0.06, -0.012, 0.0, -0.01)])

    # Leg and foot.
    add("tibialis_anterior", 0.012, 0.02, [("tibia_{side}", 0.15, 0.01, 0.0, 0.012)], [("cuneiform_medial_{side}", 0.5, 0.0, 0.0, 0.006)])
    add("extensor_hallucis_longus", 0.005, 0.008, [("fibula_{side}", 0.35, 0.0, 0.0, 0.006)], [("ph_foot_1_dist_{side}", 0.3, 0.0, 0.0, 0.004)])
    add("extensor_digitorum_longus", 0.006, 0.012, [("tibia_{side}", 0.2, 0.012, 0.0, 0.008)], [("ph_foot_3_dist_{side}", 0.25, 0.0, 0.0, 0.003)])
    add("fibularis_tertius", 0.004, 0.006, [("fibula_{side}", 0.75, 0.0, 0.0, 0.004)], [("metatarsal_5_{side}", 0.3, 0.0, 0.0, 0.0)])
    add("fibularis_longus", 0.008, 0.014, [("fibula_{side}", 0.12, 0.008, 0.0, -0.004)], [("cuneiform_medial_{side}", 0.4, 0.0, -0.004, 0.0)])
    add("fibularis_brevis", 0.006, 0.01, [("fibula_{side}", 0.55, 0.006, 0.0, -0.004)], [("metatarsal_5_{side}", 0.2, 0.0, 0.0, 0.0)])
    add("gastrocnemius", 0.02, 0.04, [("femur_{side}", 0.9, 0.015, 0.0, -0.015)], [("calcaneus_{side}", 0.8, 0.0, 0.0, -0.01)])
    add("soleus", 0.018, 0.035, [("tibia_{side}", 0.25, 0.0, 0.0, -0.01), ("fibula_{side}", 0.2, 0.0, 0.0, -0.008)], [("calcaneus_{side}", 0.75, 0.0, 0.0, -0.012)])
    add("plantaris", 0.003, 0.005, [("femur_{side}", 0.88, 0.012, 0.0, -0.012)], [("calcaneus_{side}", 0.7, 0.0, 0.005, -0.008)])
    add("popliteus", 0.006, 0.015, [("femur_{side}", 0.95, 0.015, 0.0, -0.008)], [("tibia_{side}", 0.12, 0.0, 0.0, -0.008)])
    add("tibialis_posterior", 0.01, 0.016, [("tibia_{side}", 0.35, 0.0, 0.0, -0.008)], [("navicular_{side}", 0.5, 0.0, 0.0, 0.0)])
    add("flexor_digitorum_longus", 0.007, 0.012, [("tibia_{side}", 0.4, -0.008, 0.0, -0.006)], [("ph_foot_3_dist_{side}", 0.3, 0.0, -0.003, 0.0)])
    add("flexor_hallucis_longus", 0.007, 0.01, [("fibula_{side}", 0.45, 0.0, 0.0, -0.008)], [("ph_foot_1_dist_{side}", 0.35, 0.0, -0.003, 0.0)])
    add("abductor_hallucis", 0.006, 0.012, [("calcaneus_{side}", 0.4, -0.01, 0.0, 0.0)], [("ph_foot_1_prox_{side}", 0.3, 0.0, 0.0, 0.004)])
    add("flexor_digitorum_brevis", 0.005, 0.015, [("calcaneus_{side}", 0.35, 0.0, -0.01, 0.005)], [("ph_foot_3_mid_{side}", 0.3, 0.0, -0.003, 0.0)])
    add("abductor_digiti_minimi_foot", 0.005, 0.01, [("calcaneus_{side}", 0.45, 0.015, 0.0, -0.008)], [("ph_foot_5_prox_{side}", 0.3, 0.0, 0.0, 0.0)])
    add("quadratus_plantae", 0.004, 0.012, [("calcaneus_{side}", 0.5, 0.0, -0.008, 0.0)], [("ph_foot_3_dist_{side}", 0.15, 0.0, -0.004, 0.0)])
    for d in range(1, 5):
        add(f"lumbrical_foot_{d}", 0.002, 0.003,
            [(f"metatarsal_{d + 1}_{{side}}", 0.4, 0.0, -0.004, 0.0)], [(f"ph_foot_{d + 1}_prox_{{side}}", 0.25, 0.0, 0.0, 0.003)])
    add("flexor_hallucis_brevis", 0.004, 0.008, [("cuneiform_medial_{side}", 0.6, 0.0, -0.004, 0.0)], [("ph_foot_1_prox_{side}", 0.35, 0.0, -0.002, 0.0)])
    add("adductor_hallucis", 0.004, 0.012, [("metatarsal_3_{side}", 0.5, 0.0, -0.004, 0.0)], [("ph_foot_1_prox_{side}", 0.4, 0.0, 0.0, 0.0)])
    add("flexor_digiti_minimi_foot", 0.003, 0.006, [("metatarsal_5_{side}", 0.3, 0.0, -0.003, 0.0)], [("ph_foot_5_prox_{side}", 0.35, 0.0, 0.0, 0.0)])
    for d in range(1, 4):
        add(f"plantar_interosseous_{d}", 0.002, 0.004,
            [(f"metatarsal_{d + 1}_{{side}}", 0.5, 0.0, -0.003, 0.0)], [(f"ph_foot_{d + 1}_prox_{{side}}", 0.25, 0.0, 0.0, 0.0)])
    for d in range(1, 5):
        add(f"dorsal_interosseous_foot_{d}", 0.003, 0.005,
            [(f"metatarsal_{d}_{{side}}", 0.5, 0.003, 0.0, 0.004)], [(f"ph_foot_{min(d, 4)}_prox_{{side}}", 0.25, 0.0, 0.0, 0.003)])
    add("extensor_digitorum_brevis", 0.004, 0.012, [("calcaneus_{side}", 0.3, 0.01, 0.01, 0.008)], [("ph_foot_3_prox_{side}", 0.2, 0.0, 0.004, 0.004)])
    add("extensor_hallucis_brevis", 0.003, 0.006, [("calcaneus_{side}", 0.25, 0.0, 0.012, 0.01)], [("ph_foot_1_prox_{side}", 0.3, 0.0, 0.003, 0.004)])

    # Pelvic floor. Small, on the bones, not a separate system.
    add("pubococcygeus", 0.004, 0.012, [("hip_{side}", 0.15, 0.0, -0.02, 0.0)], [("coccyx", 0.5, 0.01, -0.01, 0.0)])
    add("iliococcygeus", 0.004, 0.012, [("hip_{side}", 0.4, 0.0, -0.015, -0.01)], [("coccyx", 0.4, 0.012, -0.008, 0.0)])
    add("puborectalis", 0.004, 0.01, [("hip_{side}", 0.12, 0.0, -0.025, 0.01)], [("coccyx", 0.6, 0.008, -0.015, -0.005)])
    add("coccygeus", 0.004, 0.01, [("hip_{side}", 0.7, 0.0, -0.01, -0.02)], [("coccyx", 0.5, 0.006, 0.0, -0.008)])
    add("sphincter_ani", 0.004, 0.008, [("coccyx", 0.7, -0.012, -0.02, 0.0)], [("coccyx", 0.7, 0.012, -0.02, 0.0)], paired=False)

    return rows


def muscles():
    out = []
    for name, radius, width, paired, origins, insertions in _catalog():
        if not paired:
            out.append(Muscle(name, radius, width, origins, insertions))
            continue
        for side in ("l", "r"):
            out.append(Muscle(f"{name}_{side}", radius, width, origins, insertions))
    return out


class Musculature:
    def __init__(self, skeleton):
        self.skeleton = skeleton
        self.muscles = muscles()
        self.skin = None

    def placed(self):
        world = self.skeleton.world()
        rows = []
        for muscle in self.muscles:
            side = "l" if muscle.name.endswith("_l") else "r" if muscle.name.endswith("_r") else None
            rows.append(muscle.place(world, side))
        return rows

    def bones(self):
        return self.skeleton.world()


def install(c):
    import skeleton as skeleton_mod
    sk = getattr(c, "skeleton", None)
    if sk is None:
        skeleton_mod.install(c)
        sk = c.skeleton
    c.skin = None
    c.muscles = Musculature(sk)
    c._asleep = True
    return {"asleep": True, "skin": False, "muscles": len(c.muscles.muscles), "bones": len(sk.bones)}


def export(path):
    import skeleton as skeleton_mod
    sk = skeleton_mod.Skeleton(REF_H)
    body = Musculature(sk)
    placed = []
    longest = ("", 0.0)
    for row in body.placed():
        span = _dist(row["belly"][0], row["belly"][1])
        if span > longest[1]:
            longest = (row["name"], span)
        placed.append({
            "name": row["name"],
            "radius": row["radius"],
            "width": row["width"],
            "ligaments": [[list(a), list(b)] for a, b in row["ligaments"]],
            "belly": [list(row["belly"][0]), list(row["belly"][1])],
        })
    bones = [{"name": name, "pts": [list(p) for p in seq]} for name, seq in body.bones().items()]
    doc = {
        "asleep": True,
        "skin": False,
        "height_m": REF_H,
        "bones_n": len(bones),
        "muscles_n": len(placed),
        "longest": {"name": longest[0], "span_m": round(longest[1], 3)},
        "bones": bones,
        "muscles": placed,
    }
    path = Path(path)
    path.write_text(json.dumps(doc))
    return {"bones": len(bones), "muscles": len(placed), "longest": doc["longest"], "bytes": path.stat().st_size}


if __name__ == "__main__":
    print(export("/workspace/public/baldur-body.json"))
