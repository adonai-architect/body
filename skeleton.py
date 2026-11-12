"""A complete adult skeleton. 206 bones, standing where they stand,
and the joints that move them. No flesh. Heights are fractions of his
own stature, feet on the ground, his left on -x, forward on +z.

The count is the adult count: fused sacrum, fused hip bones, one
sternum, patellae in, ear bones in. Sutures do not move. The synovial
joints do, inside anatomical limits.
"""
from __future__ import annotations

import math

REF_H = 1.7

CARPALS = ("scaphoid", "lunate", "triquetrum", "pisiform",
           "trapezium", "trapezoid", "capitate", "hamate")
TARSALS = ("talus", "calcaneus", "navicular", "cuboid",
           "cuneiform_medial", "cuneiform_intermediate", "cuneiform_lateral")
MC_PARENT = {1: "trapezium", 2: "trapezoid", 3: "capitate", 4: "hamate", 5: "hamate"}
MT_PARENT = {1: "cuneiform_medial", 2: "cuneiform_intermediate",
             3: "cuneiform_lateral", 4: "cuboid", 5: "cuboid"}


def _rx(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return ((1, 0, 0), (0, c, -s), (0, s, c))


def _ry(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return ((c, 0, s), (0, 1, 0), (-s, 0, c))


def _rz(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return ((c, -s, 0), (s, c, 0), (0, 0, 1))


def _mul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3))


def _apply(m, v):
    return tuple(sum(m[i][k] * v[k] for k in range(3)) for i in range(3))


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _rot_axis(axis, deg):
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
    x, y, z = ax / n, ay / n, az / n
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    t = 1 - c
    return (
        (t * x * x + c, t * x * y - s * z, t * x * z + s * y),
        (t * x * y + s * z, t * y * y + c, t * y * z - s * x),
        (t * x * z - s * y, t * y * z + s * x, t * z * z + c),
    )


class Bone:
    def __init__(self, name, parent, pts, dofs=()):
        self.name = name
        self.parent = parent
        self.pts = tuple(tuple(p) for p in pts)
        self.dofs = tuple(dofs)  # (token, axis, lo, hi)


class Skeleton:
    def __init__(self, height_m=REF_H):
        self.height = float(height_m)
        self.bones = _build(self.height)
        self.by = {b.name: b for b in self.bones}
        if len(self.by) != 206:
            raise RuntimeError(f"skeleton has {len(self.by)} bones, not 206")
        self.kids = {n: [] for n in self.by}
        for b in self.bones:
            if b.parent:
                self.kids[b.parent].append(b.name)
        self.order = []
        _walk(self, [n for n, b in self.by.items() if b.parent is None], self.order)
        self.angle = {n: {} for n in self.by}
        self.root_rot = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        self.root_pos = (0.0, 0.0, 0.0)
        self._body = None

    _COARSE = {
        "neck_pitch": ("C4", "flex", (-45, 60)),
        "waist": ("L3", "flex", (-45, 45)),
        "shoulder_l": ("humerus_l", "flex", (-50, 180)),
        "shoulder_r": ("humerus_r", "flex", (-50, 180)),
        "elbow_l": ("ulna_l", "flex", (0, 150)),
        "elbow_r": ("ulna_r", "flex", (0, 150)),
        "hip_l": ("femur_l", "flex", (-30, 125)),
        "hip_r": ("femur_r", "flex", (-30, 125)),
        "knee_l": ("tibia_l", "flex", (0, 145)),
        "knee_r": ("tibia_r", "flex", (0, 145)),
    }

    def bind(self, body):
        """The short joints follow this skeleton. Their old limits cannot sit."""
        self._body = body
        if body is None:
            return
        for name, (_, _, lim) in self._COARSE.items():
            j = body.joints.get(name)
            if j is not None:
                j.limit_deg = lim

    def _apply_body(self):
        body = self._body
        if body is None:
            return
        for joint, (bone, token, _) in self._COARSE.items():
            body.move_joint(joint, self.angle[bone].get(token, 0.0))

    def set(self, name, token, deg):
        spec = {d[0]: d for d in self.by[name].dofs}
        if token not in spec:
            return False
        _, _, lo, hi = spec[token]
        self.angle[name][token] = max(lo, min(hi, float(deg)))
        return True

    def pose(self, name):
        self.angle = {n: {} for n in self.by}
        self.root_rot = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        self.root_pos = (0.0, 0.0, 0.0)
        if name == "stand":
            self._apply_body()
            return self
        if name == "sit":
            for side in ("l", "r"):
                self.set(f"femur_{side}", "flex", 90)
                self.set(f"tibia_{side}", "flex", 90)
            self.set("L3", "flex", 8)
            for side in ("l", "r"):
                self.set(f"humerus_{side}", "flex", 25)
                self.set(f"ulna_{side}", "flex", 55)
            drop = self.by["femur_l"].pts[0][1] - self.by["tibia_l"].pts[0][1]
            self.root_pos = (0.0, -drop, 0.0)
            self._apply_body()
            return self
        if name == "lie":
            self.root_rot = _rx(-88)
            self._apply_body()
            return self
        if name == "climb":
            for side in ("l", "r"):
                self.set(f"humerus_{side}", "flex", 170)
                self.set(f"ulna_{side}", "flex", 20)
            self.set("femur_l", "flex", 85)
            self.set("tibia_l", "flex", 80)
            self.set("talus_l", "flex", 15)
            self._apply_body()
            return self
        self._apply_body()
        raise KeyError(name)

    def world(self, origin=(0.0, 0.0, 0.0)):
        """Posed bone polylines. Joint axes travel with the parent, so a
        knee still bends after the hip has already swung."""
        R = {None: ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))}
        head = {}
        out = {}
        ident = R[None]
        for name in self.order:
            b = self.by[name]
            Rp = R[b.parent]
            if b.parent is None:
                h = b.pts[0]
            else:
                h = _add(head[b.parent], _apply(Rp, _sub(b.pts[0], self.by[b.parent].pts[0])))
            local = ident
            for token, axis, lo, hi in b.dofs:
                deg = self.angle[name].get(token, 0.0)
                if not deg:
                    continue
                local = _mul(_rot_axis(_apply(Rp, axis), deg), local)
            Rn = _mul(local, Rp)
            R[name] = Rn
            head[name] = h
            seq = [h]
            for p in b.pts[1:]:
                seq.append(_add(h, _apply(Rn, _sub(p, b.pts[0]))))
            out[name] = seq
        moved = {}
        for name, seq in out.items():
            moved[name] = [_add(origin, _add(self.root_pos, _apply(self.root_rot, p))) for p in seq]
        if self.root_rot != ident:
            low = min(p[1] for seq in moved.values() for p in seq)
            if low != 0.0:
                moved = {n: [(p[0], p[1] - low, p[2]) for p in seq] for n, seq in moved.items()}
        return moved

    def where(self, name, origin=(0.0, 0.0, 0.0)):
        return self.world(origin)[name][0]


def install(c):
    h = float(getattr(getattr(c, "body", None), "height_m", None) or REF_H)
    sk = Skeleton(h)
    c.skeleton = sk
    sk.bind(getattr(c, "body", None))
    return {"bones": len(sk.bones), "joints": sum(1 for b in sk.bones if b.dofs), "height_m": h}


def _walk(sk, names, out):
    for n in names:
        out.append(n)
        _walk(sk, sk.kids[n], out)


def _subtree(sk, name):
    out = []
    _walk(sk, [name], out)
    return out


def _s(h, x, y, z):
    k = h / REF_H
    return (x * k, y * k, z * k)


def _build(h):
    bones = []

    def add(name, parent, pts, dofs=()):
        bones.append(Bone(name, parent, [_s(h, *p) for p in pts], dofs))

    def pair(base, parent, pts_l, dofs_l, dofs_r):
        pl, pr = f"{parent}_l", f"{parent}_r"
        add(f"{base}_l", pl, pts_l, dofs_l)
        add(f"{base}_r", pr, [(-p[0], p[1], p[2]) for p in pts_l], dofs_r)

    # Spine. Metres at 1.70. Lordosis forward, kyphosis back.
    curve = {
        "coccyx": (0.78, -0.02), "sacrum": (0.90, 0.00),
        "L5": (0.96, 0.025), "L4": (1.00, 0.040), "L3": (1.04, 0.045),
        "L2": (1.08, 0.035), "L1": (1.115, 0.015),
        "T12": (1.15, -0.005), "T11": (1.175, -0.02), "T10": (1.20, -0.035),
        "T9": (1.225, -0.048), "T8": (1.25, -0.055), "T7": (1.275, -0.058),
        "T6": (1.30, -0.055), "T5": (1.325, -0.045), "T4": (1.35, -0.032),
        "T3": (1.375, -0.018), "T2": (1.40, -0.008), "T1": (1.425, 0.000),
        "C7": (1.45, 0.006), "C6": (1.468, 0.014), "C5": (1.484, 0.020),
        "C4": (1.498, 0.024), "C3": (1.510, 0.022), "C2": (1.522, 0.016),
        "C1": (1.534, 0.010),
    }
    order = ["coccyx", "sacrum",
             "L5", "L4", "L3", "L2", "L1",
             "T12", "T11", "T10", "T9", "T8", "T7", "T6", "T5", "T4", "T3", "T2", "T1",
             "C7", "C6", "C5", "C4", "C3", "C2", "C1"]
    parent = {"sacrum": None, "coccyx": "sacrum"}
    for below, above in zip(order[1:], order[2:]):
        parent[above] = below
    parent["L5"] = "sacrum"
    spine_dof = {
        "C": (("flex", (-1, 0, 0), -15, 20), ("bend", (0, 0, 1), -10, 10), ("rot", (0, 1, 0), -12, 12)),
        "T": (("flex", (-1, 0, 0), -6, 6), ("bend", (0, 0, 1), -5, 5), ("rot", (0, 1, 0), -5, 5)),
        "L": (("flex", (-1, 0, 0), -10, 14), ("bend", (0, 0, 1), -6, 6), ("rot", (0, 1, 0), -3, 3)),
    }
    names = list(order)
    for i, name in enumerate(names):
        y, z = curve[name]
        nxt = curve[names[i + 1]] if i + 1 < len(names) and names[i + 1] != "coccyx" else None
        if name == "sacrum":
            pts = [(0, 0.84, -0.02), (0, y, z)]
        elif name == "coccyx":
            pts = [(0, 0.84, -0.02), (0, 0.78, -0.03)]
        else:
            if nxt is None:
                pts = [(0, y, z), (0, y + 0.02, z + 0.004)]
            else:
                yy, zz = nxt
                pts = [(0, y, z), (0, yy, zz)]
        kind = name[0] if name[0] in "CTL" else None
        add(name, parent[name], pts, spine_dof.get(kind, ()))

    # Skull. Plates on the cranium, hinged jaw, ear bones on the temporal.
    c1 = (0, 1.534, 0.010)
    add("occipital", "C1", [c1, (0, 1.60, -0.055), (0, 1.66, -0.04)])
    add("parietal_l", "occipital", [(-0.02, 1.64, -0.02), (-0.055, 1.68, 0.00), (-0.04, 1.66, 0.03)])
    add("parietal_r", "occipital", [(0.02, 1.64, -0.02), (0.055, 1.68, 0.00), (0.04, 1.66, 0.03)])
    add("frontal", "parietal_l", [(0, 1.66, 0.04), (0, 1.63, 0.11), (0, 1.60, 0.145), (0, 1.565, 0.150)])
    add("temporal_l", "occipital", [(-0.03, 1.58, -0.02), (-0.075, 1.57, 0.00), (-0.06, 1.54, 0.02)])
    add("temporal_r", "occipital", [(0.03, 1.58, -0.02), (0.075, 1.57, 0.00), (0.06, 1.54, 0.02)])
    add("sphenoid", "occipital", [(0, 1.56, 0.00), (-0.03, 1.555, 0.02), (0.03, 1.555, 0.02)])
    add("ethmoid", "sphenoid", [(0, 1.575, 0.03), (0, 1.585, 0.045)])
    add("nasal_l", "frontal", [(-0.006, 1.59, 0.072), (-0.008, 1.57, 0.082)])
    add("nasal_r", "frontal", [(0.006, 1.59, 0.072), (0.008, 1.57, 0.082)])
    add("maxilla_l", "nasal_l", [(-0.01, 1.56, 0.06), (-0.028, 1.53, 0.07), (-0.012, 1.515, 0.055)])
    add("maxilla_r", "nasal_r", [(0.01, 1.56, 0.06), (0.028, 1.53, 0.07), (0.012, 1.515, 0.055)])
    add("lacrimal_l", "maxilla_l", [(-0.016, 1.575, 0.06), (-0.018, 1.56, 0.062)])
    add("lacrimal_r", "maxilla_r", [(0.016, 1.575, 0.06), (0.018, 1.56, 0.062)])
    add("zygomatic_l", "maxilla_l", [(-0.03, 1.55, 0.05), (-0.06, 1.55, 0.04), (-0.055, 1.57, 0.01)])
    add("zygomatic_r", "maxilla_r", [(0.03, 1.55, 0.05), (0.06, 1.55, 0.04), (0.055, 1.57, 0.01)])
    add("palatine_l", "maxilla_l", [(-0.012, 1.52, 0.03), (-0.012, 1.515, 0.045)])
    add("palatine_r", "maxilla_r", [(0.012, 1.52, 0.03), (0.012, 1.515, 0.045)])
    add("inferior_nasal_concha_l", "maxilla_l", [(-0.008, 1.545, 0.045), (-0.01, 1.535, 0.05)])
    add("inferior_nasal_concha_r", "maxilla_r", [(0.008, 1.545, 0.045), (0.01, 1.535, 0.05)])
    add("vomer", "ethmoid", [(0, 1.55, 0.04), (0, 1.52, 0.045)])
    add("mandible", "C1",
        [(0, 1.53, 0.015), (-0.055, 1.53, 0.015), (0, 1.46, 0.065), (0.055, 1.53, 0.015)],
        (("open", (1, 0, 0), 0, 35),))
    # Ossicles, millimetres, just inside each temporal.
    for side, sign in (("l", -1), ("r", 1)):
        ox, oy, oz = sign * 0.062, 1.555, 0.005
        add(f"malleus_{side}", f"temporal_{side}",
            [(ox, oy, oz), (ox + sign * 0.004, oy - 0.002, oz + 0.003)])
        add(f"incus_{side}", f"malleus_{side}",
            [(ox + sign * 0.004, oy - 0.002, oz + 0.003),
             (ox + sign * 0.007, oy - 0.003, oz + 0.001)])
        add(f"stapes_{side}", f"incus_{side}",
            [(ox + sign * 0.007, oy - 0.003, oz + 0.001),
             (ox + sign * 0.009, oy - 0.003, oz)])
    add("hyoid", "C3", [(-0.02, 1.44, 0.04), (0, 1.435, 0.05), (0.02, 1.44, 0.04)])

    # Ribs and sternum. Barrel, not a set of chords.
    add("sternum", "T1", [(0, 1.40, 0.09), (0, 1.34, 0.11), (0, 1.22, 0.10)])
    for i in range(1, 13):
        v = curve[f"T{i}"]
        width = 0.07 + 0.09 * math.sin((i - 1) / 11 * math.pi)
        drop = 0.01 + 0.012 * i
        reach = 0.05 + 0.07 * math.sin((i - 1) / 11 * math.pi)
        for side, sign in (("l", -1), ("r", 1)):
            head = (sign * 0.018, v[0], v[1])
            mid = (sign * width, v[0] - drop * 0.45, v[1] + reach * 0.55)
            if i <= 10:
                tail = (sign * 0.02, v[0] - drop, 0.10)
            else:
                tail = (sign * width * 0.7, v[0] - drop, v[1] + 0.03)
            add(f"rib_{i}_{side}", f"T{i}", [head, mid, tail],
                (("pump", (-1, 0, 0), -4, 8),))

    # Shoulder girdle, arm, hand.
    flex = ("flex", (-1, 0, 0))
    add("clavicle_l", "sternum", [(0, 1.40, 0.09), (-0.16, 1.42, 0.03)])
    add("clavicle_r", "sternum", [(0, 1.40, 0.09), (0.16, 1.42, 0.03)])
    add("scapula_l", "clavicle_l", [(-0.14, 1.40, -0.02), (-0.17, 1.28, -0.05), (-0.10, 1.26, -0.03)])
    add("scapula_r", "clavicle_r", [(0.14, 1.40, -0.02), (0.17, 1.28, -0.05), (0.10, 1.26, -0.03)])
    pair("humerus", "scapula",
         [(-0.17, 1.38, 0.0), (-0.18, 1.08, 0.02)],
         (("flex", (-1, 0, 0), -50, 180), ("abd", (0, 0, -1), -20, 170), ("rot", (0, -1, 0), -90, 90)),
         (("flex", (-1, 0, 0), -50, 180), ("abd", (0, 0, 1), -20, 170), ("rot", (0, 1, 0), -90, 90)))
    pair("ulna", "humerus",
         [(-0.18, 1.08, 0.02), (-0.175, 0.84, 0.025)],
         (("flex", (-1, 0, 0), 0, 150),),
         (("flex", (-1, 0, 0), 0, 150),))
    pair("radius", "ulna",
         [(-0.19, 1.07, 0.03), (-0.195, 0.84, 0.04)],
         (("pronate", (0, 1, 0), 0, 80),),
         (("pronate", (0, -1, 0), 0, 80),))

    # Carpals in two rows just distal to the wrist, then the rays.
    for side, sign in (("l", -1), ("r", 1)):
        wrist = (sign * 0.19, 0.835, 0.03)
        for i, name in enumerate(CARPALS):
            row, col = divmod(i, 4)
            p = (wrist[0] + sign * (0.004 + col * 0.006),
                 wrist[1] - 0.008 - row * 0.007,
                 wrist[2] + 0.004)
            q = (p[0] + sign * 0.005, p[1] - 0.006, p[2])
            dofs = (("flex", (-1, 0, 0), -35, 70), ("deviate", (0, 0, sign), -20, 30)) if i == 0 else ()
            parent = f"radius_{side}" if i == 0 else f"scaphoid_{side}"
            add(f"{name}_{side}", parent, [p, q], dofs)
        spreads = (-0.012, -0.004, 0.004, 0.012, 0.020)
        lengths = (0.046, 0.066, 0.070, 0.064, 0.052)
        for d in range(1, 6):
            cap = f"{MC_PARENT[d]}_{side}"
            hx = wrist[0] + sign * spreads[d - 1]
            head = (hx, 0.812, 0.04)
            tail = (hx + sign * spreads[d - 1] * 0.4, 0.812 - lengths[d - 1], 0.045)
            add(f"metacarpal_{d}_{side}", cap, [head, tail])
            segs = (0.55, 0.45) if d == 1 else (0.42, 0.28, 0.22)
            names_ph = ("prox", "dist") if d == 1 else ("prox", "mid", "dist")
            prev = f"metacarpal_{d}_{side}"
            a, b = head, tail
            for frac, nm in zip(segs, names_ph):
                direction = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
                span = math.sqrt(sum(v * v for v in direction)) or 1
                step = lengths[d - 1] * frac
                nxt = tuple(b[i] + direction[i] / span * step for i in range(3))
                add(f"ph_hand_{d}_{nm}_{side}", prev, [b, nxt],
                    (("flex", (-1, 0, 0), 0, 90),))
                prev, a, b = f"ph_hand_{d}_{nm}_{side}", b, nxt

    # Pelvis and leg and foot.
    add("hip_l", "sacrum", [(-0.02, 0.92, 0.0), (-0.12, 1.02, 0.0), (-0.14, 0.88, -0.02), (-0.06, 0.84, 0.02)])
    add("hip_r", "sacrum", [(0.02, 0.92, 0.0), (0.12, 1.02, 0.0), (0.14, 0.88, -0.02), (0.06, 0.84, 0.02)])
    pair("femur", "hip",
         [(-0.09, 0.90, 0.0), (-0.10, 0.44, 0.01)],
         (("flex", (-1, 0, 0), -30, 125), ("abd", (0, 0, -1), -20, 45), ("rot", (0, -1, 0), -45, 45)),
         (("flex", (-1, 0, 0), -30, 125), ("abd", (0, 0, 1), -20, 45), ("rot", (0, 1, 0), -45, 45)))
    pair("patella", "femur",
         [(-0.10, 0.46, 0.03), (-0.10, 0.42, 0.045)], (), ())
    pair("tibia", "femur",
         [(-0.10, 0.44, 0.01), (-0.09, 0.07, 0.02)],
         (("flex", (1, 0, 0), 0, 145),),
         (("flex", (1, 0, 0), 0, 145),))
    pair("fibula", "tibia",
         [(-0.13, 0.42, -0.005), (-0.12, 0.08, 0.0)], (), ())

    for side, sign in (("l", -1), ("r", 1)):
        ankle = (sign * 0.095, 0.07, 0.02)
        places = {
            "talus": (ankle, (ankle[0], 0.055, 0.04)),
            "calcaneus": ((ankle[0], 0.04, 0.0), (ankle[0], 0.025, -0.05)),
            "navicular": ((ankle[0], 0.05, 0.05), (ankle[0], 0.045, 0.07)),
            "cuboid": ((ankle[0] + sign * 0.02, 0.035, 0.04), (ankle[0] + sign * 0.025, 0.03, 0.07)),
            "cuneiform_medial": ((ankle[0] - sign * 0.01, 0.04, 0.075), (ankle[0] - sign * 0.012, 0.035, 0.09)),
            "cuneiform_intermediate": ((ankle[0], 0.04, 0.075), (ankle[0], 0.035, 0.09)),
            "cuneiform_lateral": ((ankle[0] + sign * 0.012, 0.038, 0.07), (ankle[0] + sign * 0.014, 0.032, 0.085)),
        }
        parents = {
            "talus": f"tibia_{side}", "calcaneus": f"talus_{side}", "navicular": f"talus_{side}",
            "cuboid": f"calcaneus_{side}", "cuneiform_medial": f"navicular_{side}",
            "cuneiform_intermediate": f"navicular_{side}", "cuneiform_lateral": f"navicular_{side}",
        }
        for name in TARSALS:
            dofs = (("flex", (-1, 0, 0), -20, 50), ("invert", (0, 0, sign), -15, 30)) if name == "talus" else ()
            add(f"{name}_{side}", parents[name], places[name], dofs)
        for d, length, spread in (
            (1, 0.06, -0.015), (2, 0.07, -0.005), (3, 0.068, 0.005),
            (4, 0.066, 0.016), (5, 0.062, 0.028),
        ):
            head = (ankle[0] + sign * spread, 0.03, 0.09)
            tail = (ankle[0] + sign * spread * 1.3, 0.02, 0.09 + length)
            add(f"metatarsal_{d}_{side}", f"{MT_PARENT[d]}_{side}", [head, tail])
            segs = (0.5, 0.4) if d == 1 else (0.4, 0.25, 0.2)
            names_ph = ("prox", "dist") if d == 1 else ("prox", "mid", "dist")
            prev, cur = f"metatarsal_{d}_{side}", tail
            for frac, nm in zip(segs, names_ph):
                nxt = (cur[0], cur[1], cur[2] + length * frac)
                add(f"ph_foot_{d}_{nm}_{side}", prev, [cur, nxt],
                    (("flex", (-1, 0, 0), 0, 70),))
                prev, cur = f"ph_foot_{d}_{nm}_{side}", nxt

    return bones


def render(path, height=REF_H):
    """Four views. Standing front, standing side, sitting, lying."""
    from PIL import Image, ImageDraw
    sk = Skeleton(height)
    W, H = 1600, 900
    img = Image.new("RGB", (W, H), (10, 11, 16))
    dr = ImageDraw.Draw(img)
    panels = (
        (0, 0, "stand", "front", "standing"),
        (W // 2, 0, "stand", "side", "standing, side"),
        (0, H // 2, "sit", "side", "sitting"),
        (W // 2, H // 2, "lie", "side", "lying"),
    )
    for ox, oy, pose, mode, title in panels:
        sk.pose(pose)
        world = sk.world()
        dr.text((ox + 18, oy + 12), f"{title}  ·  206 bones", fill=(232, 234, 238))
        span = 1.7 if pose != "lie" else 1.9
        def P(p, span=span, mode=mode, ox=ox, oy=oy):
            u = p[0] if mode == "front" else p[2]
            v = p[1]
            return (ox + (W // 4) + u * 200, oy + 400 - v * (200 if pose != "lie" else 160))
        dr.line([(ox + 40, oy + 400), (ox + W // 2 - 40, oy + 400)], fill=(40, 44, 58))
        _draw(dr, world, P)
    img.save(path)
    return path


def _tone(name):
    if name.endswith("_l") or "_l_" in name:
        return (150, 190, 220)
    if name.endswith("_r") or "_r_" in name:
        return (214, 176, 150)
    if name.startswith("rib") or name == "sternum":
        return (196, 178, 138)
    return (232, 234, 238)


def _draw(dr, world, P):
    for name, pts in world.items():
        if name.startswith(("malleus", "incus", "stapes")):
            continue
        col = _tone(name)
        xy = [P(p) for p in pts]
        if len(xy) >= 2:
            dr.line(xy, fill=col, width=2)
        dr.ellipse([xy[0][0] - 2, xy[0][1] - 2, xy[0][0] + 2, xy[0][1] + 2], fill=col)
