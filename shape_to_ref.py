"""Push the clay toward the shirtless reference: lean muscle, black briefs, short hair."""
import gzip
from pathlib import Path
import numpy as np

BASE = Path("/tmp/mh/base.obj")
TGT = Path("/tmp/mh/targets")
OUT = Path("/workspace/public/prefabs")


def load_base():
    verts = []
    body = []
    group = None
    for line in open(BASE):
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("g "):
            group = line.split()[1]
        elif line.startswith("f ") and group == "body":
            ids = [int(p.split("/")[0]) - 1 for p in line.split()[1:]]
            for i in range(1, len(ids) - 1):
                body.append((ids[0], ids[i], ids[i + 1]))
    return np.array(verts, float), body


def apply_target(verts, name, factor):
    for line in gzip.open(TGT / f"{name}.target.gz", "rt"):
        if not line.strip() or line.startswith("#"):
            continue
        i, dx, dy, dz = line.split()
        verts[int(i)] += factor * np.array([float(dx), float(dy), float(dz)])


def write_obj(path, verts, faces, note):
    lines = [f"# {note}"]
    for p in verts:
        lines.append(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}")
    for a, b, c in faces:
        lines.append(f"f {a + 1} {b + 1} {c + 1}")
    Path(path).write_text("\n".join(lines) + "\n")
    print(path, len(verts), "verts", len(faces), "faces")


class Mesh:
    def __init__(self):
        self.v = []
        self.f = []

    def add(self, p):
        self.v.append(np.asarray(p, float))
        return len(self.v) - 1

    def tri(self, a, b, c, outward):
        n = np.cross(np.asarray(b) - a, np.asarray(c) - a)
        if np.dot(n, outward) < 0:
            b, c = c, b
        i = self.add(a)
        j = self.add(b)
        k = self.add(c)
        self.f.append((i, j, k))

    def quad(self, a, b, c, d, outward):
        self.tri(a, b, c, outward)
        self.tri(a, c, d, outward)


def stitch(mesh, a, b, outward_fn):
    n = len(a)
    for k in range(n):
        k2 = (k + 1) % n
        mesh.quad(a[k], a[k2], b[k2], b[k], outward_fn((a[k] + b[k]) / 2))


def shell_from_rings(rings, thick):
    """rings are (center, rx, rz). Built just outside the body, with a hem."""
    mesh = Mesh()
    outers, inners = [], []
    for c, rx, rz in rings:
        outers.append(ering(c, rx, rz))
        inners.append(ering(c, max(rx - thick, 0.02), max(rz - thick, 0.02)))
    for i in range(len(outers) - 1):
        stitch(mesh, outers[i], outers[i + 1], lambda m: np.array([m[0], 0.0, m[2]]))
        stitch(mesh, inners[i + 1], inners[i], lambda m: -np.array([m[0], 0.0, m[2]]))
    stitch(mesh, inners[0], outers[0], lambda m: np.array([0.0, -1.0, 0.0]))
    stitch(mesh, outers[-1], inners[-1], lambda m: np.array([0.0, 1.0, 0.0]))
    return mesh


def solid_briefs():
    hip = shell_from_rings([
        (np.array([0.0, 0.80, 0.025]), 0.215, 0.140),
        (np.array([0.0, 0.90, 0.03]), 0.200, 0.130),
        (np.array([0.0, 0.98, 0.04]), 0.180, 0.118),
        (np.array([0.0, 1.045, 0.05]), 0.150, 0.105),
    ], 0.008)
    def leg(sign):
        return shell_from_rings([
            (np.array([sign * 0.11, 0.68, 0.035]), 0.112, 0.118),
            (np.array([sign * 0.11, 0.78, 0.032]), 0.120, 0.125),
            (np.array([sign * 0.11, 0.90, 0.028]), 0.125, 0.130),
        ], 0.007)
    parts = [hip, leg(-1.0), leg(1.0)]
    briefs = Mesh()
    for part in parts:
        base = len(briefs.v)
        briefs.v.extend(part.v)
        briefs.f.extend((a + base, b + base, c + base) for a, b, c in part.f)
    return briefs
    c = np.array([np.median(pts[:, 0]), float(np.median(pts[:, 1])), np.median(pts[:, 2])])
    rx = float(np.percentile(np.abs(pts[:, 0] - c[0]), 88))
    rz = float(np.percentile(np.abs(pts[:, 2] - c[2]), 88))
    return c, max(rx, 0.04), max(rz, 0.04)


def ering(c, rx, rz, n=32):
    ring = []
    for i in range(n):
        a = 2 * np.pi * i / n
        ring.append(np.array([c[0] + rx * np.cos(a), c[1], c[2] + rz * np.sin(a)]))
    return ring


def garment(sections, ease, thick, hem_down=True, top_up=True):
    mesh = Mesh()
    outers, inners = [], []
    for pts in sections:
        if len(pts) < 8:
            continue
        c, rx, rz = ellipse_from(pts)
        outers.append(ering(c, rx + ease, rz + ease))
        inners.append(ering(c, rx + ease - thick, rz + ease - thick))
    for i in range(len(outers) - 1):
        stitch(mesh, outers[i], outers[i + 1], lambda m: np.array([m[0], 0, m[2] - 0.02]))
        stitch(mesh, inners[i + 1], inners[i], lambda m: -np.array([m[0], 0, m[2] - 0.02]))
    if hem_down and outers:
        stitch(mesh, inners[0], outers[0], lambda m: np.array([0, -1, 0]))
    if top_up and outers:
        stitch(mesh, outers[-1], inners[-1], lambda m: np.array([0, 1, 0]))
    return mesh


def band_sections(body, y0, y1, step, x_lo, x_hi):
    sections = []
    y = y0
    while y <= y1:
        m = (np.abs(body[:, 1] - y) < step * 0.65) & (body[:, 0] > x_lo) & (body[:, 0] < x_hi)
        if m.sum() >= 8:
            sections.append(body[m])
        y += step
    return sections


def main():
    verts, body_faces = load_base()
    apply_target(verts, "maxmuscle", 1.0)
    apply_target(verts, "universal-male-young-averagemuscle-minweight", 0.35)
    apply_target(verts, "torso-muscle-pectoral-incr", 0.4)
    apply_target(verts, "l-upperarm-shoulder-muscle-incr", 0.55)
    apply_target(verts, "r-upperarm-shoulder-muscle-incr", 0.55)
    apply_target(verts, "l-upperarm-muscle-incr", 0.45)
    apply_target(verts, "r-upperarm-muscle-incr", 0.45)
    body = verts[:13380] * 0.1
    body[:, 1] -= body[:, 1].min()

    # A small waist pinch on the torso only, so the taper reads. Arms stay outside.
    waist = (body[:, 1] > 0.95) & (body[:, 1] < 1.15) & (np.abs(body[:, 0]) < 0.2)
    body[waist, 0] *= 0.9

    used = sorted({i for f in body_faces for i in f})
    assert max(used) < 13380
    write_obj(OUT / "baldur.obj", body, body_faces, "lean muscle, feet on the floor")

    briefs = solid_briefs()
    write_obj(OUT / "briefs.obj", np.array(briefs.v), briefs.f, "black briefs, separate from the skin")

    # Short hair: the scalp, pushed out, not the face.
    scalp = body[body[:, 1] > 1.52]
    # rebuild from faces whose centroid is on the scalp and not looking forward
    centers = body[np.array(body_faces)].mean(1)
    normals = np.cross(
        body[np.array(body_faces)[:, 1]] - body[np.array(body_faces)[:, 0]],
        body[np.array(body_faces)[:, 2]] - body[np.array(body_faces)[:, 0]],
    )
    nn = np.linalg.norm(normals, axis=1)
    normals = normals / np.maximum(nn[:, None], 1e-8)
    hair_faces = []
    hair_v = {}
    def vid(i, nrm):
        key = int(i)
        if key not in hair_v:
            hair_v[key] = len(hair_v)
            # store later
        return key
    sel = (centers[:, 1] > 1.50) & (normals[:, 2] < 0.55) & (normals[:, 1] > -0.2)
    raw = []
    for f, nrm, ok in zip(body_faces, normals, sel):
        if not ok:
            continue
        raw.append((f, nrm))
    # unique verts offset along an averaged normal
    acc = {}
    for f, nrm in raw:
        for i in f:
            acc.setdefault(i, []).append(nrm)
    keys = sorted(acc)
    remap = {i: k for k, i in enumerate(keys)}
    hv = []
    for i in keys:
        nrm = np.mean(acc[i], 0)
        nrm = nrm / (np.linalg.norm(nrm) or 1)
        hv.append(body[i] + nrm * 0.018 + np.array([0, 0.008, 0]))
    hf = [(remap[a], remap[b], remap[c]) for (a, b, c), _ in raw]
    write_obj(OUT / "hair.obj", np.array(hv), hf, "short hair cap")

    # Where the eyes sit, so the irises can be moved with the new head.
    for name, sl in (("right", body[:, 0] > 0), ("left", body[:, 0] < 0)):
        eye = body[sl & (body[:, 1] > 1.50) & (body[:, 1] < 1.60) & (body[:, 2] > 0.08)]
        if len(eye):
            print(name, "eye-region", eye.mean(0).round(4), "n", len(eye))
    print("height", round(body[:, 1].max(), 3))


if __name__ == "__main__":
    main()
