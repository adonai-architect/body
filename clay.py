"""One skin. Volumes only say where the clay bulges. Nothing is simulated underneath."""
import shutil
from pathlib import Path
import numpy as np

SRC = Path("/workspace/public/prefabs/baldur-base-skin.obj")
OUT = Path("/workspace/public/prefabs/baldur.obj")


def load(path):
    v, f = [], []
    for line in open(path):
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            ids = [int(p.split("/")[0]) - 1 for p in line.split()[1:]]
            for i in range(1, len(ids) - 1):
                f.append((ids[0], ids[i], ids[i + 1]))
    return np.array(v, float), np.array(f, np.int64)


def subdivide(v, f):
    mids = {}
    extra = []

    def mid(i, j):
        a, b = (int(i), int(j)) if i < j else (int(j), int(i))
        key = (a, b)
        if key not in mids:
            mids[key] = len(v) + len(extra)
            extra.append((v[a] + v[b]) * 0.5)
        return mids[key]

    nf = []
    for a, b, c in f:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        nf.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
    return np.vstack([v, np.array(extra)]), np.array(nf, np.int64)


def normals(v, f):
    n = np.zeros_like(v)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
    return n


def clay(v, nrm, center, radii, amount, weight, facing=None, mask=None):
    d = (v - np.asarray(center, float)) / np.asarray(radii, float)
    w = np.exp(-2.2 * np.sum(d * d, axis=1))
    if facing is not None:
        w *= np.clip(nrm @ np.asarray(facing, float), 0, 1)
    if mask is not None:
        w *= mask
    v += nrm * (amount * w)[:, None]
    weight += w
    return w


def trap(v, sign):
    """The shoulder line stays high from the neck out to the deltoid."""
    x = v[:, 0] * sign
    y = v[:, 1]
    z = v[:, 2]
    on = (x > 0.05) & (x < 0.22) & (y > 1.30) & (y < 1.48) & (z > -0.06) & (z < 0.08)
    t = np.clip((x - 0.05) / 0.14, 0, 1)
    lift = np.sin(np.pi * t) * 0.034
    v[on, 1] += lift[on]
    v[on, 0] += sign * lift[on] * 0.35


def solid_torso(v):
    """Chest comes forward. The arms move out so the ribcage side is its own wall."""
    x = np.abs(v[:, 0])
    y = v[:, 1]
    z = v[:, 2]
    sign = np.where(v[:, 0] >= 0.0, 1.0, -1.0)
    front = (y > 1.16) & (y < 1.35) & (x < 0.13) & (z > 0.07)
    pec = np.exp(-((x - 0.065) / 0.055) ** 2) * np.exp(-((y - 1.26) / 0.06) ** 2)
    v[front, 2] = v[front, 2] + (0.012 + 0.016 * pec[front])
    arm = (y < 1.20) & (y > 0.72) & (x > 0.155) & (z < 0.03)
    v[arm, 0] = v[arm, 0] + sign[arm] * 0.012
    return front | arm


def sculpt(v, f):
    """Shoulder, a real chest, and torso sides that are not the arms."""
    for s in (1.0, -1.0):
        trap(v, s)
    nrm = normals(v, f)
    w = np.zeros(len(v))
    torso = ~((v[:, 1] < 1.30) & (np.abs(v[:, 0]) > 0.175) & (v[:, 2] < 0.0))
    for s in (1.0, -1.0):
        clay(v, nrm, (s * 0.20, 1.36, 0.0), (0.06, 0.055, 0.05), 0.016, w, facing=(s, 0.15, 0.05), mask=torso)
        clay(v, nrm, (s * 0.055, 1.35, 0.10), (0.05, 0.035, 0.035), 0.008, w, facing=(0.0, 0.2, 1.0), mask=torso)
    moved = solid_torso(v)
    w[moved] += 0.4
    smooth(v, f, w, 0.15)
    return v


def smooth(v, f, weight, amount):
    acc = np.zeros_like(v)
    cnt = np.zeros(len(v))
    pairs = ((0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2))
    for i, j in pairs:
        np.add.at(acc, f[:, i], v[f[:, j]])
        np.add.at(cnt, f[:, i], 1)
    avg = acc / np.maximum(cnt, 1)[:, None]
    mask = weight > 0.15
    v[mask] = v[mask] * (1 - amount) + avg[mask] * amount


def write_obj(path, verts, faces, note):
    lines = [f"# {note}"]
    for p in verts:
        lines.append(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}")
    for a, b, c in faces:
        lines.append(f"f {a + 1} {b + 1} {c + 1}")
    Path(path).write_text("\n".join(lines) + "\n")
    print(path, len(verts), "verts", len(faces), "faces")


def hair_from(body, faces):
    """A haircut, not a second scalp and not a net. Thick on the crown, gone at the forehead."""
    nrm = normals(body, faces)
    y, z = body[:, 1], body[:, 2]
    rise = np.clip((y - 1.50) / 0.05, 0, 1)
    not_face = np.clip((0.42 - nrm[:, 2]) / 0.30, 0, 1)
    crown = np.clip((y - 1.58) / 0.05, 0, 1)
    weight = rise * not_face
    weight = np.clip(weight + crown * 0.5, 0, 1.4)
    clump = 0.004 * np.sin(body[:, 0] * 70) * np.sin(z * 55 + y * 40)
    offset = 0.010 * weight + 0.012 * crown + clump * weight
    keep = weight > 0.2
    used = keep[faces].all(axis=1)
    sel = faces[used]
    remap = {}
    verts = []
    for i in sel.reshape(-1):
        i = int(i)
        if i not in remap:
            remap[i] = len(verts)
            verts.append(body[i] + nrm[i] * offset[i])
    tris = [(remap[int(a)], remap[int(b)], remap[int(c)]) for a, b, c in sel]
    return np.array(verts, float), np.array(tris, np.int64)


def _along(points, poly):
    best_d = np.full(len(points), 1e9)
    best_s = np.zeros(len(points))
    walked = 0.0
    for i in range(len(poly) - 1):
        a = poly[i]
        b = poly[i + 1]
        ab = b - a
        length = np.linalg.norm(ab) or 1e-8
        u = ab / length
        t = np.clip((points - a) @ u, 0, length)
        d = np.linalg.norm(points - (a + t[:, None] * u), axis=1)
        better = d < best_d
        best_d = np.where(better, d, best_d)
        best_s = np.where(better, walked + t, best_s)
        walked += length
    return best_d, best_s, walked


def _swing(frm, to):
    a = frm / (np.linalg.norm(frm) or 1)
    b = to / (np.linalg.norm(to) or 1)
    v = np.cross(a, b)
    c = float(np.clip(np.dot(a, b), -1, 1))
    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], float)
    return np.eye(3) + k + k @ k * ((1 - c) / (s * s))


def pose_arms(v, f):
    """Hang the arms at the sides. Full turn, so the shoulder does not balloon."""
    right = np.array([
        [0.16, 1.38, 0.02],
        [0.19, 1.32, 0.02],
        [0.24, 1.24, 0.02],
        [0.30, 1.16, 0.04],
        [0.36, 1.10, 0.10],
        [0.42, 1.04, 0.18],
        [0.46, 0.98, 0.25],
    ], float)
    seam = np.zeros(len(v), bool)
    for sign, poly in ((1.0, right), (-1.0, right * np.array([-1.0, 1.0, 1.0]))):
        shoulder = poly[0]
        elbow = poly[3]
        dist, along, total = _along(v, poly)
        elbow_s = _along(elbow[None, :], poly)[1][0]
        radius = 0.065 + 0.07 * (along / total)
        cap = (v[:, 0] * sign > 0.10) & (np.abs(v[:, 0]) < 0.28) & (v[:, 1] > 1.28) & (v[:, 1] < 1.48) & (v[:, 2] < 0.12) & (v[:, 2] > -0.08)
        arm = ((dist < radius) & (along > 0.015)) | cap
        seam |= (dist < radius + 0.02) & (along > 0.03) & (along < 0.09)
        direction = elbow - shoulder
        target = np.array([sign * 0.04, -1.0, 0.02])
        target = target / np.linalg.norm(target) * np.linalg.norm(direction)
        R = _swing(direction, target)
        v[arm] = shoulder + (v[arm] - shoulder) @ R.T
        elbow2 = shoulder + (elbow - shoulder) @ R.T
        forearm = arm & (along > elbow_s + 0.03)
        if forearm.sum() < 10:
            continue
        wrist = v[forearm & (along > total * 0.82)].mean(0)
        fd = wrist - elbow2
        ft = np.array([sign * 0.015, -1.0, 0.025])
        ft = ft / np.linalg.norm(ft) * np.linalg.norm(fd)
        R2 = _swing(fd, ft)
        v[forearm] = elbow2 + (v[forearm] - elbow2) @ R2.T
        hand = v[forearm & (along > total * 0.85)]
        print("hand", sign, hand.mean(0).round(3) if len(hand) else None, "arm", int(arm.sum()))
    # Close the shoulder seam without puffing it.
    acc = np.zeros_like(v)
    cnt = np.zeros(len(v))
    for i, j in ((0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)):
        np.add.at(acc, f[:, i], v[f[:, j]])
        np.add.at(cnt, f[:, i], 1)
    avg = acc / np.maximum(cnt, 1)[:, None]
    v[seam] = v[seam] * 0.45 + avg[seam] * 0.55
    return v


def cover_groin(v):
    """The briefs are the body. No separate shape, no genitalia."""
    fall = np.exp(-(v[:, 0] / 0.04) ** 2) * np.exp(-((v[:, 1] - 0.88) / 0.055) ** 2)
    front = (v[:, 2] > 0.05) & (v[:, 1] > 0.80) & (v[:, 1] < 0.97) & (np.abs(v[:, 0]) < 0.07)
    pull = np.clip(fall * front, 0, 1) * 0.72
    v[:, 2] = v[:, 2] * (1 - pull) + 0.085 * pull
    return v


def main():
    if not SRC.exists():
        shutil.copy(OUT, SRC)
    v, f = load(SRC)
    # Hair and shorts are garments with their own shape. See attire.py.
    v, f = subdivide(v, f)
    v = sculpt(v, f)
    v = cover_groin(v)
    write_obj(OUT, v, f, "skin only. arms at rest.")


if __name__ == "__main__":
    main()
