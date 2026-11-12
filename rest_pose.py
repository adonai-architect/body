"""Rebuild the skin in the bind pose, then hang both arms as one piece."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/workspace/src/components/world")
from shape_to_ref import load_base, apply_target

OUT = Path("/workspace/public/prefabs/baldur-base-skin.obj")


def body_mesh():
    verts, faces = load_base()
    apply_target(verts, "maxmuscle", 1.0)
    apply_target(verts, "universal-male-young-averagemuscle-minweight", 0.35)
    apply_target(verts, "torso-muscle-pectoral-incr", 0.4)
    apply_target(verts, "l-upperarm-shoulder-muscle-incr", 0.55)
    apply_target(verts, "r-upperarm-shoulder-muscle-incr", 0.55)
    apply_target(verts, "l-upperarm-muscle-incr", 0.45)
    apply_target(verts, "r-upperarm-muscle-incr", 0.45)
    body = verts[:13380] * 0.1
    body[:, 1] -= body[:, 1].min()
    waist = (body[:, 1] > 0.95) & (body[:, 1] < 1.15) & (np.abs(body[:, 0]) < 0.2)
    body[waist, 0] *= 0.97
    return body, [f for f in faces if max(f) < 13380]


def neighbors(n, faces):
    adj = [[] for _ in range(n)]
    for a, b, c in faces:
        adj[a] += (b, c)
        adj[b] += (a, c)
        adj[c] += (a, b)
    return adj


def rot_from(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], float)
    return np.eye(3) + k + k @ k * ((1 - c) / (s * s))


def rot_axis(axis, ang):
    a = axis / (np.linalg.norm(axis) or 1)
    x, y, z = a
    c, s = np.cos(ang), np.sin(ang)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def swing(a, b):
    a = a / (np.linalg.norm(a) or 1)
    b = b / (np.linalg.norm(b) or 1)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    ang = np.arctan2(s, float(np.dot(a, b)))
    if s < 1e-8:
        return np.zeros(3), 0.0
    return axis / s, ang


def apply_swing(pts, origin, axis, ang, weight):
    out = pts.copy()
    for i, w in enumerate(weight):
        if w < 0.02:
            continue
        R = rot_axis(axis, ang * w)
        out[i] = origin + R @ (pts[i] - origin)
    return out


def hang(body, faces, sign):
    """The shoulder stays. The arm below it turns down, then the forearm hangs."""
    adj = neighbors(len(body), faces)
    seen = set(int(i) for i in np.where(body[:, 0] * sign > 0.38)[0])
    stack = list(seen)
    arm = []
    while stack:
        i = stack.pop()
        p = body[i]
        if p[1] > 1.50 or p[0] * sign < 0.10:
            continue
        if abs(p[0]) < 0.13 and p[1] < 1.46:
            continue
        arm.append(i)
        for j in adj[i]:
            if j not in seen:
                seen.add(j)
                stack.append(j)
    arm = np.array(arm, int)
    shoulder = np.array([sign * 0.16, 1.40, 0.02])
    distal = body[arm][body[arm][:, 0] * sign > 0.28]
    axis = distal.mean(0) - shoulder
    axis = axis / np.linalg.norm(axis)
    along = (body[arm] - shoulder) @ axis
    length = float(along.max())
    # Soft at the shoulder, full by the time the arm has left the chest.
    w = np.clip((along - 0.03) / 0.09, 0, 1)
    w = w * w * (3 - 2 * w)
    direction = axis * length
    target = np.array([sign * 0.06, -0.58, 0.02])
    ax, ang = swing(direction, target)
    body[arm] = apply_swing(body[arm], shoulder, ax, ang, w)
    elbow_at = length * 0.46
    elbow_band = arm[(along > elbow_at - 0.025) & (along < elbow_at + 0.025)]
    elbow = body[elbow_band].mean(0) if len(elbow_band) else shoulder + np.array([sign * 0.04, -0.28, 0.01])
    fore = arm[along > elbow_at + 0.02]
    if len(fore) > 20:
        dist = np.linalg.norm(body[fore] - elbow, axis=1)
        far = fore[np.argsort(dist)[-max(8, len(fore) // 6):]]
        wrist = body[far].mean(0)
        fd = wrist - elbow
        if np.linalg.norm(fd) > 0.05:
            ax2, ang2 = swing(fd, np.array([sign * 0.025, -0.50, 0.03]))
            R2 = rot_axis(ax2, ang2)
            body[fore] = elbow + (body[fore] - elbow) @ R2.T
    hand = body[arm][body[arm][:, 1] < 0.95]
    print(sign, "arm", len(arm), "hand", hand.mean(0).round(3) if len(hand) else None)
    return body, arm


def fitted_briefs(body, faces):
    """Boxer briefs taken off his own hips. A waistband, two leg holes, no boxes."""
    f = np.array(faces, int)
    tri = body[f]
    center = tri.mean(1)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-8)
    region = (center[:, 1] > 0.70) & (center[:, 1] < 1.015) & (np.abs(center[:, 0]) < 0.22)
    crotch = (np.abs(center[:, 0]) < 0.032) & (center[:, 1] < 0.84) & (fn[:, 1] > 0.2)
    keep = region & ~crotch
    sel = f[keep]
    vn = np.zeros_like(body)
    for k in range(3):
        np.add.at(vn, sel[:, k], fn[keep])
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-8)
    used = np.unique(sel.reshape(-1))
    remap = {int(i): n for n, i in enumerate(used)}
    verts = []
    for i in used:
        band = 0.004 if body[i, 1] > 0.985 else 0.0
        verts.append(body[i] + vn[i] * (0.008 + band))
    tris = [(remap[int(a)], remap[int(b)], remap[int(c)]) for a, b, c in sel]
    directed = {}
    for a, b, c in tris:
        directed[(a, b)] = True
        directed[(b, c)] = True
        directed[(c, a)] = True
    hem = {}
    extra = []
    inv = {n: int(i) for i, n in remap.items()}

    def hem_index(i):
        if i not in hem:
            old = inv[i]
            hem[i] = len(verts) + len(extra)
            extra.append(body[old] + vn[old] * 0.002)
        return hem[i]

    for a, b in list(directed):
        if (b, a) in directed:
            continue
        ha, hb = hem_index(a), hem_index(b)
        tris.append((a, b, hb))
        tris.append((a, hb, ha))
    verts.extend(extra)
    return np.array(verts, float), tris


def main():
    body, faces = body_mesh()
    body, right = hang(body, faces, 1.0)
    body, left = hang(body, faces, -1.0)
    # Clear the hips so the hands are not inside the briefs.
    for sign, arm in ((1.0, right), (-1.0, left)):
        low = arm[body[arm][:, 1] < 1.05]
        body[low, 0] += sign * 0.035
        body[low, 2] += 0.045
    lines = ["# skin, arms down"]
    for p in body:
        lines.append(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}")
    for a, b, c in faces:
        lines.append(f"f {a + 1} {b + 1} {c + 1}")
    OUT.write_text("\n".join(lines) + "\n")
    bv, bf = fitted_briefs(body, faces)
    brief_lines = ["# fitted boxer briefs"]
    for p in bv:
        brief_lines.append(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}")
    for a, b, c in bf:
        brief_lines.append(f"f {a + 1} {b + 1} {c + 1}")
    Path("/workspace/public/prefabs/briefs.obj").write_text("\n".join(brief_lines) + "\n")
    print("briefs", len(bv), "verts", "max |x|", round(np.abs(body[:, 0]).max(), 3))


if __name__ == "__main__":
    main()
