"""Hang the arms at the sides. The shoulder blends. The rest of the body stays."""
from pathlib import Path
import numpy as np

PATH = Path("/workspace/public/prefabs/baldur-base-skin.obj")


def load(path):
    v, f, lines = [], [], []
    for line in open(path):
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
            lines.append("v")
        else:
            lines.append(line)
            if line.startswith("f "):
                ids = [int(p.split("/")[0]) - 1 for p in line.split()[1:]]
                for i in range(1, len(ids) - 1):
                    f.append((ids[0], ids[i], ids[i + 1]))
    return np.array(v, float), lines


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


def hang(v, sign):
    hand = v[(v[:, 0] * sign > 0.34) & (v[:, 1] < 1.2)]
    shoulder = np.array([sign * 0.16, 1.40, 0.02])
    h = hand.mean(0)
    arm = h - shoulder
    arm = arm / np.linalg.norm(arm)
    target = np.array([sign * 0.23, 0.74, 0.05]) - shoulder
    target = target / np.linalg.norm(target)
    axis = np.cross(arm, target)
    ang = np.arctan2(np.linalg.norm(axis), np.dot(arm, target))
    R = rot_axis(axis, ang)
    # Which points are the arm: close to the arm line, out past the shoulder.
    d = v - shoulder
    along = d @ arm
    radial = np.linalg.norm(d - along[:, None] * arm, axis=1)
    arm_pt = (along > 0.0) & (radial < 0.105) & (v[:, 0] * sign > 0.05)
    w = np.clip((along - 0.015) / 0.07, 0, 1)
    w = np.where(arm_pt, w, 0)
    # Partial rotation keeps the shoulder attached.
    moved = 0
    for i in np.where(w > 0)[0]:
        Ri = rot_axis(axis, ang * w[i])
        v[i] = shoulder + Ri @ (v[i] - shoulder)
        moved += 1
    print(sign, "moved", moved, "hand was", h.round(3), "now", v[(v[:, 0] * sign > 0.15) & (v[:, 1] < 0.95)].mean(0).round(3) if True else "")
    return v


def main():
    v, lines = load(PATH)
    v = hang(v, 1.0)
    v = hang(v, -1.0)
    out = []
    vi = 0
    for line in lines:
        if line == "v":
            p = v[vi]
            vi += 1
            out.append(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}")
        else:
            out.append(line if line.endswith("\n") else line)
    # lines that weren't vertices were stored with their newline from the file
    text = []
    vi = 0
    for line in open(PATH):
        if line.startswith("v "):
            p = v[vi]
            vi += 1
            text.append(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
        else:
            text.append(line)
    PATH.write_text("".join(text))
    print("wrote", PATH, "verts", vi)


if __name__ == "__main__":
    main()
