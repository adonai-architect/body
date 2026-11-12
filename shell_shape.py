"""Skin shape from the 206-bone skeleton. Run on the box. The door only displays the file."""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

CELL = 0.005
SKIN = (0xC4, 0x9A, 0x78)
CLOTH = (0x6B, 0x5E, 0x3C)
CLOTH_DARK = (0x52, 0x48, 0x32)
HAIR = (0x24, 0x1E, 0x18)


class Shell:
    def __init__(self):
        self.p: list[float] = []
        self.n: list[float] = []
        self.c: list[int] = []
        self.i: list[int] = []

    def add(self, x, y, z, nx, ny, nz, color):
        j = 0.94 + 0.06 * abs(math.sin(x * 210 + y * 130 + z * 170))
        idx = len(self.p) // 3
        self.p += [x, y, z]
        self.n += [nx, ny, nz]
        self.c += [int(color[0] * j), int(color[1] * j), int(color[2] * j)]
        return idx

    def tri(self, a, b, c):
        self.i += [a, b, c]

    def tube(self, a, b, radius, color):
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        length = math.hypot(dx, dy, dz)
        if length < 0.006:
            return
        ax, ay, az = dx / length, dy / length, dz / length
        fx, fy, fz = 0.0, 0.0, 1.0
        along = fx * ax + fy * ay + fz * az
        fx, fy, fz = fx - along * ax, fy - along * ay, fz - along * az
        fl = math.hypot(fx, fy, fz)
        if fl < 0.25:
            fx, fy, fz, fl = 1.0, 0.0, 0.0, 1.0
        fx, fy, fz = fx / fl, fy / fl, fz / fl
        sx, sy, sz = ay * fz - az * fy, az * fx - ax * fz, ax * fy - ay * fx
        sl = math.hypot(sx, sy, sz) or 1.0
        sx, sy, sz = sx / sl, sy / sl, sz / sl
        guess = max(radius(0.3, 0.0), radius(0.5, math.pi), 0.01)
        rings = max(2, math.ceil(length / (CELL * 0.866)))
        around = max(8, round((math.pi * 2 * guess) / CELL))
        rows = []
        for r in range(rings + 1):
            t = r / rings
            px, py, pz = a[0] + dx * t, a[1] + dy * t, a[2] + dz * t
            twist = (r % 2) * (math.pi / around)
            row = []
            for k in range(around):
                ang = twist + (k / around) * math.pi * 2
                nx = math.cos(ang) * fx + math.sin(ang) * sx
                ny = math.cos(ang) * fy + math.sin(ang) * sy
                nz = math.cos(ang) * fz + math.sin(ang) * sz
                rad = radius(t, ang)
                row.append(self.add(px + nx * rad, py + ny * rad, pz + nz * rad, nx, ny, nz, color))
            rows.append(row)
        for r in range(rings):
            for k in range(around):
                k2 = (k + 1) % around
                a0, a1 = rows[r][k], rows[r][k2]
                b0, b1 = rows[r + 1][k], rows[r + 1][k2]
                self.tri(a0, b0, a1)
                self.tri(a1, b0, b1)

    def ring(self, y, rx, zf, zb, color, around, twist):
        row = []
        for k in range(around):
            ang = twist + (k / around) * math.pi * 2
            c, s = math.cos(ang), math.sin(ang)
            rz = zf if s > 0 else zb
            x, z = c * rx, s * rz
            nx, nz = c / rx, s / rz
            nl = math.hypot(nx, nz) or 1
            row.append(self.add(x, y, z, nx / nl, 0, nz / nl, color))
        return row

    def stitch(self, a, b):
        n = len(a)
        for k in range(n):
            k2 = (k + 1) % n
            self.tri(a[k], b[k], a[k2])
            self.tri(a[k2], b[k], b[k2])


def bell(t, at, wide):
    return math.exp(-(((t - at) / wide) ** 2))


def torso(y):
    hip = math.exp(-(((y - 0.94) / 0.08) ** 2))
    chest = math.exp(-(((y - 1.32) / 0.09) ** 2))
    ab = math.exp(-(((y - 1.16) / 0.07) ** 2))
    return (
        0.122 + hip * 0.034 + chest * 0.048,
        0.072 + ab * 0.016 + chest * 0.042,
        0.064 + hip * 0.046 + chest * 0.016,
    )


def muscle(kind, t, ang):
    front = max(0.0, math.cos(ang))
    back = max(0.0, -math.cos(ang))
    side = abs(math.sin(ang))
    if kind == "humerus":
        delt = bell(t, 0.08, 0.14) * 0.03
        belly = bell(t, 0.4, 0.18) * 0.018
        return 0.044 - 0.008 * t + delt * (0.45 + 0.55 * side) + belly * (0.5 + 0.5 * max(front, back))
    if kind == "ulna":
        return 0.036 - 0.008 * t + bell(t, 0.28, 0.2) * 0.012
    if kind == "femur":
        quad = bell(t, 0.48, 0.24) * 0.032
        return 0.06 - 0.012 * t + quad * (0.4 + 0.6 * front) + bell(t, 0.38, 0.22) * 0.014 * back + bell(t, 0.06, 0.12) * 0.016 * back
    if kind == "tibia":
        calf = bell(t, 0.3, 0.16) * 0.028
        return 0.034 - 0.002 * t + calf * (0.5 + 0.5 * back)
    return 0.02


def hang(name, p):
    arm = name.startswith(("humerus", "ulna", "metacarpal", "ph_hand"))
    if not arm:
        return p
    sign = -1 if name.endswith("_l") else 1
    drop = min(1.0, max(0.0, (1.42 - p[1]) / 0.7))
    return (p[0] + sign * 0.055 * drop, p[1], p[2])


def ends(world, name):
    pts = world[name]
    return pts[0], pts[-1]


def sculpt(x, y, z, c):
    rx, ry, rz = x - c[0], y - c[1], z - c[2]
    nose = math.exp(-((rx / 0.018) ** 2) - ((ry + 0.01) / 0.028) ** 2)
    if rz > 0:
        rz += 0.014 * nose
    if ry < -0.02:
        t = min(1.0, (-ry - 0.02) / 0.07)
        rx *= 1 - 0.22 * t
        if rz > 0 and ry < -0.045:
            rz += 0.008 * (1 - min(1.0, abs(rx) / 0.03))
    if rz > 0.02 and 0.01 < ry < 0.04:
        rz += 0.004
    for s in (-1, 1):
        dx, dy = rx - s * 0.032, ry - 0.012
        if dx * dx + dy * dy < 0.00045 and rz > 0:
            rz -= 0.009
    nl = math.hypot(rx, ry, rz) or 1
    return c[0] + rx, c[1] + ry, c[2] + rz, rx / nl, ry / nl, rz / nl


def build(world):
    skin, cloth = Shell(), Shell()
    y0, y1 = 0.9, 1.5
    rings = max(8, math.ceil((y1 - y0) / (CELL * 0.866)))
    around = max(16, round((2 * math.pi * 0.16) / CELL))
    prev = None
    for r in range(rings + 1):
        y = y0 + (y1 - y0) * r / rings
        rx, zf, zb = torso(y)
        row = skin.ring(y, rx, zf, zb, SKIN, around, (r % 2) * (math.pi / around))
        if prev:
            skin.stitch(prev, row)
        prev = row
    skin.tube(world["C7"][0], world["C1"][-1], lambda t, ang: 0.056 - t * 0.006, SKIN)
    for name in ("humerus_l", "humerus_r", "ulna_l", "ulna_r", "femur_l", "femur_r", "tibia_l", "tibia_r"):
        a, b = ends(world, name)
        kind = name.split("_")[0]
        skin.tube(hang(name, a), hang(name, b), lambda t, ang, kind=kind: muscle(kind, t, ang), SKIN)
    for name, pts in world.items():
        finger = name.startswith(("metacarpal_", "ph_hand_"))
        toe = name.startswith(("metatarsal_", "ph_foot_"))
        heel = name.startswith("calcaneus_")
        if not (finger or toe or heel):
            continue
        r = 0.009 if finger else 0.012 if toe else 0.028
        skin.tube(hang(name, pts[0]), hang(name, pts[-1]), lambda t, ang, r=r: r + (r * 0.85 - r) * t, SKIN)
    head_pts = world["occipital"] + world["frontal"]
    hc = [sum(p[i] for p in head_pts) / len(head_pts) for i in range(3)]
    sy0, sy1 = 0.5, 1.08
    s_rings = math.ceil((sy1 - sy0) / (CELL * 0.866))
    s_prev = None
    for r in range(s_rings + 1):
        y = sy0 + (sy1 - sy0) * r / s_rings
        if y < 0.98:
            left = cloth.ring(y, 0.125, 0.12, 0.115, CLOTH, 18, (r % 2) * (math.pi / 18))
            right = cloth.ring(y, 0.125, 0.12, 0.115, CLOTH, 18, (r % 2) * (math.pi / 18))
            for idx in left:
                cloth.p[idx * 3] -= 0.09
            for idx in right:
                cloth.p[idx * 3] += 0.09
            row = left + right
            if s_prev and len(s_prev) == len(row):
                cloth.stitch(s_prev, row)
            s_prev = row
        else:
            rx, zf, zb = torso(y)
            row = cloth.ring(y, rx + 0.012, zf + 0.01, zb + 0.008, CLOTH, 36, (r % 2) * (math.pi / 36))
            if s_prev and len(s_prev) == len(row) and y > 0.99:
                cloth.stitch(s_prev, row)
            s_prev = row
    for sign in (-1, 1):
        cloth.tube((sign * 0.2, 0.6, 0.06), (sign * 0.2, 0.76, 0.07), lambda t, ang: 0.058, CLOTH_DARK)
    return skin, cloth, hc


def pack(shell: Shell) -> bytes:
    return b"".join([
        struct.pack("<II", len(shell.p) // 3, len(shell.i)),
        struct.pack(f"<{len(shell.p)}f", *shell.p),
        struct.pack(f"<{len(shell.n)}f", *shell.n),
        bytes(shell.c),
        struct.pack(f"<{len(shell.i)}I", *shell.i),
    ])


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/baldur_shell.bin")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    # On the box the skeleton sits beside the loop, not beside this file.
    for candidate in (Path("/root/house/live_b20"), Path(__file__).resolve().parent):
        if (candidate / "skeleton.py").exists():
            sys.path.insert(0, str(candidate))
            break
    import skeleton
    sk = skeleton.Skeleton(1.75)
    if len(sk.bones) != 206:
        raise SystemExit(f"skeleton {len(sk.bones)}")
    skin, cloth, hc = build(sk.world())
    blob = b"BSH1" + struct.pack("<3f", *hc) + pack(skin) + pack(cloth)
    out.write_bytes(blob)
    print(f"wrote {out} bytes={len(blob)} skin={len(skin.p)//3} cloth={len(cloth.p)//3} head={tuple(round(v,3) for v in hc)}")


if __name__ == "__main__":
    main()
