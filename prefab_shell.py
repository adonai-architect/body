"""One shell per visible zero-point. The surface is the shape. Run on the box.

A prefab is an OBJ you can open and change. The body is one surface:
the face is a face, the limbs join the torso. The eyes do not store it.
They ask where a direction hits it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

STEP = 0.006


def smin(a, b, k):
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def length(v):
    return np.sqrt(np.sum(v * v, axis=-1))


def capsule(p, a, b, r):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ba = b - a
    pa = p - a
    h = np.clip(np.sum(pa * ba, axis=-1) / np.dot(ba, ba), 0.0, 1.0)
    return length(pa - ba * h[..., None]) - r


def sphere(p, c, r):
    return length(p - np.asarray(c, np.float64)) - r


def ellipsoid(p, c, r):
    r = np.asarray(r, np.float64)
    k = (p - np.asarray(c, np.float64)) / r
    return (length(k) - 1.0) * float(np.min(r))


def body_field(p, world, head):
    d = np.full(p.shape[:-1], 8.0)
    def u(a, k=0.04):
        nonlocal d
        d = smin(d, a, k)
    hc = np.asarray(head, np.float64)
    skull = ellipsoid(p, hc + (0, 0.01, 0.0), (0.084, 0.108, 0.094))
    jaw = ellipsoid(p, hc + (0, -0.048, 0.012), (0.072, 0.048, 0.068))
    head_d = smin(skull, jaw, 0.035)
    chin = sphere(p, hc + (0, -0.086, 0.045), 0.022)
    brow = capsule(p, hc + (-0.045, 0.028, 0.072), hc + (0.045, 0.03, 0.078), 0.011)
    nose = capsule(p, hc + (0, -0.002, 0.075), hc + (0, -0.03, 0.15), 0.013)
    tip = sphere(p, hc + (0, -0.03, 0.128), 0.016)
    upper = capsule(p, hc + (-0.022, -0.048, 0.088), hc + (0.022, -0.046, 0.092), 0.008)
    lower = capsule(p, hc + (-0.02, -0.062, 0.082), hc + (0.02, -0.06, 0.086), 0.009)
    head_d = smin(head_d, chin, 0.03)
    head_d = smin(head_d, brow, 0.02)
    head_d = smin(head_d, nose, 0.02)
    head_d = smin(head_d, tip, 0.018)
    head_d = smin(head_d, upper, 0.012)
    head_d = smin(head_d, lower, 0.012)
    for sign in (-1.0, 1.0):
        ear = ellipsoid(p, hc + (sign * 0.088, -0.01, 0.0), (0.012, 0.028, 0.018))
        head_d = smin(head_d, ear, 0.012)
        eye = sphere(p, hc + (sign * 0.03, 0.015, 0.09), 0.018)
        head_d = np.maximum(head_d, -eye)
    u(head_d, 0.025)
    u(capsule(p, (0, 1.45, 0.015), hc + (0, -0.07, 0.012), 0.048), 0.028)
    u(ellipsoid(p, (0, 0.97, -0.015), (0.155, 0.11, 0.115)), 0.055)
    u(ellipsoid(p, (0, 1.15, 0.01), (0.128, 0.08, 0.10)), 0.05)
    u(ellipsoid(p, (0, 1.35, 0.02), (0.17, 0.12, 0.112)), 0.05)
    limbs = {
        "humerus": 0.046, "ulna": 0.032, "femur": 0.072, "tibia": 0.038,
    }
    for name, pts in world.items():
        kind = name.split("_")[0]
        if kind not in limbs:
            continue
        a, b = np.array(pts[0]), np.array(pts[-1])
        if kind in ("humerus", "ulna"):
            sign = -1.0 if name.endswith("_l") else 1.0
            drop_a = min(1.0, max(0.0, (1.42 - a[1]) / 0.7))
            drop_b = min(1.0, max(0.0, (1.42 - b[1]) / 0.7))
            a = a + np.array([sign * 0.04 * drop_a, 0, 0])
            b = b + np.array([sign * 0.05 * drop_b, 0, 0])
        u(capsule(p, a, b, limbs[kind]), 0.035)
        if kind == "tibia":
            back = (a + b) / 2 + np.array([0, 0.04, -0.025])
            u(sphere(p, back, 0.055), 0.03)
        if kind == "femur":
            front = a * 0.45 + b * 0.55 + np.array([0, 0, 0.03])
            u(sphere(p, front, 0.08), 0.03)
    # hands and feet, small, so the limbs end in something
    for name, pts in world.items():
        if name.startswith("metacarpal_3_"):
            u(capsule(p, pts[0], pts[-1], 0.018), 0.02)
        if name.startswith("calcaneus_"):
            u(sphere(p, pts[-1], 0.028), 0.02)
        if name.startswith("metatarsal_3_"):
            u(capsule(p, pts[0], pts[-1], 0.016), 0.02)
    return d


def shorts_field(p):
    d = np.full(p.shape[:-1], 8.0)
    d = smin(d, ellipsoid(p, (0, 1.0, 0.0), (0.175, 0.1, 0.13)), 0.04)
    for sign in (-0.1, 0.1):
        d = smin(d, capsule(p, (sign, 0.98, 0.0), (sign, 0.55, 0.02), 0.09), 0.03)
        d = smin(d, sphere(p, (sign * 1.6, 0.68, 0.08), 0.045), 0.02)
    # open at the bottom and the waist is just the cloth shell, not a solid
    return d


def stone_field(p):
    c = np.array([0.0, 0.22, 0.0])
    d = ellipsoid(p, c, (0.34, 0.22, 0.28))
    d = smin(d, sphere(p, c + (0.12, 0.08, 0.06), 0.16), 0.08)
    d = smin(d, sphere(p, c + (-0.1, 0.02, -0.05), 0.14), 0.08)
    return d


def light_field(p):
    d = capsule(p, (0, 0.05, 0), (0, 1.15, 0), 0.035)
    d = smin(d, sphere(p, (0, 1.28, 0), 0.09), 0.04)
    return d


def island_field(p):
    radial = np.sqrt(p[..., 0] ** 2 + p[..., 2] ** 2)
    disc = np.maximum(radial - 14.0, np.abs(p[..., 1] - 0.0) - 0.08)
    return disc


def surface_net(field, origin, step):
    inside = field < 0
    nx, ny, nz = field.shape
    mixed = np.zeros((nx - 1, ny - 1, nz - 1), bool)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                mixed |= inside[dx:nx - 1 + dx, dy:ny - 1 + dy, dz:nz - 1 + dz]
    # a cell is mixed only if corners disagree. The OR above marks any inside corner.
    all_in = np.ones_like(mixed)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                all_in &= inside[dx:nx - 1 + dx, dy:ny - 1 + dy, dz:nz - 1 + dz]
    any_in = mixed
    mixed = any_in & ~all_in
    xs, ys, zs = np.where(mixed)
    vid = -np.ones((nx - 1, ny - 1, nz - 1), np.int32)
    vid[xs, ys, zs] = np.arange(len(xs), dtype=np.int32)
    edges = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))
    corner = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1))
    verts = np.zeros((len(xs), 3), np.float64)
    grads = np.zeros_like(verts)
    for n, (i, j, k) in enumerate(zip(xs.tolist(), ys.tolist(), zs.tolist())):
        cvals = []
        cpos = []
        for dx, dy, dz in corner:
            cvals.append(field[i + dx, j + dy, k + dz])
            cpos.append(origin + np.array([i + dx, j + dy, k + dz]) * step)
        acc = np.zeros(3)
        count = 0
        for a, b in edges:
            va, vb = cvals[a], cvals[b]
            if (va < 0) == (vb < 0):
                continue
            t = va / (va - vb)
            acc += cpos[a] * (1 - t) + cpos[b] * t
            count += 1
        if count == 0:
            acc = cpos[0]
            count = 1
        verts[n] = acc / count
        ix, iy, iz = int(i), int(j), int(k)
        def at(x, y, z):
            x = min(max(x, 0), nx - 1)
            y = min(max(y, 0), ny - 1)
            z = min(max(z, 0), nz - 1)
            return field[x, y, z]
        grads[n] = (
            at(ix + 1, iy, iz) - at(ix - 1, iy, iz),
            at(ix, iy + 1, iz) - at(ix, iy - 1, iz),
            at(ix, iy, iz + 1) - at(ix, iy, iz - 1),
        )
    tris = []
    def emit(cells):
        ids = []
        for c in cells:
            if c[0] < 0 or c[1] < 0 or c[2] < 0 or c[0] >= nx - 1 or c[1] >= ny - 1 or c[2] >= nz - 1:
                return
            v = vid[c]
            if v < 0:
                return
            ids.append(int(v))
        if len(set(ids)) < 4:
            return
        tris.append(ids)
    # quads across each sign-changing grid edge
    for axis, pairs in (
        (0, ((0, -1, -1), (0, 0, -1), (0, 0, 0), (0, -1, 0))),
        (1, ((-1, 0, -1), (0, 0, -1), (0, 0, 0), (-1, 0, 0))),
        (2, ((-1, -1, 0), (0, -1, 0), (0, 0, 0), (-1, 0, 0))),
    ):
        a = inside
        b = np.roll(inside, -1, axis=axis)
        # roll wraps; mask the last plane
        change = a != b
        if axis == 0:
            change[-1, :, :] = False
        elif axis == 1:
            change[:, -1, :] = False
        else:
            change[:, :, -1] = False
        ii, jj, kk = np.where(change)
        for i, j, k in zip(ii.tolist(), jj.tolist(), kk.tolist()):
            emit([(i + dx, j + dy, k + dz) for dx, dy, dz in pairs])
    return verts, grads, tris


def write_obj(path: Path, verts, grads, tris):
    lines = ["# prefab — alter this mesh, the zero point keeps its name"]
    for v in verts:
        lines.append(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}")
    norms = []
    for g in grads:
        n = np.linalg.norm(g) or 1.0
        norms.append(g / n)
        lines.append(f"vn {g[0] / n:.4f} {g[1] / n:.4f} {g[2] / n:.4f}")
    for ids in tris:
        # two triangles, face outward (gradient)
        a, b, c, d = ids
        mid = (verts[a] + verts[b] + verts[c] + verts[d]) / 4
        outward = norms[a] + norms[b] + norms[c] + norms[d]
        tri1 = (a, b, c)
        nrm = np.cross(verts[b] - verts[a], verts[c] - verts[a])
        if np.dot(nrm, outward) < 0:
            tri1 = (a, c, b)
        tri2 = (a, c, d)
        nrm = np.cross(verts[c] - verts[a], verts[d] - verts[a])
        if np.dot(nrm, outward) < 0:
            tri2 = (a, d, c)
        for t in (tri1, tri2):
            lines.append(f"f {t[0]+1}//{t[0]+1} {t[1]+1}//{t[1]+1} {t[2]+1}//{t[2]+1}")
    path.write_text("\n".join(lines) + "\n")


def march(fn, origin, counts):
    origin = np.asarray(origin, np.float64)
    counts = np.asarray(counts, np.int32)
    field = np.empty(counts, np.float32)
    xs = origin[0] + np.arange(counts[0]) * STEP
    zs = origin[2] + np.arange(counts[2]) * STEP
    for j in range(counts[1]):
        y = origin[1] + j * STEP
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        P = np.stack([X, np.full_like(X, y), Z], axis=-1)
        field[:, j, :] = fn(P).astype(np.float32)
    return surface_net(field, origin, STEP)


def write_shorts(path: Path):
    """A cloth shell, open at the leg, not a solid blob. Editable quads."""
    lines = ["# prefab — cargo shorts"]
    verts = []
    def add(x, y, z):
        verts.append((x, y, z))
        lines.append(f"v {x:.4f} {y:.4f} {z:.4f}")
        return len(verts)
    rings = []
    for i, y in enumerate(np.linspace(0.55, 0.96, 8)):
        row = []
        for side in (-1, 1):
            for k in range(10):
                a = k / 10 * np.pi * 2
                row.append(add(side * 0.1 + np.cos(a) * 0.095, y, np.sin(a) * 0.08))
        rings.append(row)
    waist = []
    for k in range(20):
        a = k / 20 * np.pi * 2
        waist.append(add(np.cos(a) * 0.17, 1.02, np.sin(a) * 0.12))
    for row, nxt in zip(rings, rings[1:]):
        for k in range(10):
            a, b = row[k], row[(k + 1) % 10]
            c, d = nxt[k], nxt[(k + 1) % 10]
            lines.append(f"f {a} {c} {b}")
            lines.append(f"f {b} {c} {d}")
            a, b = row[10 + k], row[10 + (k + 1) % 10]
            c, d = nxt[10 + k], nxt[10 + (k + 1) % 10]
            lines.append(f"f {a} {c} {b}")
            lines.append(f"f {b} {c} {d}")
    top = rings[-1]
    for k in range(20):
        a, b = waist[k], waist[(k + 1) % 20]
        # stitch waist down toward the two thigh rings by nearest
        lines.append(f"f {a} {b} {top[k % 20]}")
    path.write_text("\n".join(lines) + "\n")


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/prefabs")
    out.mkdir(parents=True, exist_ok=True)
    for candidate in (Path("/root/house/live_b20"), Path("/tmp/heliopolis-house/vex_scratch")):
        if (candidate / "skeleton.py").exists():
            sys.path.insert(0, str(candidate))
            break
    import skeleton
    sk = skeleton.Skeleton(1.75)
    world = sk.world()
    head_pts = world["occipital"] + world["frontal"]
    head = [sum(p[i] for p in head_pts) / len(head_pts) for i in range(3)]
    print("head", [round(v, 3) for v in head], flush=True)

    origin = (-0.5, -0.02, -0.32)
    counts = (int(1.0 / STEP), int(1.92 / STEP), int(0.64 / STEP))
    print("body grid", counts, flush=True)
    v, g, t = march(lambda p: body_field(p, world, head), origin, counts)
    write_obj(out / "baldur.obj", v, g, t)
    print("baldur", len(v), "verts", len(t), "quads", flush=True)

    write_shorts(out / "cargo_shorts.obj")
    print("shorts written", flush=True)

    v, g, t = march(stone_field, (-0.45, 0.0, -0.4), (int(0.9 / STEP), int(0.5 / STEP), int(0.8 / STEP)))
    write_obj(out / "stone.obj", v, g, t)
    v, g, t = march(light_field, (-0.15, 0.0, -0.15), (int(0.3 / STEP), int(1.45 / STEP), int(0.3 / STEP)))
    write_obj(out / "light.obj", v, g, t)
    # island is wide; a coarse step would be wrong for the body. A disc does not need 6mm.
    # Write it as an editable ring of quads directly.
    ring = ["# prefab — the island"]
    segs = 64
    for i in range(segs):
        a = i / segs * np.pi * 2
        ring.append(f"v {14.0*np.cos(a):.4f} 0 {14.0*np.sin(a):.4f}")
    ring.append("v 0 0 0")
    for i in range(segs):
        ring.append(f"f {segs+1} {i+1} {(i+1)%segs+1}")
    (out / "island.obj").write_text("\n".join(ring) + "\n")

    manifest = {
        "note": "Each visible zero point has one prefab. Edit the OBJ. The name stays.",
        "objects": [
            {"name": "baldur", "kind": "body", "at": [0, 0, 0], "file": "baldur.obj"},
            {"name": "cargo shorts", "kind": "cloth", "at": [0, 0, 0], "file": "cargo_shorts.obj"},
            {"name": "the edge stone", "kind": "stone", "at": [1.4, 0, 0.6], "file": "stone.obj"},
            {"name": "the light", "kind": "light", "at": [-1.6, 0, 0.4], "file": "light.obj"},
            {"name": "the island", "kind": "ground", "at": [0, 0, 0], "file": "island.obj"},
        ],
    }
    (out / "prefabs.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
