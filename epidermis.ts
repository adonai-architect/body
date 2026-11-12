import * as THREE from "three";
import bones from "./bones.json";

type Bone = { n: string; p: number[][]; d: [string, number, number][] };

const CELL = 0.005;
const SKIN = 0xc49a78;
const CLOTH = 0x6b5e3c;
const CLOTH_DARK = 0x524832;
const HAIR = 0x241e18;

const ALL = bones as Bone[];
const BY = new Map(ALL.map((b) => [b.n, b]));

/** Joint range from the bone itself. Skin does not get to invent a bend. */
export const LIMITS = ALL.filter((b) => b.d.length > 0).length;

class Shell {
  p: number[] = [];
  nrm: number[] = [];
  col: number[] = [];
  flex: number[] = [];
  idx: number[] = [];

  add(x: number, y: number, z: number, nx: number, ny: number, nz: number, color: number, flex: number) {
    const id = this.p.length / 3;
    const j = 0.94 + 0.06 * Math.abs(Math.sin(x * 210 + y * 130 + z * 170));
    this.p.push(x, y, z);
    this.nrm.push(nx, ny, nz);
    this.col.push((((color >> 16) & 255) / 255) * j, (((color >> 8) & 255) / 255) * j, ((color & 255) / 255) * j);
    this.flex.push(flex);
    return id;
  }

  tri(a: number, b: number, c: number) {
    this.idx.push(a, b, c);
  }

  tube(a: number[], b: number[], radius: (t: number, ang: number) => number, color: number, flex: number) {
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const dz = b[2] - a[2];
    const len = Math.hypot(dx, dy, dz);
    if (len < 0.006) return;
    const ax = dx / len;
    const ay = dy / len;
    const az = dz / len;
    let fx = 0;
    let fy = 0;
    let fz = 1;
    const along = fx * ax + fy * ay + fz * az;
    fx -= along * ax;
    fy -= along * ay;
    fz -= along * az;
    let fl = Math.hypot(fx, fy, fz);
    if (fl < 0.25) {
      fx = 1;
      fy = 0;
      fz = 0;
      fl = 1;
    }
    fx /= fl;
    fy /= fl;
    fz /= fl;
    let sx = ay * fz - az * fy;
    let sy = az * fx - ax * fz;
    let sz = ax * fy - ay * fx;
    const sl = Math.hypot(sx, sy, sz) || 1;
    sx /= sl;
    sy /= sl;
    sz /= sl;
    const guess = Math.max(radius(0.3, 0), radius(0.5, Math.PI), 0.01);
    const rings = Math.max(2, Math.ceil(len / (CELL * 0.866)));
    const around = Math.max(8, Math.round((Math.PI * 2 * guess) / CELL));
    const rows: number[][] = [];
    for (let r = 0; r <= rings; r++) {
      const t = r / rings;
      const px = a[0] + dx * t;
      const py = a[1] + dy * t;
      const pz = a[2] + dz * t;
      const twist = (r % 2) * (Math.PI / around);
      const row: number[] = [];
      for (let k = 0; k < around; k++) {
        const ang = twist + (k / around) * Math.PI * 2;
        const nx = Math.cos(ang) * fx + Math.sin(ang) * sx;
        const ny = Math.cos(ang) * fy + Math.sin(ang) * sy;
        const nz = Math.cos(ang) * fz + Math.sin(ang) * sz;
        const here = radius(t, ang);
        row.push(this.add(px + nx * here, py + ny * here, pz + nz * here, nx, ny, nz, color, flex));
      }
      rows.push(row);
    }
    for (let r = 0; r < rings; r++) {
      for (let k = 0; k < around; k++) {
        const k2 = (k + 1) % around;
        const a0 = rows[r][k];
        const a1 = rows[r][k2];
        const b0 = rows[r + 1][k];
        const b1 = rows[r + 1][k2];
        this.tri(a0, b0, a1);
        this.tri(a1, b0, b1);
      }
    }
  }

  ring(y: number, rx: number, zf: number, zb: number, color: number, flex: number, around: number, twist: number) {
    const row: number[] = [];
    for (let k = 0; k < around; k++) {
      const ang = twist + (k / around) * Math.PI * 2;
      const c = Math.cos(ang);
      const s = Math.sin(ang);
      const rz = s > 0 ? zf : zb;
      const x = c * rx;
      const z = s * rz;
      const nx = c / rx;
      const nz = s / rz;
      const nl = Math.hypot(nx, nz) || 1;
      row.push(this.add(x, y, z, nx / nl, 0, nz / nl, color, flex));
    }
    return row;
  }

  stitch(a: number[], b: number[]) {
    const n = a.length;
    for (let k = 0; k < n; k++) {
      const k2 = (k + 1) % n;
      this.tri(a[k], b[k], a[k2]);
      this.tri(a[k2], b[k], b[k2]);
    }
  }

  geo(roughness: number, sheen: number) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(this.p, 3));
    g.setAttribute("normal", new THREE.Float32BufferAttribute(this.nrm, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(this.col, 3));
    g.setAttribute("flex", new THREE.Float32BufferAttribute(this.flex, 1));
    g.setIndex(this.idx);
    const mat = new THREE.MeshPhysicalMaterial({
      vertexColors: true,
      roughness,
      metalness: 0,
      sheen,
      sheenColor: new THREE.Color(0xd9a08c),
      sheenRoughness: 0.55,
      emissive: 0x2a1814,
      emissiveIntensity: 0.1,
    });
    mat.onBeforeCompile = (shader) => {
      shader.uniforms.breath = { value: 0 };
      shader.vertexShader = shader.vertexShader
        .replace("#include <common>", "#include <common>\nattribute float flex;\nuniform float breath;")
        .replace("#include <begin_vertex>", "vec3 transformed = vec3(position) + objectNormal * breath * flex;");
      mat.userData.breath = shader.uniforms.breath;
    };
    return new THREE.Mesh(g, mat);
  }
}

function ends(name: string) {
  const b = BY.get(name);
  if (!b) return null;
  return [b.p[0], b.p[b.p.length - 1]] as [number[], number[]];
}

function torsoRadii(y: number) {
  const hip = Math.exp(-(((y - 0.94) / 0.08) ** 2));
  const chest = Math.exp(-(((y - 1.32) / 0.09) ** 2));
  const ab = Math.exp(-(((y - 1.16) / 0.07) ** 2));
  return {
    rx: 0.122 + hip * 0.034 + chest * 0.048,
    zf: 0.072 + ab * 0.016 + chest * 0.042,
    zb: 0.064 + hip * 0.046 + chest * 0.016,
  };
}

function muscle(kind: string, t: number, ang: number) {
  const front = Math.max(0, Math.cos(ang));
  const back = Math.max(0, -Math.cos(ang));
  const side = Math.abs(Math.sin(ang));
  const bell = (at: number, wide: number) => Math.exp(-(((t - at) / wide) ** 2));
  if (kind === "humerus") {
    const delt = bell(0.08, 0.14) * 0.03;
    const belly = bell(0.4, 0.18) * 0.018;
    return 0.044 - 0.008 * t + delt * (0.45 + 0.55 * side) + belly * (0.5 + 0.5 * Math.max(front, back));
  }
  if (kind === "ulna") return 0.036 - 0.008 * t + bell(0.28, 0.2) * 0.012;
  if (kind === "femur") {
    const quad = bell(0.48, 0.24) * 0.032;
    return 0.06 - 0.012 * t + quad * (0.4 + 0.6 * front) + bell(0.38, 0.22) * 0.014 * back + bell(0.06, 0.12) * 0.016 * back;
  }
  if (kind === "tibia") {
    const calf = bell(0.3, 0.16) * 0.028;
    return 0.034 - 0.002 * t + calf * (0.5 + 0.5 * back);
  }
  return 0.02;
}

function hang(name: string, p: number[]) {
  const arm = name.startsWith("humerus") || name.startsWith("ulna") || name.startsWith("metacarpal") || name.startsWith("ph_hand");
  if (!arm) return p;
  const sign = name.endsWith("_l") ? -1 : 1;
  const drop = Math.min(1, Math.max(0, (1.42 - p[1]) / 0.7));
  return [p[0] + sign * 0.055 * drop, p[1], p[2]];
}

function headCenter() {
  const o = BY.get("occipital");
  const f = BY.get("frontal");
  const pts = [...(o?.p ?? []), ...(f?.p ?? [])];
  const c = [0, 0, 0];
  if (!pts.length) return c;
  for (const p of pts) {
    c[0] += p[0];
    c[1] += p[1];
    c[2] += p[2];
  }
  return [c[0] / pts.length, c[1] / pts.length, c[2] / pts.length];
}

function sculpt(x: number, y: number, z: number, c: number[]) {
  let rx = x - c[0];
  let ry = y - c[1];
  let rz = z - c[2];
  const nose = Math.exp(-((rx / 0.018) ** 2) - ((ry + 0.01) / 0.028) ** 2);
  if (rz > 0) rz += 0.014 * nose;
  if (ry < -0.02) {
    const t = Math.min(1, (-ry - 0.02) / 0.07);
    rx *= 1 - 0.22 * t;
    if (rz > 0 && ry < -0.045) rz += 0.008 * (1 - Math.min(1, Math.abs(rx) / 0.03));
  }
  if (rz > 0.02 && ry > 0.01 && ry < 0.04) rz += 0.004;
  for (const s of [-1, 1]) {
    const dx = rx - s * 0.032;
    const dy = ry - 0.012;
    if (dx * dx + dy * dy < 0.00045 && rz > 0) rz -= 0.009;
  }
  const nl = Math.hypot(rx, ry, rz) || 1;
  return { x: c[0] + rx, y: c[1] + ry, z: c[2] + rz, nx: rx / nl, ny: ry / nl, nz: rz / nl };
}

export function pulseSkin(root: THREE.Object3D, t: number) {
  const v = Math.sin(t * 1.4) * 0.007;
  root.traverse((o) => {
    const mesh = o as THREE.Mesh;
    const mat = mesh.material as THREE.MeshPhysicalMaterial | undefined;
    const breath = mat?.userData?.breath as { value: number } | undefined;
    if (breath) breath.value = v;
  });
}

export function buildBaldur() {
  if (ALL.length !== 206) throw new Error(`skeleton ${ALL.length}, not 206`);
  const skin = new Shell();
  const cloth = new Shell();

  const y0 = 0.9;
  const y1 = 1.5;
  const rings = Math.max(8, Math.ceil((y1 - y0) / (CELL * 0.866)));
  const around = Math.max(16, Math.round((2 * Math.PI * 0.16) / CELL));
  let prev: number[] | null = null;
  for (let r = 0; r <= rings; r++) {
    const y = y0 + ((y1 - y0) * r) / rings;
    const rad = torsoRadii(y);
    const row = skin.ring(y, rad.rx, rad.zf, rad.zb, SKIN, y > 1.12 && y < 1.42 ? 1 : 0.25, around, (r % 2) * (Math.PI / around));
    if (prev) skin.stitch(prev, row);
    prev = row;
  }

  const neckA = ends("C7");
  const neckB = ends("C1");
  if (neckA && neckB) skin.tube(neckA[0], neckB[1], (t) => 0.056 - t * 0.006, SKIN, 0.4);

  for (const name of ["humerus_l", "humerus_r", "ulna_l", "ulna_r", "femur_l", "femur_r", "tibia_l", "tibia_r"]) {
    const e = ends(name);
    if (!e) continue;
    const kind = name.split("_")[0];
    skin.tube(hang(name, e[0]), hang(name, e[1]), (t, ang) => muscle(kind, t, ang), SKIN, kind === "humerus" || kind === "femur" ? 0.35 : 0.2);
  }
  for (const b of ALL) {
    const finger = b.n.startsWith("metacarpal_") || b.n.startsWith("ph_hand_");
    const toe = b.n.startsWith("metatarsal_") || b.n.startsWith("ph_foot_");
    const heel = b.n.startsWith("calcaneus_");
    if (!finger && !toe && !heel) continue;
    const r = finger ? 0.009 : toe ? 0.012 : 0.028;
    skin.tube(hang(b.n, b.p[0]), hang(b.n, b.p[b.p.length - 1]), (t) => r + (r * 0.85 - r) * t, SKIN, 0.15);
  }

  const hc = headCenter();
  const hRings = 28;
  const hAround = 32;
  const headRows: number[][] = [];
  for (let r = 1; r < hRings; r++) {
    const v = (r / hRings) * Math.PI;
    const row: number[] = [];
    const twist = (r % 2) * (Math.PI / hAround);
    for (let k = 0; k < hAround; k++) {
      const u = twist + (k / hAround) * Math.PI * 2;
      const x = hc[0] + Math.sin(v) * Math.cos(u) * 0.1;
      const y = hc[1] + Math.cos(v) * 0.118;
      const z = hc[2] + Math.sin(v) * Math.sin(u) * 0.105;
      const s = sculpt(x, y, z, hc);
      const hair = s.ny > 0.25 && s.nz < 0.04;
      row.push(skin.add(s.x, s.y, s.z, s.nx, s.ny, s.nz, hair ? HAIR : SKIN, 0));
    }
    headRows.push(row);
  }
  for (let r = 0; r < headRows.length - 1; r++) skin.stitch(headRows[r], headRows[r + 1]);

  const sy0 = 0.5;
  const sy1 = 1.08;
  const sRings = Math.ceil((sy1 - sy0) / (CELL * 0.866));
  let sPrev: number[] | null = null;
  for (let r = 0; r <= sRings; r++) {
    const y = sy0 + ((sy1 - sy0) * r) / sRings;
    let row: number[];
    if (y < 0.98) {
      const left = cloth.ring(y, 0.125, 0.12, 0.115, CLOTH, 0.5, 18, (r % 2) * (Math.PI / 18));
      for (let i = 0; i < left.length; i++) {
        const id = left[i] * 3;
        cloth.p[id] -= 0.09;
      }
      const right = cloth.ring(y, 0.125, 0.12, 0.115, CLOTH, 0.5, 18, (r % 2) * (Math.PI / 18));
      for (let i = 0; i < right.length; i++) {
        const id = right[i] * 3;
        cloth.p[id] += 0.09;
      }
      row = left.concat(right);
      if (sPrev && sPrev.length === row.length) cloth.stitch(sPrev, row);
      sPrev = row;
    } else {
      const rad = torsoRadii(y);
      row = cloth.ring(y, rad.rx + 0.012, rad.zf + 0.01, rad.zb + 0.008, CLOTH, 0.3, 36, (r % 2) * (Math.PI / 36));
      if (sPrev && sPrev.length === row.length && y > 0.99) cloth.stitch(sPrev, row);
      sPrev = row;
    }
  }

  for (const sign of [-1, 1]) {
    cloth.tube([sign * 0.2, 0.6, 0.06], [sign * 0.2, 0.76, 0.07], () => 0.058, CLOTH_DARK, 0.2);
  }

  const g = new THREE.Group();
  g.add(skin.geo(0.48, 0.35));
  g.add(cloth.geo(0.9, 0.05));

  const white = new THREE.MeshStandardMaterial({ color: 0xe8e6e1, roughness: 0.25 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x14120f, roughness: 0.2 });
  const iris = new THREE.MeshStandardMaterial({ color: 0x3e4c3a, roughness: 0.25 });
  const lip = new THREE.MeshStandardMaterial({ color: 0xa86b5c, roughness: 0.4 });
  const skinMat = new THREE.MeshStandardMaterial({ color: SKIN, roughness: 0.5 });
  for (const sign of [-1, 1]) {
    const ball = new THREE.Mesh(new THREE.SphereGeometry(0.012, 16, 12), white);
    ball.position.set(hc[0] + sign * 0.032, hc[1] + 0.012, hc[2] + 0.09);
    const color = new THREE.Mesh(new THREE.SphereGeometry(0.006, 12, 8), iris);
    color.position.set(hc[0] + sign * 0.032, hc[1] + 0.012, hc[2] + 0.1);
    const pupil = new THREE.Mesh(new THREE.SphereGeometry(0.003, 8, 6), dark);
    pupil.position.set(hc[0] + sign * 0.032, hc[1] + 0.012, hc[2] + 0.105);
    g.add(ball, color, pupil);
    const ear = new THREE.Mesh(new THREE.SphereGeometry(0.018, 12, 10), skinMat);
    ear.scale.set(0.45, 1, 0.7);
    ear.position.set(hc[0] + sign * 0.095, hc[1] - 0.005, hc[2]);
    g.add(ear);
  }
  const mouth = new THREE.Mesh(new THREE.CapsuleGeometry(0.005, 0.02, 4, 8), lip);
  mouth.rotation.z = Math.PI / 2;
  mouth.position.set(hc[0], hc[1] - 0.04, hc[2] + 0.095);
  g.add(mouth);

  g.position.set(0, 0, -2.2);
  return g;
}
