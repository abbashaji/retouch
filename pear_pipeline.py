#!/usr/bin/env python3
"""
Marquise -> Pear stone retouch pipeline (CPU only, Windows/Mac/Linux).

Idea: never let the AI redraw the whole photo. We
  1) lock the 15 stone positions/sizes/angles (stones.json),
  2) send Gemini (in the browser) only a tight crop of the necklace,
  3) paste ONLY the stone regions of Gemini's answer back onto the ORIGINAL
     full-resolution photo, so face/skin/hand/chain stay pixel-identical,
  4) auto-check each stone and tell you which ones to regenerate.

Commands:
  python pear_pipeline.py detect   photo.jpg            # find stones, edit by click, write stones.json
  python pear_pipeline.py prepare  photo.jpg            # make crop + prompt for Gemini
  python pear_pipeline.py composite photo.jpg gemini_out.png [--out final.png]
"""
import argparse, json, os, sys
import cv2, numpy as np

N_STONES = 15

# ----------------------------------------------------------------- helpers
def read(path):
    im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)  # unicode-safe on Windows
    if im is None: sys.exit(f"cannot read {path}")
    return im

def write(path, im):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, im, [cv2.IMWRITE_JPEG_QUALITY, 97] if ext.lower() in (".jpg", ".jpeg") else [])
    buf.tofile(path)

def base(path): return os.path.splitext(path)[0]

def stone_score(im):
    """High where pixels are brighter and less yellow/pink than their surroundings (= diamond)."""
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    s = max(im.shape[:2]) * 0.03
    bgL = cv2.GaussianBlur(L, (0, 0), s); bgb = cv2.GaussianBlur(b, (0, 0), s)
    sc = (L - bgL) / 18.0 + (bgb - b) / 7.0
    gold = (b > 30) & (a > 0)
    sc[gold] = 0
    return cv2.GaussianBlur(sc, (0, 0), max(1.0, im.shape[1] * 0.0015))

# ----------------------------------------------------------------- 1. detect
def necklace_roi(im):
    h, w = im.shape[:2]
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    L, a, b = lab[..., 0], lab[..., 1].astype(int) - 128, lab[..., 2].astype(int) - 128
    gold = ((b > 34) & (L > 120) & (a > 2)).astype(np.uint8)
    gold = cv2.morphologyEx(gold, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(gold)
    if len(ys) < 50: return (0, 0, w, h)
    sel = ys > np.percentile(ys, 55)
    x1, x2 = np.percentile(xs[sel], [1, 99]); y1, y2 = np.percentile(ys[sel], [1, 99.7])
    pad = 0.05 * w
    return (int(max(0, x1 - pad)), int(max(0, y1 - pad * 1.2)), int(min(w, x2 + pad)), int(min(h, y2 + pad * 2)))

def auto_detect(im):
    h, w = im.shape[:2]
    x1, y1, x2, y2 = necklace_roi(im)
    sc = stone_score(im)[y1:y2, x1:x2]
    out = []
    for thr in (1.4, 1.1, 0.8):
        m = (sc > thr).astype(np.uint8) * 255
        k = max(3, int(w * 0.002)) | 1
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n, _, st, cen = cv2.connectedComponentsWithStats(m)
        areas = st[1:, 4]
        if len(areas) == 0: continue
        cand = [(cen[i][0] + x1, cen[i][1] + y1, st[i][4]) for i in range(1, n)
                if st[i][4] > w * w * 1.5e-5 and st[i][4] < w * w * 1.5e-3 and st[i][4] / max(1, st[i][2] * st[i][3]) > 0.4]
        out = cand
        if len(cand) >= N_STONES: break
    if len(out) < 4: return []
    pts = np.array([(c[0], c[1]) for c in out]); ar = np.array([c[2] for c in out])
    # keep points lying on one smooth arc (RANSAC quadratic) -> rejects finger/skin false positives
    best = None; rng = np.random.default_rng(0)
    for _ in range(400):
        idx = rng.choice(len(pts), 3, replace=False)
        try: p = np.polyfit(pts[idx, 0], pts[idx, 1], 2)
        except Exception: continue
        d = np.abs(np.polyval(p, pts[:, 0]) - pts[:, 1])
        inl = d < 0.022 * w
        if best is None or inl.sum() > best.sum(): best = inl
    pts, ar = pts[best], ar[best]
    keep = ar > 0.35 * np.median(ar)
    pts, ar = pts[keep], ar[keep]
    if len(pts) > N_STONES:                      # drop the weakest extras
        order = np.argsort(-ar)[:N_STONES]; pts = pts[order]
    return sorted([[float(x), float(y)] for x, y in pts])

def gui_edit(im, pts):
    """Left click = add stone, right click = remove nearest, Enter = save, Esc = abort."""
    h, w = im.shape[:2]
    x1, y1, x2, y2 = necklace_roi(im)
    cx1, cy1, cx2, cy2 = max(0, x1 - w // 20), max(0, y1 - w // 20), min(w, x2 + w // 20), min(h, y2 + w // 20)
    crop = im[cy1:cy2, cx1:cx2]
    zoom = min(1800 / crop.shape[1], 950 / crop.shape[0]); pts = [list(p) for p in pts]
    def draw():
        v = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
        for i, (x, y) in enumerate(sorted(pts)):
            p = (int((x - cx1) * zoom), int((y - cy1) * zoom))
            cv2.circle(v, p, 14, (0, 0, 255), 2); cv2.putText(v, str(i + 1), (p[0] - 8, p[1] - 18), 0, 0.6, (0, 0, 255), 2)
        cv2.putText(v, f"{len(pts)}/{N_STONES}  L-click add | R-click remove | Enter save | Esc abort", (10, 28), 0, 0.8, (0, 200, 0), 2)
        cv2.imshow("stones", v)
    def mouse(ev, mx, my, *_):
        X, Y = mx / zoom + cx1, my / zoom + cy1
        if ev == cv2.EVENT_LBUTTONDOWN: pts.append([X, Y])
        elif ev == cv2.EVENT_RBUTTONDOWN and pts:
            j = int(np.argmin([(p[0] - X) ** 2 + (p[1] - Y) ** 2 for p in pts])); pts.pop(j)
        draw()
    cv2.namedWindow("stones"); cv2.setMouseCallback("stones", mouse); draw()
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k == 27: cv2.destroyAllWindows(); sys.exit("aborted")
        if k in (13, 10) and len(pts) == N_STONES: break
        if k in (13, 10): print(f"need exactly {N_STONES} stones, have {len(pts)}")
    cv2.destroyAllWindows(); return sorted(pts)

# ----------------------------------------------------------------- 2. geometry
def segment_stone(im, sc, cx, cy, r):
    """Segment one stone near (cx,cy). Returns contour-based geometry dict."""
    h, w = im.shape[:2]
    x0, y0, x1, y1 = int(max(0, cx - r)), int(max(0, cy - r)), int(min(w, cx + r)), int(min(h, cy + r))
    win = sc[y0:y1, x0:x1]
    best = None
    for thr in (1.0, 0.7, 0.5, 0.35):
        m = (win > thr).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, lbl, st, cen = cv2.connectedComponentsWithStats(m)
        if n < 2: continue
        d = [(np.hypot(cen[i][0] + x0 - cx, cen[i][1] + y0 - cy), i) for i in range(1, n) if st[i][4] > 20]
        if not d: continue
        i = min(d)[1]
        if np.hypot(cen[i][0] + x0 - cx, cen[i][1] + y0 - cy) < r * 0.6:
            best = (lbl == i).astype(np.uint8); break
    if best is None:
        return None
    cnts, _ = cv2.findContours(best, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea) + np.array([x0, y0])
    (ex, ey), (d1, d2), ang = cv2.fitEllipse(c) if len(c) >= 5 else cv2.minAreaRect(c)
    # long axis direction (unit vector), pointing from tip toward the round end
    if d1 >= d2: L, W, th = d1, d2, np.deg2rad(ang) + np.pi / 2
    else:        L, W, th = d2, d1, np.deg2rad(ang)
    # fitEllipse angle: rotation of first axis from x-axis (clockwise in image coords)
    # robust direction from PCA
    pts = c.reshape(-1, 2).astype(np.float32)
    mu = pts.mean(0); _, _, vt = np.linalg.svd(pts - mu, full_matrices=False)
    ux, uy = vt[0]
    if uy < 0: ux, uy = -ux, -uy            # axis points DOWN: tip (prong side) is up, round end down
    proj = (pts - mu) @ np.array([ux, uy]); perp = (pts - mu) @ np.array([-uy, ux])
    # brightness threshold under-segments the facet edge a little -> compensate
    return dict(cx=float(mu[0]), cy=float(mu[1]), L=float(proj.max() - proj.min()) * 1.14,
                W=float(perp.max() - perp.min()) * 1.10, ux=float(ux), uy=float(uy),
                area=float(cv2.contourArea(c)))

def pear_polygon(g, ss=1.0):
    """Teardrop of same length/area class as the marquise: pointed end up (toward prong), round end down."""
    L = g["L"]; W = min(g["W"] * 1.12, L / 1.45)        # pear is a bit fuller than marquise
    R = W / 2
    u = np.array([g["ux"], g["uy"]]); v = np.array([-g["uy"], g["ux"]])
    c0 = np.array([g["cx"], g["cy"]])
    tip = c0 - u * L / 2
    ctr = c0 + u * (L / 2 - R)
    t = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    circ = ctr + np.outer(np.cos(t), u) * R + np.outer(np.sin(t), v) * R
    pts = np.vstack([circ, tip]).astype(np.float32)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    # slightly concave flanks near the tip -> real pear look
    return hull

def raster(poly, shape, grow=0):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(poly * 8).astype(np.int32)], 255, shift=3)
    if grow: m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1,) * 2))
    return m

def geometry(im, pts):
    sc = stone_score(im); w = im.shape[1]
    r = w * 0.028
    gs = []
    for i, (x, y) in enumerate(pts):
        g = segment_stone(im, sc, x, y, r)
        if g is None:
            sys.exit(f"stone {i+1}: could not segment near ({x:.0f},{y:.0f}); click it more precisely in 'detect'")
        gs.append(g)
    return gs

# ----------------------------------------------------------------- commands
def cmd_detect(a):
    im = read(a.photo); pts = auto_detect(im)
    print(f"auto-detected {len(pts)} stones")
    if not a.no_gui: pts = gui_edit(im, pts)
    elif len(pts) != N_STONES: sys.exit(f"need {N_STONES}, got {len(pts)}. Run without --no-gui and fix by clicking.")
    gs = geometry(im, pts)
    json.dump(dict(points=pts, stones=gs), open(base(a.photo) + "_stones.json", "w"), indent=1)
    vis = im.copy()
    for i, g in enumerate(gs):
        cv2.polylines(vis, [np.round(pear_polygon(g)).astype(np.int32)], True, (0, 255, 0), max(1, im.shape[1] // 900), cv2.LINE_AA)
    x1, y1, x2, y2 = necklace_roi(im); write(base(a.photo) + "_preview_targets.jpg", vis[y1:y2, x1:x2])
    print("wrote", base(a.photo) + "_stones.json", "and *_preview_targets.jpg (green = target pear outlines)")

def load(a):
    return json.load(open(base(a.photo) + "_stones.json"))["stones"]

def crop_box(im, gs):
    h, w = im.shape[:2]
    xs = [g["cx"] for g in gs]; ys = [g["cy"] for g in gs]
    m = max(g["L"] for g in gs) * 3.0
    x1, x2, y1, y2 = min(xs) - m, max(xs) + m, min(ys) - m * 1.6, max(ys) + m * 1.6
    s = max(x2 - x1, y2 - y1) * 1.0
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    X1 = int(max(0, min(w - s, cx - s / 2))); Y1 = int(max(0, min(h - s, cy - s / 2)))
    s = int(min(s, w - X1, h - Y1)); return X1, Y1, s

PROMPT = """Edit this photo. Change ONLY the shape of the 15 dangling diamonds, from marquise cut to pear cut (teardrop: one wide rounded end facing DOWN, one pointed end facing UP toward the gold chain).
Strict rules:
- Exactly 15 stones, same positions. Do not add, remove, merge or move any stone.
- Each stone keeps the same length, same tilt and same relative size as in the original (large/small variation stays).
- Keep the thin delicate gold prong and the gold ball chain exactly as they are.
- Same crisp facets and sparkle style, adapted to pear shape. No halo, no extra gold bezel.
- Do not change skin, freckles, hand, clothing, lighting, framing or colors. Output at the same size and framing as the input."""

def cmd_prepare(a):
    im = read(a.photo); gs = load(a); X, Y, s = crop_box(im, gs)
    crop = im[Y:Y + s, X:X + s]
    up = cv2.resize(crop, (2048, 2048), interpolation=cv2.INTER_CUBIC) if s < 2048 else crop
    write(base(a.photo) + "_for_gemini.png", up)
    json.dump(dict(X=X, Y=Y, s=s), open(base(a.photo) + "_crop.json", "w"))
    open(base(a.photo) + "_prompt.txt", "w", encoding="utf-8").write(PROMPT)
    print(f"upload {base(a.photo)}_for_gemini.png to Gemini with the text in _prompt.txt; generate 4-6 times, save each as *_out1.png, *_out2.png ...")

def gemini_stones(g, s_ref):
    """Find the 15 stones in a Gemini output (already resized to crop size). Returns list of dicts with mask+geometry."""
    lab = cv2.cvtColor(g, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    bg = cv2.GaussianBlur(L, (0, 0), s_ref * 0.03)
    gold = cv2.dilate(((b > 22) & (a > -2)).astype(np.uint8), np.ones((3, 3), np.uint8))
    m = ((L > 170) & ((L - bg) > 8) & (b < 14) & (gold == 0)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lbl, st, cen = cv2.connectedComponentsWithStats(m)
    cand = []
    for i in range(1, n):
        x, y, w, h, ar = st[i]
        if ar < s_ref ** 2 * 6e-4 or ar > s_ref ** 2 * 8e-3 or h < w * 0.85 or ar / (w * h) < 0.45 or h > s_ref * 0.14: continue
        cand.append((cen[i][0], cen[i][1], ar, i))
    if len(cand) < 8: return [], lbl
    pts = np.array([(c[0], c[1]) for c in cand]); best = None; rng = np.random.default_rng(1)
    for _ in range(500):
        idx = rng.choice(len(pts), 3, replace=False)
        try: p = np.polyfit(pts[idx, 0], pts[idx, 1], 2)
        except Exception: continue
        inl = np.abs(np.polyval(p, pts[:, 0]) - pts[:, 1]) < 0.05 * s_ref
        if best is None or inl.sum() > best.sum(): best = inl
    cand = [c for c, k in zip(cand, best) if k]
    cand = sorted(sorted(cand, key=lambda c: -c[2])[:N_STONES], key=lambda c: c[0])
    out = []
    for cx, cy, ar, i in cand:
        mk = (lbl == i).astype(np.uint8); ar_raw = float(ar)
        cnts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea); full = np.zeros_like(mk); cv2.drawContours(full, [c], -1, 1, -1)
        # remove glare tendrils, then use the clean convex outline (diamonds are convex)
        kk = max(5, int(np.sqrt(ar) * 0.18)) | 1
        op = cv2.morphologyEx(full, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
        if op.sum() > 0.5 * full.sum():
            cc, _ = cv2.findContours(op, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); c = max(cc, key=cv2.contourArea)
        hull_c = cv2.convexHull(c); full = np.zeros_like(mk); cv2.drawContours(full, [hull_c], -1, 1, -1); c = hull_c
        pts_ = c.reshape(-1, 2).astype(np.float32); mu = pts_.mean(0)
        _, _, vt = np.linalg.svd(pts_ - mu, full_matrices=False); ux, uy = vt[0]
        if uy < 0: ux, uy = -ux, -uy
        pr = (pts_ - mu) @ np.array([ux, uy]); pp = (pts_ - mu) @ np.array([-uy, ux])
        ys, xs = np.nonzero(full); q = (np.stack([xs, ys], 1) - mu) @ np.array([ux, uy]); qp = np.abs((np.stack([xs, ys], 1) - mu) @ np.array([-uy, ux]))
        L_ = pr.max() - pr.min(); t = (q - pr.min()) / max(1e-6, L_)
        wt = qp[(t > 0.1) & (t < 0.4)].mean() if ((t > 0.1) & (t < 0.4)).any() else 1
        wb = qp[(t > 0.6) & (t < 0.9)].mean() if ((t > 0.6) & (t < 0.9)).any() else 1
        sol = float(ar_raw / max(cv2.contourArea(c), 1))
        out.append(dict(solidity=sol, mask=full, cx=float(mu[0]), cy=float(mu[1]), L=float(L_), W=float(pp.max() - pp.min()),
                        ux=float(ux), uy=float(uy), pearness=float(wb / max(wt, 1e-3))))
    return out, lbl

def cmd_composite(a):
    im = read(a.photo); gs = load(a); cb = json.load(open(base(a.photo) + "_crop.json"))
    X, Y, s = cb["X"], cb["Y"], cb["s"]; crop = im[Y:Y + s, X:X + s].copy()
    tg = [dict(g, cx=g["cx"] - X, cy=g["cy"] - Y) for g in gs]
    # ---- 1. find stones in every Gemini output
    cands = []   # per output: (image, stones)
    for f in a.gemini:
        g = cv2.resize(read(f), (s, s), interpolation=cv2.INTER_CUBIC)
        st, _ = gemini_stones(g, s)
        print(f"{os.path.basename(f)}: found {len(st)} stones" + ("" if len(st) == N_STONES else "  -> SKIPPED (need exactly 15; regenerate it)"))
        if len(st) == N_STONES: cands.append((f, g, st))
    if not cands: sys.exit("no usable Gemini output")
    # ---- 2. clear the old marquise stones (inpaint), then transplant best candidate per stone
    erase = np.zeros((s, s), np.uint8)
    for g in tg:
        ang = np.rad2deg(np.arctan2(-g["ux"], g["uy"]))
        cv2.ellipse(erase, (int(g["cx"]), int(g["cy"])), (int(g["W"] / 2 * 1.12) + 3, int(g["L"] / 2 * 1.08) + 3), ang, 0, 360, 255, -1)
    base_img = cv2.inpaint(crop, erase, 4, cv2.INPAINT_TELEA)
    out = base_img.astype(np.float32); report = []
    for i, go in enumerate(tg):
        best = None
        for f, g, st in cands:
            gg = st[i]; k = go["L"] / gg["L"]
            pen = abs(np.log(k)) * 1.0 + max(0, 1.25 - gg["pearness"]) * 2.0 + abs(gg["W"] / gg["L"] - 0.62) * 1.0 + (6 if gg["solidity"] < 0.78 else 0) + (6 if not (0.7 < k < 1.4) else 0) + (4 if not (0.45 < gg["W"] / gg["L"] < 0.85) else 0)
            if best is None or pen < best[0]: best = (pen, f, g, gg, k)
        pen, f, g, gg, k = best
        th = np.arctan2(go["uy"], go["ux"]) - np.arctan2(gg["uy"], gg["ux"])
        c, sn = np.cos(th) * k, np.sin(th) * k
        M = np.array([[c, -sn, 0], [sn, c, 0]], np.float32)
        M[:, 2] = np.array([go["cx"], go["cy"]]) - M[:, :2] @ np.array([gg["cx"], gg["cy"]])
        grow = max(2, int(gg["W"] * 0.07))
        mk = cv2.dilate(gg["mask"], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1,) * 2))
        wm = cv2.warpAffine(mk.astype(np.float32), M, (s, s), flags=cv2.INTER_LINEAR)
        wi = cv2.warpAffine(g, M, (s, s), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        alpha = cv2.GaussianBlur(wm, (0, 0), max(1.0, grow * 0.5)); alpha = np.clip(alpha * 1.1, 0, 1)
        # local SKIN colour match (stones themselves and gold are left untouched)
        wmu = (wm > 0.5).astype(np.uint8)
        a1 = cv2.cvtColor(wi, cv2.COLOR_BGR2LAB).astype(np.float32); a0 = cv2.cvtColor(base_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        goldm = ((a1[..., 2] - 128 > 20) | (a0[..., 2] - 128 > 24))
        ring = (cv2.dilate(wmu, np.ones((41, 41), np.uint8)) > 0) & (cv2.dilate(wmu, np.ones((11, 11), np.uint8)) == 0) & ~goldm
        if ring.sum() > 80:
            off = np.array([np.median(a0[..., ch][ring]) - np.median(a1[..., ch][ring]) for ch in range(3)], np.float32)
            stonew = cv2.GaussianBlur(cv2.warpAffine(gg["mask"].astype(np.float32), M, (s, s)), (0, 0), 1.5)
            skinw = np.clip(1 - stonew * 1.5, 0, 1) * (~(a1[..., 2] - 128 > 20))
            wi = cv2.cvtColor(np.clip(a1 + off * skinw[..., None], 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        out = out * (1 - alpha[..., None]) + wi.astype(np.float32) * alpha[..., None]
        report.append((i + 1, os.path.basename(f), gg["pearness"], k, pen))
    final_crop = np.clip(out, 0, 255).astype(np.uint8)
    final = im.copy(); final[Y:Y + s, X:X + s] = final_crop
    outp = a.out or base(a.photo) + "_final.png"; write(outp, final)
    print(f"saved {outp}")
    bad = []
    for i, fn, pr, k, pen in report:
        flag = pr < 1.12 or not (0.75 < k < 1.3) or pen > 0.9
        if flag: bad.append(i)
        print(f"  stone {i:2d}: from {fn:14s} pear-ness {pr:4.2f} size x{k:4.2f}" + ("   <-- check" if flag else ""))
    print("all stones look OK (still zoom in and check)" if not bad else f"check / regenerate / fix by hand: stones {bad}")
    ys0, xs0 = [g["cy"] for g in tg], [g["cx"] for g in tg]
    m = int(max(g["L"] for g in tg) * 2.2)
    bx = (int(max(0, min(xs0) - m)), int(max(0, min(ys0) - m)), int(min(s, max(xs0) + m)), int(min(s, max(ys0) + m * 2)))
    sheet = np.hstack([crop[bx[1]:bx[3], bx[0]:bx[2]], final_crop[bx[1]:bx[3], bx[0]:bx[2]]])
    write(base(outp) + "_compare.jpg", sheet)

# ================================================================= TILE MODE
import glob

TILE_PROMPT = """Edit this photo. It shows {n} dangling diamonds on a gold ball chain. Change ONLY the shape of these {n} diamonds from marquise cut to pear cut (teardrop: wide rounded end facing DOWN, single pointed end facing UP toward the chain).
Strict rules:
- Exactly {n} stones, same positions, same size, same tilt, same relative size differences. Do not add, remove, merge or move any stone.
- Keep the thin gold prong at the top of each stone and the gold ball chain exactly as they are.
- Photorealistic diamond: crisp facets, sharp bright reflections, same sparkle style. No blur, no glow, no halo, no extra bezel.
- Do not change skin, freckles, hand, clothing, lighting, framing, zoom or colors. Output the same framing as the input."""

def tile_plan(n, per=4, stride=3):
    tiles, i = [], 0
    while True:
        tiles.append(list(range(i, min(n, i + per))))
        if i + per >= n: break
        i += stride
    return tiles

def tile_box(im, gs, idx):
    h, w = im.shape[:2]; sub = [gs[i] for i in idx]
    m = max(g["L"] for g in sub) * 1.3
    x1 = min(g["cx"] for g in sub) - m; x2 = max(g["cx"] for g in sub) + m
    y1 = min(g["cy"] for g in sub) - m * 1.25; y2 = max(g["cy"] for g in sub) + m * 1.25
    side = max(x2 - x1, y2 - y1); cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = min(side, w, h)
    X = int(max(0, min(w - side, cx - side / 2))); Y = int(max(0, min(h - side, cy - side / 2)))
    return X, Y, int(side)

def cmd_prepare_tiles(a):
    im = read(a.photo); gs = load(a); b = base(a.photo); tiles = []
    for k, idx in enumerate(tile_plan(len(gs), a.per, a.per - 1), 1):
        X, Y, s = tile_box(im, gs, idx)
        tile = cv2.resize(im[Y:Y + s, X:X + s], (1024, 1024), interpolation=cv2.INTER_CUBIC)
        write(f"{b}_tile{k}_for_gemini.png", tile)
        open(f"{b}_tile{k}_prompt.txt", "w", encoding="utf-8").write(TILE_PROMPT.format(n=len(idx)))
        tiles.append(dict(k=k, idx=idx, X=X, Y=Y, s=s))
        print(f"tile {k}: stones {idx[0]+1}-{idx[-1]+1} ({len(idx)} stones), crop {s}px -> upload {os.path.basename(b)}_tile{k}_for_gemini.png")
    json.dump(tiles, open(b + "_tiles.json", "w"))
    print(f"\nFor each tile generate 4-6 results and save as {os.path.basename(b)}_tile<k>_out1.jpg, _out2.jpg ... (any image format).")

def tile_candidates(g, exp_area, exp_L):
    lab = cv2.cvtColor(g, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a_, b_ = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    bg = cv2.GaussianBlur(L, (0, 0), exp_L * 1.6)
    gold = cv2.dilate(((b_ > 22) & (a_ > -2)).astype(np.uint8), np.ones((3, 3), np.uint8))
    m = ((L > 165) & ((L - bg) > 8) & (b_ < 15) & (gold == 0)).astype(np.uint8) * 255
    k1 = max(3, int(exp_L * 0.07)) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1)))
    n, lbl, st, cen = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        x, y, w, h, ar = st[i]
        if not (0.3 * exp_area < ar < 3.5 * exp_area) or h < w * 0.8 or ar / (w * h) < 0.45: continue
        mk = (lbl == i).astype(np.uint8)
        cnts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); c = max(cnts, key=cv2.contourArea)
        full = np.zeros_like(mk); cv2.drawContours(full, [c], -1, 1, -1)
        kk = max(5, int(np.sqrt(ar) * 0.18)) | 1
        op = cv2.morphologyEx(full, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
        if op.sum() > 0.5 * full.sum():
            cc, _ = cv2.findContours(op, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); c = max(cc, key=cv2.contourArea)
        hc = cv2.convexHull(c); full = np.zeros_like(mk); cv2.drawContours(full, [hc], -1, 1, -1)
        pts_ = hc.reshape(-1, 2).astype(np.float32); mu = pts_.mean(0)
        _, _, vt = np.linalg.svd(pts_ - mu, full_matrices=False); ux, uy = vt[0]
        if uy < 0: ux, uy = -ux, -uy
        pr = (pts_ - mu) @ np.array([ux, uy]); pp = (pts_ - mu) @ np.array([-uy, ux])
        ys, xs = np.nonzero(full); xy = np.stack([xs, ys], 1) - mu
        q = xy @ np.array([ux, uy]); qp = np.abs(xy @ np.array([-uy, ux]))
        L_ = pr.max() - pr.min(); t = (q - pr.min()) / max(1e-6, L_)
        wt = qp[(t > 0.1) & (t < 0.4)].mean() if ((t > 0.1) & (t < 0.4)).any() else 1
        wb = qp[(t > 0.6) & (t < 0.9)].mean() if ((t > 0.6) & (t < 0.9)).any() else 1
        out.append(dict(mask=full, cx=float(mu[0]), cy=float(mu[1]), L=float(L_), W=float(pp.max() - pp.min()),
                        ux=float(ux), uy=float(uy), pearness=float(wb / max(wt, 1e-3)),
                        solidity=float(ar / max(cv2.contourArea(hc), 1))))
    return out

def match_tile(cands, targets):
    """targets/cands: positions in the same tile frame (Gemini frame may be shifted/zoomed a bit). Returns {target_idx: cand}."""
    if not cands: return {}
    T = np.array([[t["cx"], t["cy"]] for t in targets]); C = np.array([[c["cx"], c["cy"]] for c in cands])
    Lm = np.median([t["L"] for t in targets]); A = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
    for rad, mode in ((2.5, "shift"), (1.2, "sim"), (0.8, "sim")):
        P_ = T @ A[:, :2].T + A[:, 2]
        d = np.linalg.norm(P_[:, None] - C[None], axis=2); j = d.argmin(1); ok = d[np.arange(len(T)), j] < rad * Lm
        if ok.sum() == 0: break
        if mode == "shift" or ok.sum() < 3:
            A[:, 2] += np.median(C[j[ok]] - P_[ok], axis=0)
        else:
            M, _ = cv2.estimateAffinePartial2D(T[ok].astype(np.float32), C[j[ok]].astype(np.float32), method=cv2.RANSAC, ransacReprojThreshold=0.3 * Lm)
            if M is not None: A = M.astype(np.float32)
    P_ = T @ A[:, :2].T + A[:, 2]
    pairs = sorted([(np.linalg.norm(P_[i] - C[j]), i, j) for i in range(len(T)) for j in range(len(C))])
    used_t, used_c, res = set(), set(), {}
    for dist, i, j in pairs:
        if dist > 0.7 * Lm: break
        if i in used_t or j in used_c: continue
        used_t.add(i); used_c.add(j); res[i] = cands[j]
    return res

def cmd_composite_tiles(a):
    im = read(a.photo); gs = load(a); b = base(a.photo); tiles = json.load(open(b + "_tiles.json"))
    H, W_ = im.shape[:2]; best = {}   # stone idx -> (pen, img, gg, M, name)
    exp_L = float(np.median([g["L"] for g in gs])) / 1.14
    exp_area = 0.62 * float(np.median([(g["L"] / 1.14) * (g["W"] / 1.10) for g in gs]))
    for t in tiles:
        files = sorted(f for f in glob.glob(f"{b}_tile{t['k']}_out*.*") if not f.endswith(".json"))
        if not files: print(f"tile {t['k']}: no outputs found (expected {os.path.basename(b)}_tile{t['k']}_out1.jpg ...)"); continue
        tg = [dict(gs[i], cx=gs[i]["cx"] - t["X"], cy=gs[i]["cy"] - t["Y"]) for i in t["idx"]]
        for f in files:
            g = cv2.resize(read(f), (t["s"], t["s"]), interpolation=cv2.INTER_AREA)   # downscale = sharper stones
            cands = tile_candidates(g, exp_area, exp_L); mt = match_tile(cands, tg)
            print(f"tile {t['k']} {os.path.basename(f)}: {len(cands)} stone candidates, matched {len(mt)}/{len(tg)}")
            for ti, gg in mt.items():
                gi = t["idx"][ti]; go = gs[gi]; k = go["L"] / gg["L"]
                pen = abs(np.log(k)) + max(0, 1.25 - gg["pearness"]) * 2 + abs(gg["W"] / gg["L"] - 0.62) \
                      + (6 if gg["solidity"] < 0.78 else 0) + (6 if not (0.7 < k < 1.4) else 0) + (4 if not (0.45 < gg["W"] / gg["L"] < 0.85) else 0)
                th = np.arctan2(go["uy"], go["ux"]) - np.arctan2(gg["uy"], gg["ux"])
                c_, s_ = np.cos(th) * k, np.sin(th) * k
                M = np.array([[c_, -s_, 0], [s_, c_, 0]], np.float32)
                M[:, 2] = np.array([go["cx"], go["cy"]]) - M[:, :2] @ np.array([gg["cx"], gg["cy"]])  # gg coords are tile-local; M maps them onto the full photo
                if gi not in best or pen < best[gi][0]: best[gi] = (pen, g, gg, M, os.path.basename(f), t["X"], t["Y"])
    if not best: sys.exit("nothing matched")
    erase = np.zeros((H, W_), np.uint8)
    for gi in best:
        g = gs[gi]; ang = np.rad2deg(np.arctan2(-g["ux"], g["uy"]))
        cv2.ellipse(erase, (int(g["cx"]), int(g["cy"])), (int(g["W"] / 2 * 1.12) + 3, int(g["L"] / 2 * 1.08) + 3), ang, 0, 360, 255, -1)
    base_img = cv2.inpaint(im, erase, 4, cv2.INPAINT_TELEA); out = base_img.astype(np.float32); report = []
    for gi in sorted(best):
        pen, g, gg, M, name, X, Y = best[gi]
        grow = max(2, int(gg["W"] * 0.07))
        mk = cv2.dilate(gg["mask"], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1,) * 2))
        wm = cv2.warpAffine(mk.astype(np.float32), M, (W_, H), flags=cv2.INTER_LINEAR)
        wi = cv2.warpAffine(g, M, (W_, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        alpha = np.clip(cv2.GaussianBlur(wm, (0, 0), max(1.0, grow * 0.5)) * 1.1, 0, 1)
        wmu = (wm > 0.5).astype(np.uint8)
        a1 = cv2.cvtColor(wi, cv2.COLOR_BGR2LAB).astype(np.float32); a0 = cv2.cvtColor(base_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        goldm = ((a1[..., 2] - 128 > 20) | (a0[..., 2] - 128 > 24))
        ring = (cv2.dilate(wmu, np.ones((41, 41), np.uint8)) > 0) & (cv2.dilate(wmu, np.ones((11, 11), np.uint8)) == 0) & ~goldm
        if ring.sum() > 80:
            off = np.array([np.median(a0[..., ch][ring]) - np.median(a1[..., ch][ring]) for ch in range(3)], np.float32)
            stonew = cv2.GaussianBlur(cv2.warpAffine(gg["mask"].astype(np.float32), M, (W_, H)), (0, 0), 1.5)
            skinw = np.clip(1 - stonew * 1.5, 0, 1) * (~(a1[..., 2] - 128 > 20))
            wi = cv2.cvtColor(np.clip(a1 + off * skinw[..., None], 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        out = out * (1 - alpha[..., None]) + wi.astype(np.float32) * alpha[..., None]
        report.append((gi + 1, name, gg["pearness"], gs[gi]["L"] / gg["L"], pen))
    final = np.clip(out, 0, 255).astype(np.uint8); outp = a.out or b + "_final_tiles.png"; write(outp, final)
    print(f"saved {outp}"); bad = [gi + 1 for gi in range(len(gs)) if gi not in best]
    for i, fn, pr, k, pen in report:
        flag = pr < 1.12 or not (0.75 < k < 1.3) or pen > 0.9
        if flag: bad.append(i)
        print(f"  stone {i:2d}: from {fn:22s} pear-ness {pr:4.2f} size x{k:4.2f}" + ("   <-- check" if flag else ""))
    miss = [gi + 1 for gi in range(len(gs)) if gi not in best]
    if miss: print("NO usable candidate (still marquise!):", miss)
    print("check / regenerate / fix by hand: stones", sorted(set(bad)) if bad else "none")
    xs = [g["cx"] for g in gs]; ys = [g["cy"] for g in gs]; m = int(max(g["L"] for g in gs) * 1.5)
    bx = (int(max(0, min(xs) - m)), int(max(0, min(ys) - m)), int(min(W_, max(xs) + m)), int(min(H, max(ys) + m * 2)))
    write(base(outp) + "_compare.jpg", np.hstack([im[bx[1]:bx[3], bx[0]:bx[2]], final[bx[1]:bx[3], bx[0]:bx[2]]]))

if __name__ == "__main__":
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("detect"); d.add_argument("photo"); d.add_argument("--no-gui", action="store_true"); d.set_defaults(f=cmd_detect)
    r = sp.add_parser("prepare"); r.add_argument("photo"); r.set_defaults(f=cmd_prepare)
    c = sp.add_parser("composite"); c.add_argument("photo"); c.add_argument("gemini", nargs="+"); c.add_argument("--out"); c.set_defaults(f=cmd_composite)
    t = sp.add_parser("prepare-tiles"); t.add_argument("photo"); t.add_argument("--per", type=int, default=4); t.set_defaults(f=cmd_prepare_tiles)
    ct = sp.add_parser("composite-tiles"); ct.add_argument("photo"); ct.add_argument("--out"); ct.set_defaults(f=cmd_composite_tiles)
    a = p.parse_args(); a.f(a)
