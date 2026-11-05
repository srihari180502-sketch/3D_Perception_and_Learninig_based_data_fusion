#!/usr/bin/env python

import argparse
import json
import pickle
import random
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from joblib import Parallel, delayed, dump, load

import matplotlib
matplotlib.use("Agg")  # save figures to disk; no GUI needed
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.cluster import DBSCAN
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------
CLASSES = ["Car", "Van", "Truck", "Pedestrian", "Cyclist"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
BG = len(CLASSES)   # background label used by the LiDAR / fusion classifiers
IGNORE = -1         # cluster sits on an object type we don't model (Tram, Misc, ...)

CELL = 8
HOG_KW = dict(orientations=9, pixels_per_cell=(CELL, CELL),
              cells_per_block=(2, 2), block_norm="L2-Hys")

# Canonical HOG window per class: (width_px, height_px), multiples of CELL.
WINDOWS = {
    "Car": (64, 40),
    "Van": (64, 48),
    "Truck": (72, 56),
    "Pedestrian": (24, 64),
    "Cyclist": (40, 64),
}
# Image pyramid factors. A class window of height h px then matches objects of height h/scale px.
SCALES = [1.6, 1.15, 0.8, 0.57, 0.4, 0.28, 0.2, 0.14]
ROI_TOP = 100  # rows above this are sky; skipped by the sliding window to save time
IMG_W, IMG_H = 1242, 375   # nominal KITTI image size (used to normalise / clip refined boxes)

# KITTI difficulty: (min box height px, max occlusion level, max truncation)
DIFFICULTY = {"easy": (40, 0, 0.15), "moderate": (25, 1, 0.30), "hard": (25, 2, 0.50)}

COLORS = {"Car": "red", "Van": "orange", "Truck": "magenta",
          "Pedestrian": "cyan", "Cyclist": "yellow"}
METHODS = ["Camera-only", "LiDAR-only", "Fusion"]


# ----------------------------------------------------------------------------------------
# KITTI I/O
# ----------------------------------------------------------------------------------------
def read_labels(path):
    objs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) < 15:
            continue
        objs.append(dict(
            type=f[0], trunc=float(f[1]), occ=int(float(f[2])),
            box=np.array(f[4:8], dtype=np.float64),
            hwl=np.array(f[8:11], dtype=np.float64),     # height, width, length (m)
            loc=np.array(f[11:14], dtype=np.float64),    # bottom-centre in rectified camera coords
            ry=float(f[14]),
        ))
    return objs


def read_calib(path):
    d = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            d[k] = np.array([float(x) for x in v.split()])
    P2 = d["P2"].reshape(3, 4)
    R0 = np.eye(4)
    R0[:3, :3] = d["R0_rect"].reshape(3, 3)
    Tr = np.eye(4)
    Tr[:3, :4] = d["Tr_velo_to_cam"].reshape(3, 4)
    return P2, R0, Tr


def load_rgb(root, fid):
    return np.asarray(Image.open(Path(root) / "image_2" / f"{fid}.png").convert("RGB"))


def load_gray(root, fid):
    return np.asarray(Image.open(Path(root) / "image_2" / f"{fid}.png").convert("L"))


# ----------------------------------------------------------------------------------------
# Box utilities
# ----------------------------------------------------------------------------------------
def iou_many(box, boxes):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(boxes) == 0:
        return np.zeros(0)
    ix0 = np.maximum(box[0], boxes[:, 0])
    iy0 = np.maximum(box[1], boxes[:, 1])
    ix1 = np.minimum(box[2], boxes[:, 2])
    iy1 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a + b - inter + 1e-9)


def overlap_ratio(box, boxes):
    """Intersection area / area of `box` (how much of `box` lies inside each of `boxes`)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(boxes) == 0:
        return np.zeros(0)
    ix0 = np.maximum(box[0], boxes[:, 0])
    iy0 = np.maximum(box[1], boxes[:, 1])
    ix1 = np.minimum(box[2], boxes[:, 2])
    iy1 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    return inter / ((box[2] - box[0]) * (box[3] - box[1]) + 1e-9)


def nms(boxes, scores, thr):
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_many(boxes[i], boxes[order[1:]])
        order = order[1:][ious < thr]
    return np.asarray(keep, dtype=int)


def empty_dets():
    return {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, dtype=int)}


def merge_dets(parts):
    if not parts:
        return empty_dets()
    return {"boxes": np.concatenate([p[0] for p in parts]),
            "scores": np.concatenate([p[1] for p in parts]),
            "labels": np.concatenate([p[2] for p in parts]).astype(int)}


# ----------------------------------------------------------------------------------------
# Camera: HOG + linear SVM sliding window
# ----------------------------------------------------------------------------------------
def blocks_for(cls):
    w, h = WINDOWS[cls]
    return h // CELL - 1, w // CELL - 1


def crop_feature(g, box, cls, flip=False):
    """HOG feature of `box` resized to the class window. None if the crop is degenerate."""
    ih, iw = g.shape
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(iw, x1), min(ih, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    patch = g[y0:y1, x0:x1]
    if flip:
        patch = patch[:, ::-1]
    w, h = WINDOWS[cls]
    patch = Image.fromarray(np.ascontiguousarray(patch)).resize((w, h), Image.BILINEAR)
    return hog(np.asarray(patch), **HOG_KW).astype(np.float32)


def score_map(B, W, b):
    """Linear SVM score of every window position, via per-offset correlation of HOG blocks."""
    R, C, _ = B.shape
    br, bc, _ = W.shape
    Rp, Cp = R - br + 1, C - bc + 1
    if Rp <= 0 or Cp <= 0:
        return None
    S = np.full((Rp, Cp), b, dtype=np.float32)
    for i in range(br):
        for j in range(bc):
            S += B[i:i + Rp, j:j + Cp] @ W[i, j]
    return S


def camera_windows(g, model, thresh, per_map=60, want_feats=False):
    """Raw sliding-window detections per class: {class: (boxes, scores, feats_or_None)}."""
    ih, iw = g.shape
    roi = g[ROI_TOP:]
    acc = {c: ([], [], []) for c in CLASSES}
    for s in SCALES:
        h, w = int(roi.shape[0] * s), int(iw * s)
        if h < 2 * CELL or w < 2 * CELL:
            continue
        im = np.asarray(Image.fromarray(np.ascontiguousarray(roi)).resize((w, h), Image.BILINEAR))
        B = hog(im, feature_vector=False, **HOG_KW)
        B = B.reshape(B.shape[0], B.shape[1], -1).astype(np.float32)
        for c in CLASSES:
            Wt, b = model[c]
            S = score_map(B, Wt, b)
            if S is None:
                continue
            idx = np.argwhere(S > thresh)
            if len(idx) == 0:
                continue
            sc = S[idx[:, 0], idx[:, 1]]
            top = np.argsort(-sc)[:per_map]
            idx, sc = idx[top], sc[top]
            wpx, hpx = WINDOWS[c]
            x0 = idx[:, 1] * CELL / s
            y0 = idx[:, 0] * CELL / s + ROI_TOP
            acc[c][0].append(np.stack([x0, y0, x0 + wpx / s, y0 + hpx / s], axis=1))
            acc[c][1].append(sc)
            if want_feats:
                br, bc = Wt.shape[:2]
                acc[c][2].append(np.stack([B[r:r + br, q:q + bc].ravel() for r, q in idx]))
    out = {}
    for c in CLASSES:
        if acc[c][0]:
            feats = np.concatenate(acc[c][2]) if want_feats else None
            out[c] = (np.concatenate(acc[c][0]), np.concatenate(acc[c][1]), feats)
        else:
            out[c] = (np.zeros((0, 4)), np.zeros(0), None)
    return out


def encode_box(box, gt):
    """Regression target: how far `box` must move to become `gt`, in units of box width / height."""
    w, h = box[2] - box[0], box[3] - box[1]
    return (np.asarray(gt) - np.asarray(box)) / np.array([w, h, w, h])


def decode_boxes(boxes, t):
    boxes = np.asarray(boxes, dtype=np.float64)
    w = (boxes[:, 2] - boxes[:, 0])[:, None]
    h = (boxes[:, 3] - boxes[:, 1])[:, None]
    out = boxes + t * np.hstack([w, h, w, h])
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, IMG_W - 1)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, IMG_H - 1)
    bad = (out[:, 2] - out[:, 0] < 4) | (out[:, 3] - out[:, 1] < 4)
    out[bad] = boxes[bad]
    return out


def jitter_box(box, rng, iw, ih):
    """Random shift / rescale of a ground-truth box that still overlaps it with IoU >= 0.4."""
    w, h = box[2] - box[0], box[3] - box[1]
    for _ in range(10):
        cx = (box[0] + box[2]) / 2 + rng.uniform(-0.12, 0.12) * w
        cy = (box[1] + box[3]) / 2 + rng.uniform(-0.12, 0.12) * h
        nw, nh = w * rng.uniform(0.8, 1.25), h * rng.uniform(0.8, 1.25)
        nb = np.array([cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2])
        nb[[0, 2]] = np.clip(nb[[0, 2]], 0, iw - 1)
        nb[[1, 3]] = np.clip(nb[[1, 3]], 0, ih - 1)
        if nb[2] - nb[0] > 4 and nb[3] - nb[1] > 4 and iou_many(box, nb[None])[0] >= 0.4:
            return nb
    return None


def fit_box_regressors(reg_x, reg_t):
    reg = {}
    for c in CLASSES:
        if len(reg_x[c]) < 50:
            reg[c] = None
            continue
        r = Ridge(alpha=10.0).fit(np.vstack(reg_x[c]).astype(np.float32), np.vstack(reg_t[c]))
        reg[c] = (r.coef_.astype(np.float32), r.intercept_.astype(np.float32))
    return reg


def refine_camera_boxes(g, boxes, cls, model):
    """Ridge regression from the HOG feature of a detection to a corrected box."""
    reg = model.get("reg", {}).get(cls) if isinstance(model.get("reg"), dict) else None
    if reg is None or len(boxes) == 0:
        return boxes
    coef, icpt = reg
    t = np.zeros((len(boxes), 4))
    for i, b in enumerate(boxes):
        f = crop_feature(g, b, cls)
        if f is not None:
            t[i] = coef @ f + icpt
    return decode_boxes(boxes, np.clip(t, -0.6, 0.6))


def camera_detect(g, model, thresh=-0.5, nms_thr=0.4, max_per_class=40):
    raw = camera_windows(g, model, thresh)
    parts = []
    for ci, c in enumerate(CLASSES):
        boxes, scores, _ = raw[c]
        if len(boxes) == 0:
            continue
        keep = nms(boxes, scores, nms_thr)[:max_per_class]
        boxes, scores = boxes[keep], scores[keep]
        boxes = refine_camera_boxes(g, boxes, c, model)       # no-op if the model has no regressors
        keep = nms(boxes, scores, 0.5)                        # refined duplicates may now coincide
        parts.append((boxes[keep], scores[keep], np.full(len(keep), ci)))
    return merge_dets(parts)


def camera_box_scores(g, boxes, model):
    """SVM score of every class on every box (used as camera features for fusion)."""
    out = np.full((len(boxes), len(CLASSES)), -2.0, dtype=np.float32)
    for i, b in enumerate(boxes):
        for j, c in enumerate(CLASSES):
            f = crop_feature(g, b, c)
            if f is not None:
                Wt, bias = model[c]
                out[i, j] = float(f @ Wt.ravel() + bias)
    return out


def camera_samples_frame(root, fid, n_bg, seed):
    """Positive / negative HOG samples for each class from one training frame."""
    rng = np.random.default_rng(seed)
    g = load_gray(root, fid)
    ih, iw = g.shape
    objs = read_labels(Path(root) / "label_2" / f"{fid}.txt")
    all_boxes = np.array([o["box"] for o in objs]).reshape(-1, 4)
    res = {c: {"pos": [], "neg": [], "reg_x": [], "reg_t": []} for c in CLASSES}

    for o in objs:
        t, b = o["type"], o["box"]
        if t == "DontCare" or (b[3] - b[1]) < 20 or (b[2] - b[0]) < 8:
            continue
        for c in CLASSES:
            if t == c:
                if o["trunc"] > 0.5 or o["occ"] > 2:
                    continue
                for flip in (False, True):  # horizontal flip doubles the positives
                    f = crop_feature(g, b, c, flip)
                    if f is not None:
                        res[c]["pos"].append(f)
                for _ in range(4):          # jittered boxes -> training data for box regression
                    nb = jitter_box(b, rng, iw, ih)
                    if nb is None:
                        continue
                    f = crop_feature(g, nb, c)
                    if f is not None:
                        res[c]["reg_x"].append(f)
                        res[c]["reg_t"].append(encode_box(nb, b))
            elif rng.random() < 0.5:        # other classes are negatives too (one-vs-rest)
                f = crop_feature(g, b, c)
                if f is not None:
                    res[c]["neg"].append(f)

    for c in CLASSES:  # random background windows with the class's aspect ratio
        w, h = WINDOWS[c]
        got = tries = 0
        while got < n_bg and tries < 60:
            tries += 1
            H = float(np.exp(rng.uniform(np.log(25), np.log(260))))
            Wd = H * w / h
            if Wd >= iw or H >= ih - ROI_TOP:
                continue
            x0 = rng.uniform(0, iw - Wd)
            y0 = rng.uniform(ROI_TOP, ih - H)
            win = np.array([x0, y0, x0 + Wd, y0 + H])
            if len(all_boxes) and overlap_ratio(win, all_boxes).max() > 0.1:
                continue
            f = crop_feature(g, win, c)
            if f is not None:
                res[c]["neg"].append(f)
                got += 1
    return res


def mine_frame(root, fid, model, per_class=60, thresh=-0.5):
    """Hard negatives: windows scoring above `thresh` (i.e. inside or beyond the SVM margin) that overlap no label."""
    g = load_gray(root, fid)
    objs = read_labels(Path(root) / "label_2" / f"{fid}.txt")
    all_boxes = np.array([o["box"] for o in objs]).reshape(-1, 4)
    raw = camera_windows(g, model, thresh=thresh, want_feats=True)
    out = {c: [] for c in CLASSES}
    for c in CLASSES:
        boxes, scores, feats = raw[c]
        if len(boxes) == 0:
            continue
        keep = nms(boxes, scores, 0.3)
        n = 0
        for k in keep:
            if len(all_boxes) and overlap_ratio(boxes[k], all_boxes).max() > 0.1:
                continue
            out[c].append(feats[k])
            n += 1
            if n >= per_class:
                break
    return out


def fit_svms(pos, neg):
    model = {}
    for c in CLASSES:
        if len(pos[c]) == 0:
            raise RuntimeError(f"No positive training crops for class '{c}'. "
                               f"Increase --n-svm or check the dataset path.")
        X = np.vstack(pos[c] + neg[c]).astype(np.float32)
        y = np.concatenate([np.ones(len(pos[c])), np.zeros(len(neg[c]))])
        svm = LinearSVC(C=0.05, class_weight="balanced", max_iter=5000).fit(X, y)
        br, bc = blocks_for(c)
        model[c] = (svm.coef_[0].reshape(br, bc, 36).astype(np.float32),
                    float(svm.intercept_[0]))
    return model


def train_camera_model(root, svm_ids, n_hard, rounds, n_jobs, refine=True):
    print(f"\n[camera] collecting HOG samples from {len(svm_ids)} frames ...")
    parts = Parallel(n_jobs=n_jobs, verbose=2)(
        delayed(camera_samples_frame)(root, fid, 6, 1000 + i) for i, fid in enumerate(svm_ids))
    pos = {c: [] for c in CLASSES}
    neg = {c: [] for c in CLASSES}
    reg_x = {c: [] for c in CLASSES}
    reg_t = {c: [] for c in CLASSES}
    for p in parts:
        for c in CLASSES:
            pos[c].extend(p[c]["pos"])
            neg[c].extend(p[c]["neg"])
            reg_x[c].extend(p[c]["reg_x"])
            reg_t[c].extend(p[c]["reg_t"])
    print("[camera] positives per class:", {c: len(pos[c]) for c in CLASSES})
    print("[camera] negatives per class:", {c: len(neg[c]) for c in CLASSES})

    model = fit_svms(pos, neg)
    print("[camera] initial SVMs trained")

    # Several rounds of hard-negative mining, each on different frames: the sliding window
    # sees ~100k windows per image, so the SVM needs many examples of what it gets wrong.
    n_hard = min(n_hard, len(svm_ids))
    order = svm_ids[::-1]
    for rnd in range(rounds if n_hard > 0 else 0):
        start = (rnd * n_hard) % len(order)
        mine_ids = (order + order)[start:start + n_hard]
        print(f"[camera] mining round {rnd + 1}/{rounds} on {len(mine_ids)} frames ...")
        mined = Parallel(n_jobs=n_jobs, verbose=2)(
            delayed(mine_frame)(root, fid, model) for fid in mine_ids)
        added = 0
        for m in mined:
            for c in CLASSES:
                neg[c].extend(m[c])
                added += len(m[c])
        model = fit_svms(pos, neg)
        print(f"[camera] retrained with {added} new hard negatives "
              f"(total negatives per class: {len(neg['Car'])})")
    if refine:
        model["reg"] = fit_box_regressors(reg_x, reg_t)
        print("[camera] box regressors trained:", {c: len(reg_x[c]) for c in CLASSES})
    return model


# ----------------------------------------------------------------------------------------
# LiDAR: ground removal -> DBSCAN -> cluster features
# ----------------------------------------------------------------------------------------
def project(P2, pts):
    h = np.c_[pts, np.ones(len(pts))] @ P2.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return h[:, 0] / h[:, 2], h[:, 1] / h[:, 2], h[:, 2]


def fit_ground(P, rng):
    """RANSAC ground plane n.p + d = 0 in camera coords (y points down). n.p + d = height above ground."""
    default = (np.array([0.0, -1.0, 0.0]), 1.65)
    cand = P[(P[:, 1] > 0.9) & (P[:, 1] < 2.6) & (P[:, 2] < 50)]
    if len(cand) < 300:
        return default
    if len(cand) > 20000:
        cand = cand[rng.choice(len(cand), 20000, replace=False)]
    best_cnt, best = 0, None
    for _ in range(100):
        p = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n = n / nn
        if abs(n[1]) < 0.9:
            continue
        if n[1] > 0:
            n = -n
        d = -n @ p[0]
        cnt = np.count_nonzero(np.abs(cand @ n + d) < 0.12)
        if cnt > best_cnt:
            best_cnt, best = cnt, (n, d)
    if best is None:
        return default
    n, d = best
    inl = cand[np.abs(cand @ n + d) < 0.12]
    c = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
    n2 = vt[-1]
    if n2[1] > 0:
        n2 = -n2
    if abs(n2[1]) < 0.9:
        return n, float(d)
    return n2, float(-n2 @ c)


def cluster_features(P, I, hgt):
    """Hand-crafted descriptors of one LiDAR cluster (P: n x 3 camera coords, I: intensity, hgt: height above ground)."""
    n = len(P)
    c = P.mean(0)
    xz = P[:, [0, 2]] - c[[0, 2]]
    ev, evec = np.linalg.eigh(np.cov(xz.T) + 1e-6 * np.eye(2))
    major = evec[:, 1]
    pl, pw = xz @ major, xz @ evec[:, 0]
    length, width = np.ptp(pl), np.ptp(pw)
    rel = np.arctan2(major[1], major[0]) - np.arctan2(c[2], c[0])  # major axis vs line of sight
    height = np.ptp(hgt)
    ev3 = np.sort(np.linalg.eigvalsh(np.cov(P.T) + 1e-6 * np.eye(3)))[::-1]
    bins = np.histogram(hgt, bins=[-1, 0.5, 1.0, 1.5, 2.0, 10])[0] / n
    vol = max(length, 0.1) * max(width, 0.1) * max(height, 0.1)
    f = [n, np.log(n), c[0], c[2], np.hypot(c[0], c[2]), np.arctan2(c[0], c[2]),
         length, width, height, hgt.min(), hgt.max(), hgt.mean(), hgt.std(),
         np.ptp(P[:, 0]), np.ptp(P[:, 2]), P[:, 0].std(), P[:, 2].std(),
         np.sqrt(ev3[0]), np.sqrt(ev3[1]), np.sqrt(ev3[2]), ev3[1] / ev3[0], ev3[2] / ev3[0],
         np.cos(2 * rel), np.sin(2 * rel), length / (width + 0.05),
         I.mean(), I.std(), I.max(), n / vol, *bins]
    return np.asarray(f, dtype=np.float32)


N_FEATS = len(cluster_features(np.random.default_rng(0).random((10, 3)),
                               np.random.default_rng(1).random(10),
                               np.random.default_rng(2).random(10)))


def inside_gt_box(P, o, margin=0.3):
    h, w, l = o["hwl"]
    d = P - o["loc"]
    c, s = np.cos(o["ry"]), np.sin(o["ry"])
    xo = c * d[:, 0] - s * d[:, 2]
    zo = s * d[:, 0] + c * d[:, 2]
    yo = d[:, 1]  # label location is the bottom centre; y points down, so the box spans [-h, 0]
    return ((np.abs(xo) <= l / 2 + margin) & (np.abs(zo) <= w / 2 + margin)
            & (yo <= margin) & (yo >= -h - margin))


def lidar_frame(root, fid, cache_dir, eps=0.7, min_samples=5, min_pts=8):
    """LiDAR cluster proposals for one frame (cached): features, projected 2D boxes, training labels."""
    cache_path = Path(cache_dir) / f"lidar_v2_{fid}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    t0 = time.perf_counter()
    root = Path(root)
    P2, R0, Tr = read_calib(root / "calib" / f"{fid}.txt")
    iw, ih = Image.open(root / "image_2" / f"{fid}.png").size
    velo = np.fromfile(root / "velodyne" / f"{fid}.bin", dtype=np.float32).reshape(-1, 4)
    pts = (R0 @ Tr @ np.c_[velo[:, :3], np.ones(len(velo))].T).T[:, :3]
    u, v, depth = project(P2, pts)
    fov = (depth > 1.0) & (depth < 60.0) & (u >= 0) & (u < iw) & (v >= 0) & (v < ih)

    result = {"feats": np.zeros((0, N_FEATS), np.float32), "boxes": np.zeros((0, 4)),
              "labels": np.zeros(0, dtype=int),
              "gt_boxes": np.zeros((0, 4))}
    P, I = pts[fov], velo[fov, 3]
    if len(P) >= 50:
        rng = np.random.default_rng(int(fid) if str(fid).isdigit() else 0)
        n_g, d_g = fit_ground(P, rng)
        hgt = P @ n_g + d_g
        keep = (hgt > 0.25) & (hgt < 3.5)
        Pn, In, Hn = P[keep], I[keep], hgt[keep]

        feats, boxes, members = [], [], []
        if len(Pn) >= min_pts:
            cl = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Pn)
            for k in range(cl.max() + 1):
                idx = np.where(cl == k)[0]
                if len(idx) < min_pts:
                    continue
                Pc, Ic, Hc = Pn[idx], In[idx], Hn[idx]
                feat = cluster_features(Pc, Ic, Hc)
                length, width, height = feat[6], feat[7], feat[8]
                if length > 14 or width > 14 or height > 4.0:   # walls, fences, merged structures
                    continue
                # project the cluster plus a copy dropped onto the ground plane (box reaches the road)
                twin = Pc.copy()
                twin[:, 1] = -(n_g[0] * twin[:, 0] + n_g[2] * twin[:, 2] + d_g) / n_g[1]
                cu, cv, _ = project(P2, np.vstack([Pc, twin]))
                box = np.array([max(0, np.nanmin(cu)), max(0, np.nanmin(cv)),
                                min(iw - 1, np.nanmax(cu)), min(ih - 1, np.nanmax(cv))])
                if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                    continue
                feats.append(feat)
                boxes.append(box)
                members.append(Pc)

        if feats:
            labels = np.full(len(feats), BG, dtype=int)
            gt_boxes = np.full((len(feats), 4), np.nan)
            label_path = root / "label_2" / f"{fid}.txt"
            if label_path.exists():
                objs = [o for o in read_labels(label_path) if o["type"] != "DontCare"]
                for i, Pc in enumerate(members):
                    best_frac, best_type, best_box = 0.0, None, None
                    for o in objs:
                        frac = inside_gt_box(Pc, o).mean()
                        if frac > best_frac:
                            best_frac, best_type, best_box = frac, o["type"], o["box"]
                    if best_frac >= 0.5:
                        labels[i] = CLASS_ID.get(best_type, IGNORE)
                        if best_type in CLASS_ID:
                            gt_boxes[i] = best_box      # regression target for box refinement
            result = {"feats": np.stack(feats), "boxes": np.stack(boxes), "labels": labels,
                      "gt_boxes": gt_boxes}

    result["t"] = time.perf_counter() - t0
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(result, fh)
    return result


def fusion_features(lidar_feats, cam_scores, crop_sc):
    """LiDAR geometry + sliding-window SVM scores on the box + image-crop classifier scores."""
    return np.hstack([lidar_feats, cam_scores, cam_scores.max(axis=1, keepdims=True), crop_sc])


# ---- image-crop classifier: appearance of each LiDAR cluster box, learned on the cluster boxes themselves ----
CROP_SIZE = 48


def crop_hog(g, box):
    ih, iw = g.shape
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(iw, x1), min(ih, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    patch = Image.fromarray(np.ascontiguousarray(g[y0:y1, x0:x1])).resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)
    return hog(np.asarray(patch), **HOG_KW).astype(np.float32)


CROP_DIM = (CROP_SIZE // CELL - 1) ** 2 * 36


def crop_worker(root, fid, boxes):
    """HOG of every cluster box (resized to a square). Returns (features, seconds)."""
    t0 = time.perf_counter()
    if len(boxes) == 0:
        return np.zeros((0, CROP_DIM), np.float32), 0.0
    g = load_gray(root, fid)
    out = np.zeros((len(boxes), CROP_DIM), np.float32)
    for i, b in enumerate(boxes):
        f = crop_hog(g, b)
        if f is not None:
            out[i] = f
    return out, time.perf_counter() - t0


def fit_crop_model(X, y, rng, max_bg=30000):
    """6-way linear SVM (5 classes + background) on cluster crops; background is subsampled for speed."""
    obj = np.where((y != IGNORE) & (y != BG))[0]
    bg = np.where(y == BG)[0]
    if len(bg) > max_bg:
        bg = rng.choice(bg, max_bg, replace=False)
    idx = np.concatenate([obj, bg])
    return LinearSVC(C=0.02, class_weight="balanced", max_iter=2000).fit(X[idx], y[idx])


def crop_scores(clf, X):
    out = np.full((len(X), BG + 1), -3.0, dtype=np.float32)
    if len(X):
        d = clf.decision_function(X)
        out[:, clf.classes_] = d if d.ndim == 2 else np.stack([-d, d], axis=1)
    return out


def oof_crop_scores(X, y, frame_idx, folds=3):
    """Out-of-fold scores, so the fusion forest sees the same score quality on training and unseen frames."""
    out = np.zeros((len(X), BG + 1), dtype=np.float32)
    rng = np.random.default_rng(0)
    for k in range(folds):
        te = frame_idx % folds == k
        out[te] = crop_scores(fit_crop_model(X[~te], y[~te], rng), X[te])
    return out


def box_reg_features(feats, boxes):
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    norm = boxes / np.array([IMG_W, IMG_H, IMG_W, IMG_H])
    return np.hstack([feats, norm, w[:, None] / IMG_W, h[:, None] / IMG_H, (w / h)[:, None]])


def fit_lidar_box_regressor(lid_list):
    """Learn to turn a cluster's projected box into the ground-truth 2D box (trees on cluster features)."""
    X, T = [], []
    for l in lid_list:
        m = (l["labels"] >= 0) & (l["labels"] < BG) & ~np.isnan(l["gt_boxes"]).any(axis=1)
        if m.any():
            X.append(box_reg_features(l["feats"][m], l["boxes"][m]))
            T.append(encode_box_many(l["boxes"][m], l["gt_boxes"][m]))
    if not X:
        return None
    reg = ExtraTreesRegressor(n_estimators=150, min_samples_leaf=3, n_jobs=-1, random_state=42)
    return reg.fit(np.vstack(X), np.clip(np.vstack(T), -2, 2))


def encode_box_many(boxes, gts):
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    return (gts - boxes) / np.stack([w, h, w, h], axis=1)


def apply_lidar_box_regressor(reg, lid_list):
    if reg is None:
        return [l["boxes"] for l in lid_list]
    counts = [len(l["boxes"]) for l in lid_list]
    if sum(counts) == 0:
        return [l["boxes"] for l in lid_list]
    X = np.vstack([box_reg_features(l["feats"], l["boxes"]) for l in lid_list if len(l["boxes"])])
    B = np.vstack([l["boxes"] for l in lid_list if len(l["boxes"])])
    refined = decode_boxes(B, reg.predict(X))
    return np.split(refined, np.cumsum(counts)[:-1]) if len(counts) > 1 else [refined]


def oof_lidar_boxes(lid_list, folds=3):
    """Out-of-fold refined boxes, so training frames get boxes of the same quality as unseen frames."""
    out = [None] * len(lid_list)
    idx = np.arange(len(lid_list))
    for k in range(folds):
        test = idx[k::folds]
        train = np.setdiff1d(idx, test)
        reg = fit_lidar_box_regressor([lid_list[i] for i in train])
        for i, b in zip(test, apply_lidar_box_regressor(reg, [lid_list[i] for i in test])):
            out[i] = b
    return out


def camera_worker(root, fid, boxes, cam_model, run_detector):
    """Camera SVM scores on the (refined) LiDAR cluster boxes, and optionally the camera-only detector."""
    g = load_gray(root, fid)
    t0 = time.perf_counter()
    out = {"cam_scores": camera_box_scores(g, boxes, cam_model)}
    out["t_box"] = time.perf_counter() - t0
    if run_detector:
        t0 = time.perf_counter()
        out["cam"] = camera_detect(g, cam_model)
        out["t_cam"] = time.perf_counter() - t0
    return out


def make_rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                  class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=42)


def predict_all(rf, feat_list):
    """One batched predict_proba for all frames; returns a (n_clusters x 6) matrix per frame."""
    counts = [len(f) for f in feat_list]
    if sum(counts) == 0:
        return [np.zeros((0, BG + 1)) for _ in feat_list]
    P = np.zeros((sum(counts), BG + 1))
    P[:, rf.classes_] = rf.predict_proba(np.vstack([f for f in feat_list if len(f)]))
    return np.split(P, np.cumsum(counts)[:-1])


def dets_from_proba(P, boxes, min_score=0.03, nms_thr=0.5):
    parts = []
    for ci in range(len(CLASSES)):
        sel = np.where(P[:, ci] >= min_score)[0]
        if len(sel) == 0:
            continue
        keep = nms(boxes[sel], P[sel, ci], nms_thr)
        parts.append((boxes[sel][keep], P[sel, ci][keep], np.full(len(keep), ci)))
    return merge_dets(parts)


# ----------------------------------------------------------------------------------------
# Evaluation (KITTI-style AP, 40 recall points, 2D boxes)
# ----------------------------------------------------------------------------------------
def build_gt(objs, difficulty):
    min_h, max_occ, max_trunc = DIFFICULTY[difficulty]
    gt = {"cls": {c: {"boxes": [], "ignore": []} for c in CLASSES}, "dontcare": []}
    for o in objs:
        if o["type"] == "DontCare":
            gt["dontcare"].append(o["box"])
        elif o["type"] in CLASS_ID:
            b = o["box"]
            ignore = ((b[3] - b[1]) < min_h or o["occ"] > max_occ or o["trunc"] > max_trunc)
            gt["cls"][o["type"]]["boxes"].append(b)
            gt["cls"][o["type"]]["ignore"].append(ignore)
    return gt


def eval_class(dets, gts, ci, iou_thr, min_h):
    cname = CLASSES[ci]
    records, npos = [], 0
    for d, g in zip(dets, gts):
        gb = np.array(g["cls"][cname]["boxes"]).reshape(-1, 4)
        gi = np.array(g["cls"][cname]["ignore"], dtype=bool)
        dc = np.array(g["dontcare"]).reshape(-1, 4)
        npos += int((~gi).sum())
        sel = d["labels"] == ci
        boxes, scores = d["boxes"][sel], d["scores"][sel]
        matched = np.zeros(len(gb), dtype=bool)
        for k in np.argsort(-scores):
            b = boxes[k]
            ious = iou_many(b, gb)
            ious[matched] = -1.0
            if len(ious) and ious.max() >= iou_thr:
                j = int(ious.argmax())
                if gi[j]:
                    continue                      # matched an ignored GT: neither TP nor FP
                matched[j] = True
                records.append((scores[k], 1))
            else:
                if (b[3] - b[1]) < min_h:
                    continue                      # too small for this difficulty
                if len(dc) and overlap_ratio(b, dc).max() > 0.5:
                    continue                      # inside a DontCare region
                records.append((scores[k], 0))
    if npos == 0:
        return None
    records.sort(key=lambda r: -r[0])
    flags = np.array([r[1] for r in records], dtype=float)
    scores = np.array([r[0] for r in records])
    tp = np.cumsum(flags)
    fp = np.cumsum(1 - flags)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / npos
    ap = 0.0
    for r in np.linspace(1 / 40, 1, 40):
        m = prec[rec >= r]
        ap += (m.max() if len(m) else 0.0) / 40
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    b = int(f1.argmax()) if len(f1) else 0
    return {"ap": float(ap), "npos": npos,
            "best_f1": float(f1[b]) if len(f1) else 0.0,
            "best_p": float(prec[b]) if len(f1) else 0.0,
            "best_r": float(rec[b]) if len(f1) else 0.0,
            "best_thr": float(scores[b]) if len(f1) else 0.0,
            "prec": prec, "rec": rec}


def evaluate(dets, gts, iou_thr, difficulty):
    min_h = DIFFICULTY[difficulty][0]
    return {c: eval_class(dets, gts, ci, iou_thr, min_h) for ci, c in enumerate(CLASSES)}


def print_table(results, title="AP (%)"):
    print("\n" + "=" * 86)
    print(f"{title:<14}" + "".join(f"{c:>11}" for c in CLASSES) + f"{'mAP':>11}")
    print("-" * 86)
    for m in METHODS:
        aps = [results[m][c]["ap"] * 100 if results[m][c] else np.nan for c in CLASSES]
        row = "".join(f"{a:>11.1f}" if not np.isnan(a) else f"{'n/a':>11}" for a in aps)
        print(f"{m:<14}{row}{np.nanmean(aps):>11.1f}")
    print("-" * 86)
    print("Best-F1 operating point  (precision / recall):")
    for m in METHODS:
        cells = []
        for c in CLASSES:
            r = results[m][c]
            cells.append(f"{r['best_p']:.2f}/{r['best_r']:.2f}" if r else "n/a")
        print(f"{m:<14}" + "".join(f"{x:>11}" for x in cells))
    print("=" * 86)


def proposal_recall(boxes_list, gts, iou_thr):
    """Fraction of (valid) GT objects that have at least one LiDAR cluster box with IoU >= thr.
    This is the hard ceiling for LiDAR-only and fusion: no classifier can recover a missed cluster."""
    hit = {c: 0 for c in CLASSES}
    tot = {c: 0 for c in CLASSES}
    for boxes, g in zip(boxes_list, gts):
        for c in CLASSES:
            for b, ign in zip(g["cls"][c]["boxes"], g["cls"][c]["ignore"]):
                if ign:
                    continue
                tot[c] += 1
                if len(boxes) and iou_many(b, boxes).max() >= iou_thr:
                    hit[c] += 1
    return {c: (hit[c] / tot[c] if tot[c] else None) for c in CLASSES}, tot


def print_diagnostics(results, raw_boxes, ref_boxes, gts, iou_thr):
    rec_raw, tot = proposal_recall(raw_boxes, gts, iou_thr)
    rec_ref, _ = proposal_recall(ref_boxes, gts, iou_thr)
    rec_loose, _ = proposal_recall(ref_boxes, gts, 0.3)
    fmt = lambda x: f"{x * 100:>10.1f}%" if x is not None else f"{'n/a':>11}"
    print("\nDiagnostics - where is recall lost?  (valid ground-truth objects in the validation frames)")
    print(f"{'':<34}" + "".join(f"{c:>11}" for c in CLASSES))
    print(f"{'GT objects':<34}" + "".join(f"{tot[c]:>11}" for c in CLASSES))
    print(f"{f'cluster recall, raw, IoU>={iou_thr}':<34}" + "".join(fmt(rec_raw[c]) for c in CLASSES))
    print(f"{f'cluster recall, refined, IoU>={iou_thr}':<34}" + "".join(fmt(rec_ref[c]) for c in CLASSES)
          + "   <- ceiling for LiDAR-only & fusion")
    print(f"{'cluster recall, refined, IoU>=0.3':<34}" + "".join(fmt(rec_loose[c]) for c in CLASSES)
          + "   <- a cluster exists but the box is loose")
    for m in METHODS:
        row = [results[m][c]["rec"][-1] if results[m][c] is not None and len(results[m][c]["rec"]) else None
               for c in CLASSES]
        print(f"{m + ' max recall':<34}" + "".join(fmt(x) for x in row))


def plot_pr(results, path):
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(4 * len(CLASSES), 3.8), sharey=True)
    colors = {"Camera-only": "tab:blue", "LiDAR-only": "tab:green", "Fusion": "tab:red"}
    for ax, c in zip(axes, CLASSES):
        for m in METHODS:
            r = results[m][c]
            if r:
                ax.plot(r["rec"], r["prec"], color=colors[m], label=f"{m} (AP {r['ap'] * 100:.1f})")
        ax.set_title(c)
        ax.set_xlabel("Recall")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel("Precision")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def visualize(root, fids, val_ids, all_dets, gts, out_dir, thresholds):
    for fid in fids:
        i = val_ids.index(fid)
        img = load_rgb(root, fid)
        fig, axes = plt.subplots(len(METHODS), 1, figsize=(13, 3.9 * len(METHODS)))
        for ax, m in zip(axes, METHODS):
            ax.imshow(img)
            for c in CLASSES:   # ground truth in green (only the five modelled classes)
                for b in gts[i]["cls"][c]["boxes"]:
                    ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                               fill=False, edgecolor="lime", linewidth=1.2))
            d = all_dets[m][i]
            for b, s, l in zip(d["boxes"], d["scores"], d["labels"]):
                if s < thresholds[m]:
                    continue
                col = COLORS[CLASSES[l]]
                ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                           fill=False, edgecolor=col, linewidth=1.8))
                ax.text(b[0], b[1] - 2, f"{CLASSES[l]} {s:.2f}", color=col, fontsize=7,
                        bbox=dict(facecolor="black", alpha=0.5, pad=0.5, edgecolor="none"))
            ax.set_title(f"Frame {fid} - {m}   (green = ground truth)")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(Path(out_dir) / f"detections_{fid}.png", dpi=120)
        plt.close(fig)


# ----------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=r"D:\KITTI_object_dataset\training",
                    help="KITTI 'training' folder (contains image_2, label_2, velodyne, calib)")
    ap.add_argument("--out", default="kitti_fusion_output", help="output folder (caches, plots, results)")
    ap.add_argument("--n-svm", type=int, default=3000, help="frames used to train the camera SVMs")
    ap.add_argument("--n-hard", type=int, default=300, help="frames per hard-negative mining round (0 = off)")
    ap.add_argument("--mine-rounds", type=int, default=3, help="number of hard-negative mining rounds")
    ap.add_argument("--n-fusion", type=int, default=2900, help="frames used to train the LiDAR / fusion forests")
    ap.add_argument("--n-val", type=int, default=600, help="validation frames used for evaluation")
    ap.add_argument("--jobs", type=int, default=-1, help="parallel worker processes (-1 = all cores)")
    ap.add_argument("--difficulty", choices=list(DIFFICULTY), default="moderate")
    ap.add_argument("--iou", type=float, default=0.5, help="2D IoU threshold for a true positive")
    ap.add_argument("--vis", type=int, default=5, help="number of validation frames to visualise")
    ap.add_argument("--no-refine", action="store_true", help="disable learned box refinement (ablation)")
    ap.add_argument("--retrain", action="store_true", help="ignore the cached camera model")
    ap.add_argument("--quick", action="store_true", help="tiny run to test that everything works")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.n_svm, args.n_hard, args.n_fusion, args.n_val, args.vis = 150, 20, 150, 40, 3
        args.mine_rounds = 2
    refine = not args.no_refine

    root = Path(args.data)
    for sub in ("image_2", "label_2", "velodyne", "calib"):
        if not (root / sub).is_dir():
            raise SystemExit(f"Missing folder: {root / sub}\n"
                             f"Point --data at the KITTI 'training' folder.")
    out_dir = Path(args.out)
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- splits ----
    ids = sorted(p.stem for p in (root / "label_2").glob("*.txt"))
    random.Random(42).shuffle(ids)
    cut = int(0.8 * len(ids))
    train_ids, val_pool = ids[:cut], ids[cut:]
    if args.n_svm + args.n_fusion > len(train_ids):
        scale = len(train_ids) / (args.n_svm + args.n_fusion)
        args.n_svm, args.n_fusion = int(args.n_svm * scale), int(args.n_fusion * scale)
        print(f"Not enough training frames; using n_svm={args.n_svm}, n_fusion={args.n_fusion}")
    svm_ids = train_ids[:args.n_svm]
    fus_ids = train_ids[args.n_svm:args.n_svm + args.n_fusion]
    val_ids = val_pool[:args.n_val]
    print(f"Frames: total={len(ids)}  svm={len(svm_ids)}  fusion={len(fus_ids)}  val={len(val_ids)}  "
          f"box refinement={'on' if refine else 'off'}")

    # ---- 1) camera SVMs (+ box regressors) ----
    t_start = time.time()
    model_path = out_dir / (f"camera_model_{len(svm_ids)}_{args.n_hard}x{args.mine_rounds}"
                            f"_r{int(refine)}.joblib")
    if model_path.exists() and not args.retrain:
        cam_model = load(model_path)
        print(f"[camera] loaded cached model {model_path}")
    else:
        cam_model = train_camera_model(root, svm_ids, args.n_hard, args.mine_rounds, args.jobs, refine)
        dump(cam_model, model_path)

    # ---- 2) LiDAR clusters (cached on disk) ----
    print(f"\n[lidar] clustering {len(fus_ids)} fusion-training frames ...")
    fus_lid = Parallel(n_jobs=args.jobs, verbose=2)(
        delayed(lidar_frame)(root, fid, cache_dir) for fid in fus_ids)
    print(f"[lidar] clustering {len(val_ids)} validation frames ...")
    val_lid = Parallel(n_jobs=args.jobs, verbose=2)(
        delayed(lidar_frame)(root, fid, cache_dir) for fid in val_ids)

    # ---- 3) learned refinement of the cluster boxes ----
    val_raw_boxes = [l["boxes"] for l in val_lid]
    if refine:
        print("[lidar] training box regressor (3-fold out-of-fold boxes for the training frames) ...")
        fus_boxes = oof_lidar_boxes(fus_lid)
        box_reg = fit_lidar_box_regressor(fus_lid)
        val_boxes = apply_lidar_box_regressor(box_reg, val_lid)
    else:
        fus_boxes, val_boxes = [l["boxes"] for l in fus_lid], val_raw_boxes

    # ---- 4) camera scores on the cluster boxes, then the two forests ----
    print("[fusion] camera SVM scores on cluster boxes ...")
    fus_cam = Parallel(n_jobs=args.jobs, verbose=2)(
        delayed(camera_worker)(root, fid, b, cam_model, False) for fid, b in zip(fus_ids, fus_boxes))
    y = np.concatenate([l["labels"] for l in fus_lid])
    Xl = np.vstack([l["feats"] for l in fus_lid])
    Xs = np.vstack([c["cam_scores"] for c in fus_cam])
    ok = y != IGNORE
    names = CLASSES + ["Background"]
    print("[lidar] training clusters per label:",
          {names[k]: int((y == k).sum()) for k in range(BG + 1)}, " ignored:", int((~ok).sum()))
    if len(np.unique(y[ok])) < 2:
        raise SystemExit("Not enough labelled clusters to train. Check calibration / velodyne files.")

    print("[fusion] HOG crops of the cluster boxes ...")
    fus_crops = Parallel(n_jobs=args.jobs, verbose=2)(
        delayed(crop_worker)(root, fid, b) for fid, b in zip(fus_ids, fus_boxes))
    Xc = np.vstack([c[0] for c in fus_crops])
    del fus_crops
    frame_idx = np.concatenate([np.full(len(b), i) for i, b in enumerate(fus_boxes)])
    print("[fusion] training image-crop classifier (3-fold out-of-fold scores for the training frames) ...")
    S_fus = oof_crop_scores(Xc, y, frame_idx)
    crop_clf = fit_crop_model(Xc, y, np.random.default_rng(1))
    del Xc

    print("[lidar] training LiDAR-only forest ...")
    rf_lidar = make_rf().fit(Xl[ok], y[ok])
    print("[fusion] training fusion forest ...")
    Xf = fusion_features(Xl, Xs, S_fus)
    rf_fusion = make_rf().fit(Xf[ok], y[ok])
    imp, n_l = rf_fusion.feature_importances_, Xl.shape[1]
    print("[fusion] forest feature importance:",
          {"LiDAR geometry": f"{imp[:n_l].sum() * 100:.1f}%",
           "sliding-window SVM scores": f"{imp[n_l:n_l + len(CLASSES) + 1].sum() * 100:.1f}%",
           "image-crop classifier": f"{imp[n_l + len(CLASSES) + 1:].sum() * 100:.1f}%"})

    # ---- 5) validation ----
    print(f"\n[eval] running all three detectors on {len(val_ids)} validation frames ...")
    val_cam = Parallel(n_jobs=args.jobs, verbose=2)(
        delayed(camera_worker)(root, fid, b, cam_model, True) for fid, b in zip(val_ids, val_boxes))
    val_crops = Parallel(n_jobs=args.jobs, verbose=2)(
        delayed(crop_worker)(root, fid, b) for fid, b in zip(val_ids, val_boxes))
    counts = [len(b) for b in val_boxes]
    t0 = time.perf_counter()
    S_val = crop_scores(crop_clf, np.vstack([c[0] for c in val_crops]))
    S_val = np.split(S_val, np.cumsum(counts)[:-1])
    t_crop = np.mean([c[1] for c in val_crops]) + (time.perf_counter() - t0) / len(val_ids)

    t0 = time.perf_counter()
    P_lidar = predict_all(rf_lidar, [l["feats"] for l in val_lid])
    t_rf_l = (time.perf_counter() - t0) / len(val_ids)
    t0 = time.perf_counter()
    P_fusion = predict_all(rf_fusion, [fusion_features(l["feats"], c["cam_scores"], sv)
                                       for l, c, sv in zip(val_lid, val_cam, S_val)])
    t_rf_f = (time.perf_counter() - t0) / len(val_ids)

    all_dets = {m: [] for m in METHODS}
    gts = []
    for fid, boxes, c, Pl, Pf in zip(val_ids, val_boxes, val_cam, P_lidar, P_fusion):
        all_dets["Camera-only"].append(c["cam"])
        all_dets["LiDAR-only"].append(dets_from_proba(Pl, boxes))
        all_dets["Fusion"].append(dets_from_proba(Pf, boxes))
        gts.append(build_gt(read_labels(root / "label_2" / f"{fid}.txt"), args.difficulty))

    results = {m: evaluate(all_dets[m], gts, args.iou, args.difficulty) for m in METHODS}
    print(f"\nEvaluation: 2D boxes, IoU >= {args.iou}, KITTI difficulty = {args.difficulty}, "
          f"{len(val_ids)} validation frames")
    print_table(results, "AP (%)")
    if args.iou > 0.3:   # same detections, looser overlap: separates 'wrong object' from 'loose box'
        loose = {m: evaluate(all_dets[m], gts, 0.3, args.difficulty) for m in METHODS}
        print("\nSame detections scored with a looser overlap requirement (localisation check):")
        print_table(loose, "AP IoU>=0.3")
    print_diagnostics(results, val_raw_boxes, val_boxes, gts, args.iou)

    # ---- timing ----
    t_cam = np.mean([c["t_cam"] for c in val_cam])
    t_lid = np.mean([l.get("t", 0.0) for l in val_lid])
    t_box = np.mean([c["t_box"] for c in val_cam])
    print("\nAverage CPU time per frame (single core, seconds):")
    print(f"  Camera-only : {t_cam:.2f}  (multi-scale sliding window + box regression)")
    print(f"  LiDAR-only  : {t_lid + t_rf_l:.2f}  (ground removal + DBSCAN + features + forest)")
    print(f"  Fusion      : {t_lid + t_box + t_crop + t_rf_f:.2f}  (LiDAR pipeline + camera scores on "
          f"{int(np.mean([len(b) for b in val_boxes]))} cluster boxes)")

    # ---- save everything ----
    summary = {m: {c: ({k: v for k, v in results[m][c].items() if k not in ("prec", "rec")}
                       if results[m][c] else None) for c in CLASSES} for m in METHODS}
    summary["settings"] = vars(args)
    with open(out_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    plot_pr(results, out_dir / "pr_curves.png")
    if args.vis > 0:
        visualize(root, val_ids[:args.vis], val_ids, all_dets, gts, out_dir,
                  thresholds={"Camera-only": 0.5, "LiDAR-only": 0.4, "Fusion": 0.4})
    print(f"\nSaved results.json, pr_curves.png and detection images to: {out_dir.resolve()}")
    print(f"Total run time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":   # required on Windows when using multiprocessing
    main()


"""
KITTI object detection with classical ML: camera-only vs LiDAR-only vs camera+LiDAR fusion.

Classes: Car, Van, Truck, Pedestrian, Cyclist.  CPU only, no deep learning.

  Camera-only : HOG + linear SVM (one SVM per class), multi-scale sliding window,
                hard-negative mining, per-class NMS.
  LiDAR-only  : ground-plane RANSAC -> DBSCAN clusters -> hand-crafted 3D cluster
                features -> Random Forest (5 classes + background).
  Fusion      : the same LiDAR clusters, but the Random Forest also sees the camera
                SVM scores computed on the cluster's projected 2D box (feature-level fusion).

Both the camera detections and the LiDAR cluster boxes are refined by small learned box regressors
(ridge on HOG features / extra-trees on cluster features); disable with --no-refine for an ablation.

All three are scored with the same 2D-box protocol (KITTI-style AP with 40 recall points,
difficulty filters, DontCare handling), so the comparison is apples-to-apples.

Data splits (all disjoint, from the KITTI "training" folder, shuffled with a fixed seed):
  svm frames    -> train the camera SVMs (+ hard-negative mining)
  fusion frames -> train the LiDAR-only and fusion Random Forests
                   (camera scores here come from SVMs that never saw these frames)
  val frames    -> evaluation only

Run:   python kitti_fusion.py --data "D:\\KITTI_object_dataset\\training"
Quick: python kitti_fusion.py --data "D:\\KITTI_object_dataset\\training" --quick
"""