"""
pipeline.py — Automated Crystallographic Validation & Anomaly Attribution

Core, reusable pipeline. Accepts EITHER a 4-char PDB ID (downloaded from RCSB)
OR a local .pdb / .cif file path / raw text. Returns a per-outlier DataFrame
with attribution (biological / packing / error), confidence and explanation.

Used by both app.py (Streamlit) and api.py (FastAPI).
"""
from __future__ import annotations
import os, io, math, tempfile
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
import requests
import gemmi
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    cache_dir: str = "pdb_cache"
    z_outlier: float = 3.5
    contact_radius: float = 4.0
    exposure_radius: float = 8.0
    min_class_per_type: int = 5
    rf_estimators: int = 300

CFG = Config()

RCSB_CIF = "https://files.rcsb.org/download/{pid}.cif"
RCSB_PDB = "https://files.rcsb.org/download/{pid}.pdb"
RCSB_META = "https://data.rcsb.org/rest/v1/core/entry/{pid}"

BACKBONE = ("N", "CA", "C", "O", "CB")

# Engh-Huber ideal geometry (subset): (mean, sigma) in A / degrees
ENGH_HUBER = {
    "N-CA": (1.459, 0.020), "CA-C": (1.525, 0.021),
    "C-O": (1.229, 0.019),  "CA-CB": (1.521, 0.033),
    "N-CA-C": (111.0, 2.7),
}

FEATURE_COLS = ["abs_z", "sym_contact", "n_sym_contacts",
                "b_factor", "b_z", "exposure", "resolution", "r_free"]


# --------------------------------------------------------------------------- #
# 1. Ingestion — PDB ID, file path, or raw text
# --------------------------------------------------------------------------- #
def _fetch(url, timeout=25):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r


def load_input(source: str, cfg: Config = CFG):
    """
    Resolve `source` to a (gemmi.Structure, meta, label) tuple.

    source can be:
      - a 4-character PDB ID  -> downloaded from RCSB (cached)
      - a path to a local .pdb / .cif file
      - raw mmCIF / PDB text
    """
    os.makedirs(cfg.cache_dir, exist_ok=True)

    # case A: existing file path
    if os.path.exists(source) and os.path.isfile(source):
        st = gemmi.read_structure(source)
        st.setup_entities()
        label = os.path.basename(source)
        return st, _meta_from_structure(st), label

    # case B: looks like a 4-char PDB id
    token = source.strip()
    if len(token) == 4 and token.isalnum():
        pid = token.upper()
        path = os.path.join(cfg.cache_dir, f"{pid}.cif")
        if not os.path.exists(path):
            txt = _fetch(RCSB_CIF.format(pid=pid)).text
            with open(path, "w") as f:
                f.write(txt)
        st = gemmi.read_structure(path)
        st.setup_entities()
        meta = _meta_from_structure(st)
        _enrich_meta_from_api(pid, meta)
        return st, meta, pid

    # case C: raw text (mmCIF or PDB)
    if "ATOM" in source or "_atom_site" in source or "data_" in source:
        suffix = ".cif" if ("_atom_site" in source or "data_" in source) else ".pdb"
        with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as tf:
            tf.write(source)
            tmp = tf.name
        st = gemmi.read_structure(tmp)
        st.setup_entities()
        os.unlink(tmp)
        return st, _meta_from_structure(st), "uploaded"

    raise ValueError(
        f"Could not interpret input '{source[:40]}...'. "
        "Provide a 4-char PDB ID, a file path, or PDB/mmCIF text."
    )


def load_bytes(data: bytes, filename: str, cfg: Config = CFG):
    """Load from uploaded file bytes (used by Streamlit / FastAPI)."""
    suffix = os.path.splitext(filename)[1] or ".pdb"
    with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as tf:
        tf.write(data)
        tmp = tf.name
    st = gemmi.read_structure(tmp)
    st.setup_entities()
    os.unlink(tmp)
    return st, _meta_from_structure(st), filename


def _meta_from_structure(st):
    return {"resolution": st.resolution or 3.0, "r_work": None, "r_free": None}


def _enrich_meta_from_api(pid, meta):
    try:
        j = _fetch(RCSB_META.format(pid=pid)).json()
        ref = j.get("refine", [{}]) or [{}]
        meta["r_work"] = ref[0].get("ls_r_factor_r_work")
        meta["r_free"] = ref[0].get("ls_r_factor_r_free")
        res = j.get("rcsb_entry_info", {}).get("resolution_combined")
        if res:
            meta["resolution"] = res[0]
    except Exception:
        pass
    if meta.get("resolution") in (None, 0.0):
        meta["resolution"] = 3.0
    if meta.get("r_free") in (None, 0.0):
        meta["r_free"] = 0.25


# --------------------------------------------------------------------------- #
# 2. Geometry extraction
# --------------------------------------------------------------------------- #
def _atom_map(res):
    return {a.name: a.pos for a in res if a.name in BACKBONE}


def extract_geometry(label, st):
    rows = []
    model = st[0]
    for chain in model:
        residues = [r for r in chain if r.find_atom("CA", "*")]
        for idx, res in enumerate(residues):
            m = _atom_map(res)
            rn, num = res.name, res.seqid.num
            for a, b in (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB")):
                if a in m and b in m:
                    rows.append(dict(pid=label, chain=chain.name, resnum=num,
                        resname=rn, kind="bond", class_key=f"{rn}:{a}-{b}",
                        geom=f"{a}-{b}", value=m[a].dist(m[b])))
            if {"N", "CA", "C"} <= m.keys():
                ang = math.degrees(gemmi.calculate_angle(m["N"], m["CA"], m["C"]))
                rows.append(dict(pid=label, chain=chain.name, resnum=num,
                    resname=rn, kind="angle", class_key=f"{rn}:N-CA-C",
                    geom="N-CA-C", value=ang))
            if idx > 0:
                prev = _atom_map(residues[idx - 1])
                if "C" in prev and {"N", "CA", "C"} <= m.keys():
                    phi = math.degrees(gemmi.calculate_dihedral(
                        prev["C"], m["N"], m["CA"], m["C"]))
                    rows.append(dict(pid=label, chain=chain.name, resnum=num,
                        resname=rn, kind="torsion", class_key=f"{rn}:phi",
                        geom="phi", value=phi))
            if idx < len(residues) - 1:
                nxt = _atom_map(residues[idx + 1])
                if "N" in nxt and {"N", "CA", "C"} <= m.keys():
                    psi = math.degrees(gemmi.calculate_dihedral(
                        m["N"], m["CA"], m["C"], nxt["N"]))
                    rows.append(dict(pid=label, chain=chain.name, resnum=num,
                        resname=rn, kind="torsion", class_key=f"{rn}:psi",
                        geom="psi", value=psi))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. Priors (Engh-Huber + data-driven robust stats)
# --------------------------------------------------------------------------- #
def _circular_median(a):
    a = np.radians(np.asarray(a, float))
    return math.degrees(math.atan2(np.mean(np.sin(a)), np.mean(np.cos(a))))


def _circular_mad(a, c):
    d = (np.asarray(a, float) - c + 180) % 360 - 180
    return 1.4826 * np.median(np.abs(d))


def build_priors(geo, cfg=CFG):
    priors = {}
    for ck, grp in geo.groupby("class_key"):
        vals = grp["value"].values
        kind = grp["kind"].iloc[0]
        geom = grp["geom"].iloc[0]
        has_eh = geom in ENGH_HUBER
        enough = len(vals) >= cfg.min_class_per_type

        if kind == "torsion":
            if not enough:
                continue
            center = _circular_median(vals)
            spread = max(_circular_mad(vals, center), 1e-3)
        else:
            if enough:
                center = float(np.median(vals))
                spread = max(1.4826 * float(np.median(np.abs(vals - center))), 1e-3)
                if has_eh:                          # shrink toward literature
                    eh_c, eh_s = ENGH_HUBER[geom]
                    center = 0.5 * center + 0.5 * eh_c
                    spread = 0.5 * spread + 0.5 * eh_s
            elif has_eh:                            # pure literature prior
                center, spread = ENGH_HUBER[geom]
            else:
                continue
        priors[ck] = dict(center=center, spread=spread, kind=kind, n=len(vals))
    return priors


# --------------------------------------------------------------------------- #
# 4. Outlier scoring
# --------------------------------------------------------------------------- #
def score_outliers(geo, priors, cfg=CFG):
    recs = []
    for r in geo.itertuples(index=False):
        p = priors.get(r.class_key)
        if not p:
            continue
        if p["kind"] == "torsion":
            d = (r.value - p["center"] + 180) % 360 - 180
        else:
            d = r.value - p["center"]
        z = d / p["spread"]
        recs.append(dict(pid=r.pid, chain=r.chain, resnum=r.resnum,
            resname=r.resname, kind=r.kind, class_key=r.class_key,
            value=r.value, expected=p["center"], z=z, abs_z=abs(z)))
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["is_outlier"] = df["abs_z"] > cfg.z_outlier
    return df


# --------------------------------------------------------------------------- #
# 5. Context features (attribution signals)
# --------------------------------------------------------------------------- #
def residue_context(st, meta, chain_name, resnum, cfg=CFG):
    model = st[0]
    chain = next((c for c in model if c.name == chain_name), None)
    res = next((r for r in chain if r.seqid.num == resnum), None) if chain else None
    ca = res.find_atom("CA", "*") if res else None
    center = ca.pos if ca else (res[0].pos if res and len(res) else gemmi.Position(0, 0, 0))

    ns = gemmi.NeighborSearch(model, st.cell, cfg.exposure_radius).populate()

    sym_marks = ns.find_atoms(center, "\0", radius=cfg.contact_radius)
    sym = [m for m in sym_marks if m.image_idx != 0]
    sym_dists = [center.dist(m.to_cra(model).atom.pos) for m in sym]
    sym_contact = min(sym_dists) if sym_dists else cfg.contact_radius
    n_sym = len(sym)

    exposure = len(ns.find_atoms(center, "\0", radius=cfg.exposure_radius))
    bfac = float(np.mean([a.b_iso for a in res])) if res and len(res) else float("nan")

    return dict(sym_contact=sym_contact, n_sym_contacts=n_sym, exposure=exposure,
                b_factor=bfac, resolution=meta.get("resolution") or 3.0,
                r_free=meta.get("r_free") or 0.25)


def _bfactor_baseline(st):
    bs = np.array([a.b_iso for ch in st[0] for r in ch for a in r], float)
    med = np.median(bs)
    mad = 1.4826 * np.median(np.abs(bs - med)) + 1e-6
    return med, mad


# --------------------------------------------------------------------------- #
# 6. Weak labels + attribution model
# --------------------------------------------------------------------------- #
def weak_label(row, exposure_median, cfg=CFG):
    near_sym = (row["sym_contact"] < cfg.contact_radius - 0.3) and (row["n_sym_contacts"] >= 1)
    high_b = (row["b_z"] > 2.0) or (row["b_factor"] > 60)
    poor_res = (row["resolution"] is not None and row["resolution"] > 2.6)
    buried = row["exposure"] >= exposure_median
    if near_sym and not high_b:
        return "packing"
    if high_b or (poor_res and row["abs_z"] > cfg.z_outlier + 1.0):
        return "error"
    if buried and not near_sym:
        return "biological"
    return "packing" if near_sym else ("error" if high_b else "biological")


def _explain(row):
    a = row["attribution"]
    if a == "packing":
        return (f"Symmetry contact {row['sym_contact']:.2f} A "
                f"({int(row['n_sym_contacts'])} mate(s)) — likely crystal-packing artefact.")
    if a == "error":
        return (f"High relative B (z={row['b_z']:.1f}) / reso {row['resolution']:.2f} A — "
                f"likely refinement/data error; inspect density.")
    return ("Buried, no symmetry contact — possible functional (biological) strain; "
            "worth keeping.")


def _fit_attribution(feat, cfg=CFG):
    X = feat[FEATURE_COLS].fillna(feat[FEATURE_COLS].median())
    y = feat["weak_label"]
    clf = Pipeline([
        ("scale", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=cfg.rf_estimators,
                                      random_state=42, class_weight="balanced")),
    ])
    if y.nunique() < 2:                       # single class -> trivial
        feat["attribution"] = y
        feat["confidence"] = 1.0
        for c in ["biological", "packing", "error"]:
            feat[f"p_{c}"] = 1.0 if c == y.iloc[0] else 0.0
        return feat, None
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    for i, c in enumerate(clf.named_steps["rf"].classes_):
        feat[f"p_{c}"] = proba[:, i]
    for c in ["biological", "packing", "error"]:
        if f"p_{c}" not in feat:
            feat[f"p_{c}"] = 0.0
    feat["attribution"] = clf.predict(X)
    feat["confidence"] = proba.max(axis=1)
    imp = pd.Series(clf.named_steps["rf"].feature_importances_, index=FEATURE_COLS)
    return feat, imp.sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# 7. Public entry point
# --------------------------------------------------------------------------- #
def run(source=None, *, st=None, meta=None, label=None, cfg: Config = CFG):
    """
    Run the full pipeline on ONE structure.

    Pass either `source` (PDB ID / path / raw text) OR a pre-loaded
    (st, meta, label). Returns dict with keys:
      report (DataFrame), n_measurements, n_outliers, importance (Series|None),
      meta, label, scored (DataFrame).
    """
    if st is None:
        st, meta, label = load_input(source, cfg)
    geo = extract_geometry(label, st)
    if geo.empty:
        raise ValueError("No protein backbone found (need N/CA/C atoms).")
    priors = build_priors(geo, cfg)
    scored = score_outliers(geo, priors, cfg)
    outliers = scored[scored["is_outlier"]].reset_index(drop=True) if not scored.empty else scored

    summary = dict(label=label, meta=meta,
                   n_measurements=int(len(geo)),
                   n_outliers=int(len(outliers)),
                   scored=scored, importance=None,
                   report=pd.DataFrame())
    if outliers.empty:
        return summary

    ctx = pd.DataFrame([residue_context(st, meta, r.chain, r.resnum, cfg)
                        for r in outliers.itertuples(index=False)])
    med, mad = _bfactor_baseline(st)
    ctx["b_z"] = (ctx["b_factor"] - med) / mad
    feat = pd.concat([outliers, ctx], axis=1)
    exp_med = float(np.median(feat["exposure"]))
    feat["weak_label"] = feat.apply(lambda r: weak_label(r, exp_med, cfg), axis=1)
    feat, importance = _fit_attribution(feat, cfg)
    feat["explanation"] = feat.apply(_explain, axis=1)

    cols = ["pid", "chain", "resnum", "resname", "class_key", "value", "expected",
            "abs_z", "attribution", "confidence",
            "p_biological", "p_packing", "p_error", "explanation"]
    cols = [c for c in cols if c in feat.columns]
    report = feat[cols].sort_values("abs_z", ascending=False).reset_index(drop=True)
    summary["report"] = report
    summary["importance"] = importance
    summary["feat"] = feat
    return summary

