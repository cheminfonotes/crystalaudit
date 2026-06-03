"""
api.py — FastAPI service for crystalaudit.

Run:  uvicorn api:app --reload      Docs: http://127.0.0.1:8000/docs
  GET  /analyze/{pdb_id}    analyze by 4-char PDB ID
  POST /analyze             analyze an uploaded .pdb/.cif file
  GET  /health
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import crystalaudit as ca

app = FastAPI(title="crystalaudit API",
              description="Attribute protein-geometry outliers to "
                          "biological / packing / error.",
              version=ca.__version__)


def _serialize(res: "ca.Result") -> dict:
    out = res.summary()
    out["feature_importance"] = (res.importance.round(4).to_dict()
                                 if res.importance is not None else None)
    out["outliers"] = (res.report.round(4).to_dict(orient="records")
                       if not res.report.empty else [])
    return out


@app.get("/health")
def health():
    return {"status": "ok", "version": ca.__version__}


@app.get("/analyze/{pdb_id}")
def analyze_id(pdb_id: str,
               z: float = Query(3.5, ge=2.5, le=6.0),
               contact_radius: float = Query(4.0, ge=3.0, le=6.0)):
    if len(pdb_id) != 4 or not pdb_id.isalnum():
        raise HTTPException(400, "pdb_id must be 4-character alphanumeric.")
    try:
        res = ca.analyze(pdb_id, z_outlier=z, contact_radius=contact_radius)
    except Exception as e:
        raise HTTPException(502, f"{type(e).__name__}: {e}")
    return JSONResponse(_serialize(res))


@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...),
                       z: float = Query(3.5, ge=2.5, le=6.0),
                       contact_radius: float = Query(4.0, ge=3.0, le=6.0)):
    if not file.filename.lower().endswith((".pdb", ".cif", ".ent")):
        raise HTTPException(400, "Upload a .pdb, .cif or .ent file.")
    try:
        stc, meta, label = ca.load_bytes(await file.read(), file.filename)
        res = ca.analyze(st=stc, meta=meta, label=label,
                         z_outlier=z, contact_radius=contact_radius)
    except Exception as e:
        raise HTTPException(422, f"{type(e).__name__}: {e}")
    return JSONResponse(_serialize(res))
