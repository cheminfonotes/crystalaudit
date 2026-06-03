# crystalaudit

**Automated crystallographic validation & anomaly attribution.**
Existing tools (MolProbity, WHAT_CHECK) *flag* protein-geometry outliers.
`crystalaudit` explains **why** each one is there — **biological** strain,
crystal **packing**, or refinement **error**.

```
PDB ID / .pdb / .cif
  → Gemmi geometry (bonds, angles, φ/ψ)
  → Engh–Huber + data-driven robust priors (median + MAD)
  → outlier detection (modified z-score)
  → context features (symmetry contact, B-factor z, exposure, resolution)
  → RandomForest attribution → report + plots
```

## Install (from GitHub)

```bash
pip install git+https://github.com/cheminfonotes/crystalaudit.git
```

Optional extras: `pip install "crystalaudit[app] @ git+https://github.com/cheminfonotes/crystalaudit.git"`
(`app` = Streamlit UI, `api` = FastAPI, `all` = both).

## Use it in 1–2 lines (local **or** Colab)

```python
import crystalaudit as ca

ca.analyze("1UBQ").show()        # PDB ID → inline plot + table
```

```python
ca.analyze("model.pdb").report   # local file → pandas DataFrame
```

In a notebook, a **bare expression** auto-renders a rich HTML summary:

```python
import crystalaudit as ca
ca.analyze("1UBQ")               # ← Colab shows chips + table automatically
```

### Colab one-liner

```python
!pip install -q git+https://github.com/cheminfonotes/crystalaudit.git
import crystalaudit as ca; ca.analyze("1UBQ").show()
```

## The `Result` object

`ca.analyze(...)` returns a `Result` with:

| Member | What |
|---|---|
| `.report` | DataFrame: residue, geometry, value, expected, `abs_z`, `attribution`, `confidence`, `explanation` |
| `.counts` | `{'biological': n, 'packing': n, 'error': n}` |
| `.plot()` | 3-panel matplotlib `Figure` (attribution, importance, cause scatter) |
| `.show()` | inline plot **and** table (notebook) |
| `.to_csv(path)` / `.to_html(path)` | export |
| `.summary()` | dict of headline numbers |

## Command line

```bash
crystalaudit 1UBQ
crystalaudit model.pdb --z 4.0 --csv out.csv --png overview.png
```

## Web app & REST API (optional)

```bash
pip install "crystalaudit[all] @ git+https://github.com/cheminfonotes/crystalaudit.git"
streamlit run app.py                 # browser UI: type an ID or drop a file
uvicorn api:app --reload             # REST: GET /analyze/{id}, POST /analyze
```

## Inputs accepted

`analyze()` takes a **4-char PDB ID** (downloaded from RCSB), a **file path**
(`.pdb` / `.cif`), raw **PDB/mmCIF text**, or a pre-loaded gemmi structure
(`st=`, `meta=`, `label=`).

## How attribution works (honestly)

There is no gold-standard labelled dataset for crystallographic outlier causes,
so labels are **bootstrapped with weak supervision** (domain rules: near a
symmetry mate → packing; high relative B / poor resolution → error; buried &
isolated → biological), then a RandomForest generalises them and returns
probabilities + feature importances. Swap in curated labels (e.g. PDB-REDO) to
upgrade accuracy.

## License

MIT — see [LICENSE](LICENSE).
