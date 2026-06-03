"""
crystalaudit — Automated Crystallographic Validation & Anomaly Attribution.

Existing tools *flag* protein-geometry outliers; crystalaudit explains *why*
each one is there: biological strain, crystal packing, or refinement error.

Quickstart
----------
    import crystalaudit as ca
    ca.analyze("1UBQ").show()        # PDB ID -> inline plot + table
    ca.analyze("model.pdb").report   # local file -> DataFrame

In a notebook a bare `ca.analyze("1UBQ")` renders a rich HTML summary.
"""
from ._core import Config, run, load_input, load_bytes
from .result import Result

__version__ = "0.1.0"
__all__ = ["analyze", "Config", "Result", "load_bytes"]


def analyze(source=None, *, st=None, meta=None, label=None,
            z_outlier=3.5, contact_radius=4.0, cfg=None):
    """
    Analyze ONE structure and return a :class:`Result`.

    Parameters
    ----------
    source : str
        A 4-character PDB ID (downloaded from RCSB), a path to a local
        ``.pdb`` / ``.cif`` file, or raw PDB/mmCIF text.
    st, meta, label :
        Optionally pass a pre-loaded gemmi structure instead of `source`.
    z_outlier : float
        Modified z-score threshold for flagging (default 3.5).
    contact_radius : float
        Symmetry-contact search radius in Å (default 4.0).

    Returns
    -------
    Result
        Has ``.report`` (DataFrame), ``.plot()``, ``.show()``, ``.to_csv()``,
        ``.to_html()`` and rich notebook display.
    """
    cfg = cfg or Config(z_outlier=z_outlier, contact_radius=contact_radius)
    summary = run(source, st=st, meta=meta, label=label, cfg=cfg)
    return Result(summary)
