"""Command-line interface: `crystalaudit <PDB_ID|file> [--z 3.5] [--csv out.csv]`."""
import argparse
import sys
from . import analyze


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="crystalaudit",
        description="Flag protein geometry outliers and attribute each to "
                    "biological / packing / error.")
    p.add_argument("source", help="4-char PDB ID, or path to a .pdb/.cif file")
    p.add_argument("--z", type=float, default=3.5, help="outlier z-threshold")
    p.add_argument("--contact-radius", type=float, default=4.0,
                   help="symmetry-contact radius (Å)")
    p.add_argument("--csv", metavar="PATH", help="write report CSV")
    p.add_argument("--html", metavar="PATH", help="write report HTML")
    p.add_argument("--png", metavar="PATH", help="write overview figure PNG")
    args = p.parse_args(argv)

    try:
        res = analyze(args.source, z_outlier=args.z,
                      contact_radius=args.contact_radius)
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(f"{res.label}: {res.n_outliers} outliers / "
          f"{res.n_measurements} measurements  counts={res.counts}")
    if not res.report.empty:
        cols = ["chain", "resnum", "resname", "class_key",
                "value", "expected", "abs_z", "attribution", "confidence"]
        print(res.report[cols].head(15).to_string(index=False))
    if args.csv:
        print("wrote", res.to_csv(args.csv))
    if args.html:
        print("wrote", res.to_html(args.html))
    if args.png:
        res.plot().savefig(args.png, dpi=130)
        print("wrote", args.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
