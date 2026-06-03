"""Plotting helpers for crystalaudit results."""
import numpy as np
import matplotlib.pyplot as plt

CMAP = {"packing": "#7b2ff7", "error": "#f72585", "biological": "#00b4a0"}


def overview_figure(summary):
    """3-panel overview: attribution counts, feature importance, cause scatter."""
    report = summary["report"]
    feat = summary.get("feat")
    imp = summary.get("importance")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"{summary['label']} — {summary['n_outliers']} outliers "
                 f"/ {summary['n_measurements']} measurements",
                 fontsize=13, fontweight="bold")

    if report.empty:
        for a in ax:
            a.axis("off")
        ax[1].text(0.5, 0.5, "No outliers above threshold",
                   ha="center", va="center", fontsize=12)
        plt.tight_layout()
        return fig

    vc = report["attribution"].value_counts()
    ax[0].bar(vc.index, vc.values, color=[CMAP.get(k, "#888") for k in vc.index])
    ax[0].set_title("Attribution"); ax[0].set_ylabel("count")

    if imp is not None:
        imp.sort_values().plot.barh(ax=ax[1], color="#00b4d8")
        ax[1].set_title("Feature importance")
    else:
        ax[1].axis("off"); ax[1].set_title("Feature importance (n/a)")

    if feat is not None and {"sym_contact", "b_z"} <= set(feat.columns):
        for a, g in feat.groupby("attribution"):
            ax[2].scatter(g["sym_contact"], g["b_z"], s=45, alpha=.8,
                          label=a, color=CMAP.get(a, "#888"))
        ax[2].set_xlabel("nearest symmetry contact (Å)")
        ax[2].set_ylabel("B-factor z-score")
        ax[2].set_title("Cause separation"); ax[2].legend()
    else:
        ax[2].axis("off")

    plt.tight_layout()
    return fig


def html_report(summary):
    """Self-contained HTML string for a structure's report."""
    report = summary["report"]
    rows = []
    for r in report.itertuples(index=False):
        c = CMAP.get(r.attribution, "#666")
        rows.append(
            f"<tr><td>{r.chain}{r.resnum} {r.resname}</td>"
            f"<td>{r.class_key}</td><td>{r.value:.3f}</td>"
            f"<td>{r.expected:.3f}</td><td>{r.abs_z:.1f}</td>"
            f"<td><b style='color:{c}'>{r.attribution}</b></td>"
            f"<td>{r.confidence:.0%}</td>"
            f"<td style='max-width:340px'>{r.explanation}</td></tr>")
    return (
        "<table style='border-collapse:collapse;width:100%;font-size:13px'>"
        "<tr style='text-align:left;color:#789'>"
        "<th>Residue</th><th>Geometry</th><th>Value</th><th>Expected</th>"
        "<th>|z|</th><th>Attribution</th><th>Conf.</th><th>Explanation</th></tr>"
        + "".join(rows) + "</table>")
