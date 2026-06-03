"""Result object — rich display + convenience methods."""
import pandas as pd
from . import viz


class Result:
    """
    Wraps one structure's analysis.

    Attributes
    ----------
    report : DataFrame   per-outlier table (attribution, confidence, explanation)
    importance : Series  RandomForest feature importances (or None)
    meta : dict          resolution / r_free
    label : str          structure id / filename

    Methods
    -------
    plot()       -> matplotlib Figure (3-panel overview)
    show()       -> plot inline + display the table (notebook-friendly)
    to_csv(path) / to_html(path)
    """

    def __init__(self, summary: dict):
        self._s = summary
        self.report: pd.DataFrame = summary["report"]
        self.importance = summary.get("importance")
        self.meta = summary.get("meta", {})
        self.label = summary.get("label")
        self.n_outliers = summary.get("n_outliers", 0)
        self.n_measurements = summary.get("n_measurements", 0)

    # ---- numbers ----
    @property
    def counts(self) -> dict:
        if self.report.empty:
            return {}
        return self.report["attribution"].value_counts().to_dict()

    def summary(self) -> dict:
        return {"label": self.label, "resolution": self.meta.get("resolution"),
                "r_free": self.meta.get("r_free"),
                "n_measurements": self.n_measurements,
                "n_outliers": self.n_outliers, "attribution_counts": self.counts}

    # ---- visuals ----
    def plot(self):
        """Return the 3-panel overview matplotlib Figure."""
        return viz.overview_figure(self._s)

    def show(self):
        """Inline plot + table. Best inside Jupyter / Colab."""
        fig = self.plot()
        try:
            from IPython.display import display
            import matplotlib.pyplot as plt
            plt.show()
            if not self.report.empty:
                display(self.report)
        except Exception:
            fig.savefig(f"{self.label}_overview.png", dpi=130)
        return self

    # ---- export ----
    def to_csv(self, path=None):
        path = path or f"{self.label}_validation.csv"
        self.report.to_csv(path, index=False)
        return path

    def to_html(self, path=None):
        html = viz.html_report(self._s)
        if path:
            with open(path, "w") as f:
                f.write(html)
            return path
        return html

    # ---- rich auto-display (Colab/Jupyter shows this for a bare expression) ----
    def _repr_html_(self):
        s = self.summary()
        res = s["resolution"]
        head = (f"<h3 style='margin-bottom:2px'>🔬 {self.label}</h3>"
                f"<p style='color:#888;margin-top:0'>"
                f"{s['n_outliers']} outliers / {s['n_measurements']:,} measurements"
                f"{f' · {res:.2f} Å' if res else ''}</p>")
        if self.report.empty:
            return head + "<p>No outliers above threshold ✅</p>"
        chips = " ".join(
            f"<span style='background:{viz.CMAP.get(k,'#888')};color:#fff;"
            f"padding:2px 10px;border-radius:12px;margin-right:6px'>{k}: {v}</span>"
            for k, v in self.counts.items())
        return head + f"<p>{chips}</p>" + viz.html_report(self._s)

    def __repr__(self):
        return (f"<crystalaudit.Result {self.label}: "
                f"{self.n_outliers} outliers, counts={self.counts}>")
