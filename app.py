"""
app.py — Streamlit UI for crystalaudit.

Run:  streamlit run app.py
Enter a 4-char PDB ID OR upload a .pdb/.cif file → anomaly attribution + plots.
"""
import streamlit as st
import crystalaudit as ca
from crystalaudit.viz import CMAP

st.set_page_config(page_title="crystalaudit", page_icon="🔬", layout="wide")

st.markdown(
    "<h1 style='margin-bottom:0'>🔬 crystalaudit</h1>"
    "<p style='color:#888;margin-top:4px'>Existing tools <i>flag</i> geometry "
    "outliers. This one explains <b>why</b> — biological strain, crystal packing, "
    "or refinement error.</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Input")
    mode = st.radio("Source", ["PDB ID", "Upload file"], horizontal=True)
    pdb_id = uploaded = None
    if mode == "PDB ID":
        pdb_id = st.text_input("4-char PDB ID", "1UBQ", max_chars=4).strip()
    else:
        uploaded = st.file_uploader("Structure file", type=["pdb", "cif", "ent"])
    st.divider()
    st.header("Parameters")
    z_thr = st.slider("Outlier z-threshold", 2.5, 6.0, 3.5, 0.1)
    contact_r = st.slider("Symmetry-contact radius (Å)", 3.0, 6.0, 4.0, 0.1)
    run = st.button("Analyze ▶", type="primary", use_container_width=True)

if not run:
    st.info("Enter a PDB ID or upload a structure, then press **Analyze ▶**.")
    st.stop()

try:
    with st.spinner("Running pipeline…"):
        if mode == "Upload file":
            if not uploaded:
                st.error("Please upload a file."); st.stop()
            stc, meta, label = ca.load_bytes(uploaded.getvalue(), uploaded.name)
            res = ca.analyze(st=stc, meta=meta, label=label,
                             z_outlier=z_thr, contact_radius=contact_r)
        else:
            if not pdb_id:
                st.error("Please enter a PDB ID."); st.stop()
            res = ca.analyze(pdb_id, z_outlier=z_thr, contact_radius=contact_r)
except Exception as e:
    st.error(f"Failed: {type(e).__name__}: {e}"); st.stop()

s = res.summary()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Structure", res.label)
c2.metric("Measurements", f"{res.n_measurements:,}")
c3.metric("Outliers", res.n_outliers)
c4.metric("Resolution", f"{s['resolution']:.2f} Å" if s["resolution"] else "—")

if res.report.empty:
    st.success("No geometry outliers above threshold. ✅"); st.stop()

st.subheader("Attribution summary")
cols = st.columns(len(res.counts))
for col, (k, v) in zip(cols, res.counts.items()):
    col.markdown(
        f"<div style='border-left:4px solid {CMAP.get(k,'#888')};padding-left:10px'>"
        f"<b style='color:{CMAP.get(k,'#888')}'>{k}</b><br>"
        f"<span style='font-size:26px'>{v}</span></div>", unsafe_allow_html=True)

st.subheader("Flagged outliers")
st.dataframe(res.report, use_container_width=True, height=380)

st.subheader("Visual overview")
st.pyplot(res.plot())

st.subheader("Export")
st.download_button("⬇ Download report (CSV)",
                   res.report.to_csv(index=False).encode(),
                   file_name=f"{res.label}_validation.csv", mime="text/csv")
