"""
Phase 7 - Streamlit dashboard.

Reads the output of scripts/run_pipeline.py (data/pipeline_output/ranked_entities.csv)
and presents it as a ranked, filterable alert table with a click-through
interactive evidence graph per selected entity.

Run with:
    streamlit run cointrace/dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from cointrace import config
from cointrace.ingestion.pipeline import ingest_file
from cointrace.graph.builder import build_graph
from cointrace.graph.clustering import build_entity_clusters
from cointrace.explain.evidence import extract_evidence_subgraph, plain_language_explanation
from cointrace.feedback.store import latest_labels_only, load_feedback, record_feedback

st.set_page_config(page_title="CoinTrace", layout="wide")


@st.cache_data(show_spinner="Loading ranked entities...")
def load_ranked_entities() -> pd.DataFrame:
    path = config.PIPELINE_OUTPUT_DIR / "ranked_entities.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0)


@st.cache_resource(show_spinner="Rebuilding graph for evidence view...")
def load_graph_and_clusters():
    """Cached separately (and more expensively) from the ranked table -
    only needed when the user actually opens an entity's evidence view."""
    clean_df, _ = ingest_file(config.TRANSACTIONS_CSV, enrich=False)
    g = build_graph(clean_df)
    wallet_to_entity = build_entity_clusters(clean_df)
    return g, wallet_to_entity


def render_evidence_graph(g, wallets: list[str]):
    """Renders the evidence subgraph as a real interactive graph (drag,
    zoom, click-to-inspect a node, toggle layout) using streamlit-agraph.

    Keeps the canvas itself deliberately sparse (short id, no edge labels)
    and pushes every detail - amounts, timestamps, ASN org, etc. - into a
    full table underneath, plus hop-radius and node-type controls so the
    graph can be shrunk on demand instead of dumping everything on-canvas
    at once."""
    try:
        from streamlit_agraph import agraph, Node, Edge, Config
    except ImportError:
        st.warning("Install `streamlit-agraph` to see the interactive evidence graph "
                   "(`pip install streamlit-agraph`).")
        return

    key_suffix = "_".join(sorted(wallets))

    hops = st.slider("Hops from this entity", 1, 3, config.EVIDENCE_SUBGRAPH_HOPS,
                      key=f"hops_{key_suffix}",
                      help="Fewer hops = fewer nodes = an easier-to-read graph.")
    sub = extract_evidence_subgraph(g, wallets, hops=hops)
    if sub.number_of_nodes() == 0:
        st.info("No evidence subgraph available for this entity.")
        return

    kind_colors = {"Wallet": "#4C9AFF", "Transaction": "#F87171",
                   "IP": "#34D399", "ASN": "#FBBF24"}
    kind_shapes = {"Wallet": "dot", "Transaction": "square",
                   "IP": "triangle", "ASN": "diamond"}

    all_kinds = sorted({d.get("kind", "?") for _, d in sub.nodes(data=True)})
    legend_cols = st.columns(len(kind_colors) + 1)
    for col, (kind, color) in zip(legend_cols, kind_colors.items()):
        col.markdown(f"<span style='color:{color}; font-size:1.3em;'>&#9679;</span> {kind}",
                     unsafe_allow_html=True)
    legend_cols[-1].markdown("**Thick ring** = this entity's wallet")

    visible_kinds = st.multiselect(
        "Show node types", all_kinds, default=all_kinds, key=f"kinds_{key_suffix}",
        help="Untick a type to hide it from the graph and simplify the view - "
             "it stays available in the table below either way.",
    )
    kept = {n for n, d in sub.nodes(data=True) if d.get("kind") in visible_kinds}
    view = sub.subgraph(kept).copy() if kept != set(sub.nodes) else sub
    if view.number_of_nodes() == 0:
        st.info("All node types are hidden — tick at least one above to see the graph.")
        return

    degrees = dict(view.to_undirected().degree())
    max_degree = max(degrees.values(), default=1) or 1

    nodes = []
    detail_rows = []
    for node, data in view.nodes(data=True):
        kind = data.get("kind", "?")
        raw_id = node.split(":", 1)[-1]
        is_seed = kind == "Wallet" and raw_id in wallets

        size = 14 + 10 * (degrees.get(node, 1) / max_degree) + (8 if is_seed else 0)
        nodes.append(Node(
            id=node,
            label=raw_id[:8] + ("…" if len(raw_id) > 8 else ""),
            size=size,
            shape=kind_shapes.get(kind, "dot"),
            color=kind_colors.get(kind, "#999999"),
            title=f"{kind}: {raw_id}",  # short hover hint; full detail is in the table
            borderWidth=4 if is_seed else 1,
            font={"color": "#e5e7eb", "size": 11},
        ))

        row = {"kind": kind, "id": raw_id, "this entity's wallet": is_seed}
        if kind == "Transaction":
            row["amount_btc"] = data.get("amount_btc")
            row["timestamp"] = data.get("timestamp")
        elif kind == "ASN":
            row["asn"] = data.get("asn", raw_id)
        detail_rows.append(row)

    edges = []
    for u, v, data in view.edges(data=True):
        edges.append(Edge(source=u, target=v, label=data.get("kind", ""), color="#6b7280"))

    layout = st.radio("Layout", ["Force-directed", "Hierarchical"], horizontal=True,
                       key=f"layout_{key_suffix}")

    graph_config = Config(
        width="100%",
        height=520,
        directed=True,
        physics=True,
        hierarchical=(layout == "Hierarchical"),
        nodeHighlightBehavior=True,
        highlightColor="#F7A72A",
        collapsible=False,
        node={"labelProperty": "label"},
        link={"labelProperty": "label", "renderLabel": False},
    )

    clicked_node = agraph(nodes=nodes, edges=edges, config=graph_config)
    if clicked_node:
        st.caption(f"Selected: **{clicked_node}** — {dict(view.nodes.get(clicked_node, {}))}")

    with st.expander(f"Full detail table ({len(detail_rows)} nodes in this view)"):
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, height=300)


def main():
    st.title("CoinTrace — Ranked Investigative Leads")
    st.caption("Offline AI-powered monitoring & analysis of Bitcoin transaction traffic")

    ranked = load_ranked_entities()
    if ranked.empty:
        st.error(
            "No pipeline output found. Run these first:\n\n"
            "```\npython scripts/generate_data.py\npython scripts/run_pipeline.py\n```"
        )
        return

    with st.sidebar:
        st.header("Filters")
        min_score = st.slider("Minimum risk score", 0.0, 1.0, config.RISK_ALERT_THRESHOLD, 0.01)
        typology_options = sorted(
            {t.strip() for reasons in ranked["motif_reasons"].dropna()
             for t in reasons.split(",")}
        )
        selected_typologies = st.multiselect("Typology filter", typology_options)
        show_all_columns = st.checkbox("Show all feature columns", value=False)

        st.divider()
        st.header("Feedback / learning")
        n_labeled = len(latest_labels_only(load_feedback()))
        st.caption(f"{n_labeled} entities labeled so far.")
        if config.LEARNED_WEIGHTS_JSON.exists():
            st.caption("Fusion weights have been re-fit from feedback at least once — "
                       "current ranking uses learned weights, not the static defaults.")
        else:
            st.caption(f"Need {config.MIN_FEEDBACK_FOR_RETRAIN} labeled entities "
                       "(both classes) before weights start adapting.")

    filtered = ranked[ranked["risk_score"] >= min_score]
    if selected_typologies:
        pattern = "|".join(selected_typologies)
        filtered = filtered[filtered["motif_reasons"].str.contains(pattern, na=False)]

    st.subheader(f"Ranked alerts ({len(filtered):,} of {len(ranked):,} entities)")

    default_cols = ["risk_score", "isolation_forest_score", "motif_score", "motif_reasons"]
    if "autoencoder_score" in filtered.columns:
        default_cols.insert(2, "autoencoder_score")
    display_cols = list(filtered.columns) if show_all_columns else default_cols

    st.dataframe(filtered[display_cols], use_container_width=True, height=350)

    if "latitude" in ranked.columns and "longitude" in ranked.columns:
        geo_points = filtered.dropna(subset=["latitude", "longitude"])
        st.subheader(f"Geographic footprint ({len(geo_points):,} located entities)")
        if geo_points.empty:
            st.caption("No entities in the current filter have a resolved broadcast location.")
        else:
            st.map(geo_points, latitude="latitude", longitude="longitude", size=25)
    else:
        st.caption(
            "No location data in this run — add GeoLite2-City.mmdb and "
            "GeoLite2-ASN.mmdb to data/geoip/ and re-run scripts/run_pipeline.py."
        )

    st.divider()
    st.subheader("Inspect an entity")
    selected_entity = st.selectbox("Entity ID", filtered.index.tolist())

    if selected_entity:
        row = ranked.loc[selected_entity]
        has_geo = "city" in ranked.columns and pd.notna(row.get("city"))
        cols = st.columns(4 if has_geo else 3)
        cols[0].metric("Risk score", f"{row['risk_score']:.3f}")
        cols[1].metric("Transactions", int(row.get("tx_count", 0)))
        cols[2].metric("Distinct IPs", int(row.get("distinct_ip_count", 0)))
        if has_geo:
            cols[3].metric("Location", f"{row['city']}, {row.get('country', '?')}")

        explanation = plain_language_explanation(
            entity_id=selected_entity,
            motif_reasons=row.get("motif_reasons", "none"),
            distinct_ip_count=int(row.get("distinct_ip_count", 0)),
            tx_count=int(row.get("tx_count", 0)),
            round_trip_time_mean_seconds=float(row.get("round_trip_time_mean", 0)),
        )
        st.info(explanation)

        st.markdown("**Was this call right?** Your answer is saved locally and used to "
                     "re-fit the scoring weights next time the pipeline runs — see "
                     "`cointrace/feedback/`.")
        fb_cols = st.columns([1, 1, 3])
        if fb_cols[0].button("✅ Confirm illicit", key=f"fb_illicit_{selected_entity}"):
            record_feedback(selected_entity, is_illicit=True)
            st.success(f"Recorded {selected_entity} as confirmed illicit.")
        if fb_cols[1].button("❌ False positive", key=f"fb_benign_{selected_entity}"):
            record_feedback(selected_entity, is_illicit=False)
            st.success(f"Recorded {selected_entity} as a false positive.")

        with st.expander("Evidence subgraph (click to load)"):
            g, wallet_to_entity = load_graph_and_clusters()
            wallets = [w for w, e in wallet_to_entity.items() if e == selected_entity]
            render_evidence_graph(g, wallets)


if __name__ == "__main__":
    main()
