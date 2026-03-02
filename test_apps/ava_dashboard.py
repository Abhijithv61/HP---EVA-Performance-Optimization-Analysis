
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AVA Bot · Performance Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0f172a; }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  .metric-card {
    background: linear-gradient(135deg,#1e293b,#0f172a);
    border: 1px solid #334155; border-radius: 12px;
    padding: 20px 24px; margin-bottom: 8px;
  }
  .metric-card .label { color:#94a3b8; font-size:13px; font-weight:600;
    text-transform:uppercase; letter-spacing:.6px; margin-bottom:4px; }
  .metric-card .value { color:#f1f5f9; font-size:32px; font-weight:700; line-height:1; }
  .metric-card .delta { font-size:13px; margin-top:4px; }
  .delta-bad  { color:#f87171; }
  .delta-good { color:#4ade80; }
  .delta-warn { color:#fbbf24; }
  .section-title {
    font-size:20px; font-weight:700; color:#e2e8f0;
    border-left:4px solid #6366f1; padding-left:12px;
    margin: 28px 0 16px 0;
  }
  .insight-box {
    background:#1e293b; border-left:4px solid #6366f1;
    border-radius:8px; padding:14px 18px; margin:12px 0;
    color:#cbd5e1; font-size:14px; line-height:1.6;
  }
  .insight-box b { color:#a5b4fc; }
  .warn-box {
    background:#1e293b; border-left:4px solid #f59e0b;
    border-radius:8px; padding:14px 18px; margin:12px 0;
    color:#fcd34d; font-size:14px;
  }
  .crit-box {
    background:#1e293b; border-left:4px solid #ef4444;
    border-radius:8px; padding:14px 18px; margin:12px 0;
    color:#fca5a5; font-size:14px;
  }
  .tab-content { padding-top: 12px; }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    background: #1e293b; border-radius: 8px;
    color: #94a3b8; padding: 8px 20px;
  }
  .stTabs [aria-selected="true"] {
    background: #6366f1 !important; color: white !important;
  }
</style>
""", unsafe_allow_html=True)

PALETTE = px.colors.qualitative.Plotly
CHART_BG = "rgba(0,0,0,0)"
GRID_COLOR = "#1e293b"
FONT_COLOR = "#e2e8f0"

def chart_layout(fig, title="", subtitle=""):
    full_title = title
    if subtitle:
        full_title += f"<br><span style='font-size:13px;color:#94a3b8;font-weight:400'>{subtitle}</span>"
    fig.update_layout(
        title={"text": full_title, "font": {"color": FONT_COLOR, "size": 17}},
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        font={"color": FONT_COLOR, "size": 12},
        margin=dict(t=80, b=50, l=50, r=20),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": FONT_COLOR}),
        hoverlabel=dict(bgcolor="#1e293b", font_color="#f1f5f9")
    )
    return fig

ABBREV = {
    "case_create": "Case Create",
    "case_update": "Case Update",
    "defects_search": "Defects Search",
    "firmware_recommendation": "Firmware Rec.",
    "license_management": "License Mgmt",
    "troubleshooting": "Troubleshooting"
}
SOURCE_ORDER = list(ABBREV.keys())
COLOR_MAP = {ABBREV[s]: PALETTE[i] for i, s in enumerate(SOURCE_ORDER)}

# ── Load data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("dashboard_interactions.csv", parse_dates=["timestamp"])
    tdf = pd.read_csv("dashboard_traces.csv", parse_dates=["top_timestamp"])
    df["source_label"] = df["source"].map(ABBREV)
    tdf["source_label"] = tdf["source"].map(ABBREV)
    df["tokens_per_sec"] = df["totalTokens"] / df["latency"]
    df["cost_per_1k_tokens"] = (df["totalCost"] / df["totalTokens"]) * 1000
    df["output_ratio"] = df["outputTokens"] / df["inputTokens"]
    df["latency_per_1k_tokens"] = (df["latency"] / df["totalTokens"]) * 1000
    return df, tdf

df_full, tdf_full = load_data()

# ── Sidebar filters ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 AVA Bot Analytics")
    st.markdown("---")
    st.markdown("### Filters")

    selected_sources = st.multiselect(
        "Select Sources",
        options=SOURCE_ORDER,
        default=SOURCE_ORDER,
        format_func=lambda x: ABBREV[x]
    )

    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("", [
        "📊 Executive Summary",
        "⚡ Latency Analysis",
        "🔬 Pipeline Deep Dive",
        "💰 Cost & Tokens",
        "🔍 Metadata Intelligence",
        "🚨 Optimization Playbook"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#475569;font-size:12px'>"
        "AVA Bot · Staging Environment<br>Data: Feb 13–17, 2026<br>22 interactions · 6 sources"
        "</div>", unsafe_allow_html=True
    )

# ── Filter dataframes ────────────────────────────────────────────────────────
df = df_full[df_full["source"].isin(selected_sources)].copy()
tdf = tdf_full[tdf_full["source"].isin(selected_sources)].copy()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE SUMMARY
# ════════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Summary":
    st.markdown("# 📊 Executive Summary")
    st.markdown(
        "<div class='insight-box'>Performance analysis of the <b>AVA Bot</b> across "
        "<b>6 use-case sources</b> on the <b>staging environment</b>, "
        "covering <b>22 interactions</b> logged between Feb 13–17, 2026. "
        "All timings are end-to-end wall-clock latencies.</div>",
        unsafe_allow_html=True
    )

    # KPI Row 1
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metrics = [
        (c1, "Total Interactions", len(df), None, None),
        (c2, "Avg Latency", f"{df['latency'].mean():.1f}s", "Global", None),
        (c3, "Median Latency", f"{df['latency'].median():.1f}s", "", None),
        (c4, "Max Latency", f"{df['latency'].max():.1f}s", "Troubleshooting outlier", "bad"),
        (c5, "Total Cost (Session)", f"${df['totalCost'].sum():.3f}", "", None),
        (c6, "Avg Cost/Interaction", f"${df['totalCost'].mean():.4f}", "", None),
    ]
    for col, label, value, note, kind in metrics:
        delta_class = f"delta-{kind}" if kind else "delta-warn"
        note_html = f"<div class='delta {delta_class}'>{note}</div>" if note else ""
        col.markdown(
            f"<div class='metric-card'><div class='label'>{label}</div>"
            f"<div class='value'>{value}</div>{note_html}</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Latency by Source</div>", unsafe_allow_html=True)
        src_stats = df.groupby("source_label")["latency"].agg(["mean","std","count"]).reset_index()
        src_stats["se"] = src_stats["std"] / np.sqrt(src_stats["count"])
        src_stats = src_stats.sort_values("mean", ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=src_stats["mean"], y=src_stats["source_label"], orientation="h",
            error_x=dict(type="data", array=src_stats["se"].fillna(0), visible=True,
                        color="#94a3b8", thickness=2),
            text=[f"{v:.1f}s" for v in src_stats["mean"]], textposition="outside",
            marker_color=[COLOR_MAP.get(l, "#6366f1") for l in src_stats["source_label"]],
            hovertemplate="%{y}: %{x:.2f}s<extra></extra>"
        ))
        fig = chart_layout(fig, "Avg Response Latency", "Error bars = standard error of mean")
        fig.update_xaxes(title_text="Avg Latency (s)")
        fig.update_yaxes(title_text="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Efficiency Ranking</div>", unsafe_allow_html=True)
        eff = df.groupby("source_label")["latency_per_1k_tokens"].mean().reset_index()
        eff = eff.sort_values("latency_per_1k_tokens", ascending=True)
        fig2 = px.bar(
            eff, x="latency_per_1k_tokens", y="source_label", orientation="h",
            text=[f"{v:.2f}s" for v in eff["latency_per_1k_tokens"]],
            color="source_label", color_discrete_map=COLOR_MAP
        )
        fig2.update_traces(textposition="outside", cliponaxis=False, showlegend=False)
        fig2 = chart_layout(fig2, "Processing Efficiency", "Latency per 1,000 tokens — lower is better")
        fig2.update_xaxes(title_text="s per 1k Tokens")
        fig2.update_yaxes(title_text="")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("<div class='section-title'>Source Performance Scorecard</div>", unsafe_allow_html=True)

    scorecard_data = []
    for src in selected_sources:
        sub = df[df["source"] == src]
        if len(sub) == 0: continue
        label = ABBREV[src]
        scorecard_data.append({
            "Source": label,
            "Interactions": len(sub),
            "Avg Latency (s)": f"{sub['latency'].mean():.1f}",
            "Max Latency (s)": f"{sub['latency'].max():.1f}",
            "Avg Tokens": f"{sub['totalTokens'].mean():,.0f}",
            "Avg Cost ($)": f"${sub['totalCost'].mean():.4f}",
            "Session Cost ($)": f"${sub['totalCost'].sum():.3f}",
            "s / 1k Tokens": f"{sub['latency_per_1k_tokens'].mean():.2f}",
        })
    sc_df = pd.DataFrame(scorecard_data)
    st.dataframe(sc_df, use_container_width=True, hide_index=True)

    # Statistical significance note
    st.markdown(
        "<div class='insight-box'>"
        "📈 <b>Statistical Significance:</b> Latency difference between <b>case_create</b> and "
        "<b>license_management</b> is statistically significant "
        "(Welch's t-test: t=2.74, <b>p=0.023</b>). "
        "<b>case_update vs license_management</b> is highly significant (t=10.6, <b>p=0.0002</b>). "
        "Token count is a significant predictor of latency "
        "(Spearman r=0.68, <b>p=0.0005</b>) and cost (r=0.99, <b>p&lt;0.0001</b>)."
        "</div>", unsafe_allow_html=True
    )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — LATENCY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Latency Analysis":
    st.markdown("# ⚡ Latency Analysis")

    tab1, tab2, tab3 = st.tabs(["Distribution", "Session Turn Growth", "Outlier Inspector"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(
                df, x="source_label", y="latency", color="source_label",
                color_discrete_map=COLOR_MAP, points="all",
                labels={"source_label": "Source", "latency": "Latency (s)"}
            )
            fig.update_traces(jitter=0.4, pointpos=-1.8)
            fig = chart_layout(fig, "Latency Distribution by Source", "All data points plotted")
            fig.update_xaxes(title_text="Source")
            fig.update_yaxes(title_text="Latency (s)")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.violin(
                df, y="latency", x="source_label", color="source_label",
                color_discrete_map=COLOR_MAP, box=True, points="all",
                labels={"source_label": "Source", "latency": "Latency (s)"}
            )
            fig2 = chart_layout(fig2, "Latency Spread (Violin)", "Width = density of observations")
            fig2.update_xaxes(title_text="Source")
            fig2.update_yaxes(title_text="Latency (s)")
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        # Pairwise t-test table
        st.markdown("<div class='section-title'>Pairwise Statistical Significance (Welch's t-test)</div>", unsafe_allow_html=True)
        sources_avail = df["source"].unique()
        ttest_rows = []
        for i, s1 in enumerate(sources_avail):
            for j, s2 in enumerate(sources_avail):
                if j <= i: continue
                g1 = df[df["source"]==s1]["latency"].dropna().values
                g2 = df[df["source"]==s2]["latency"].dropna().values
                if len(g1) >= 2 and len(g2) >= 2:
                    t, p = stats.ttest_ind(g1, g2, equal_var=False)
                    ttest_rows.append({
                        "Source A": ABBREV[s1], "Source B": ABBREV[s2],
                        "t-stat": round(t, 3), "p-value": round(p, 4),
                        "Significant (p<0.05)": "✅ Yes" if p < 0.05 else "❌ No"
                    })
        if ttest_rows:
            tt_df = pd.DataFrame(ttest_rows)
            st.dataframe(tt_df, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("<div class='section-title'>Token Growth Per Session Turn</div>", unsafe_allow_html=True)
        fig3 = go.Figure()
        for src in selected_sources:
            sub = df[df["source"]==src].sort_values("timestamp").reset_index(drop=True)
            label = ABBREV[src]
            fig3.add_trace(go.Scatter(
                x=list(range(1, len(sub)+1)), y=sub["totalTokens"],
                mode="lines+markers", name=label,
                line=dict(color=COLOR_MAP.get(label), width=2),
                marker=dict(size=8),
                hovertemplate=f"{label}<br>Turn %{{x}}: %{{y:,}} tokens<extra></extra>"
            ))
        fig3 = chart_layout(fig3, "Token Count Growth Across Session Turns",
                            "Growing context window → increasing latency & cost per turn")
        fig3.update_xaxes(title_text="Session Turn", dtick=1)
        fig3.update_yaxes(title_text="Total Tokens")
        fig3.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5))
        st.plotly_chart(fig3, use_container_width=True)

        growth_data = []
        for src in selected_sources:
            sub = df[df["source"]==src].sort_values("timestamp").reset_index(drop=True)
            if len(sub) > 1:
                tokens = sub["totalTokens"].tolist()
                growth = np.mean([(tokens[i]-tokens[i-1])/tokens[i-1]*100 for i in range(1,len(tokens))])
                growth_data.append({"Source": ABBREV[src], "Avg Token Growth/Turn": f"+{growth:.1f}%",
                                    "Max Tokens": f"{max(tokens):,}", "Min Tokens": f"{min(tokens):,}"})
        if growth_data:
            st.dataframe(pd.DataFrame(growth_data), use_container_width=True, hide_index=True)

        st.markdown(
            "<div class='warn-box'>⚠️ <b>firmware_recommendation</b> grows +70.3% per turn — "
            "fastest context explosion. Implement rolling summarization to cap token growth.</div>",
            unsafe_allow_html=True
        )

    with tab3:
        st.markdown("<div class='section-title'>Outlier Inspector</div>", unsafe_allow_html=True)
        global_mean = df["latency"].mean()
        global_std  = df["latency"].std()
        threshold   = global_mean + 2 * global_std
        outliers = df[df["latency"] > threshold][["source_label","sessionId","latency","totalTokens","totalCost"]]

        col1, col2, col3 = st.columns(3)
        col1.metric("Global Mean Latency", f"{global_mean:.2f}s")
        col2.metric("Std Dev", f"{global_std:.2f}s")
        col3.metric("Outlier Threshold (μ+2σ)", f"{threshold:.2f}s")

        fig4 = px.scatter(
            df, x="timestamp", y="latency", color="source_label",
            color_discrete_map=COLOR_MAP, size="totalTokens", size_max=20,
            labels={"timestamp":"Timestamp","latency":"Latency (s)","source_label":"Source"}
        )
        fig4.add_hline(y=threshold, line_dash="dash", line_color="#ef4444",
                       annotation_text=f"Outlier threshold: {threshold:.1f}s",
                       annotation_font_color="#ef4444")
        fig4 = chart_layout(fig4, "Latency Over Time", "Bubble size = token count | Red line = outlier threshold")
        fig4.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5))
        st.plotly_chart(fig4, use_container_width=True)

        if len(outliers) > 0:
            st.markdown(f"<div class='crit-box'>🚨 {len(outliers)} outlier interaction(s) detected above {threshold:.1f}s</div>", unsafe_allow_html=True)
            st.dataframe(outliers, use_container_width=True, hide_index=True)
        else:
            st.markdown("<div class='insight-box'>✅ No outliers in current filter selection.</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PIPELINE DEEP DIVE
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Pipeline Deep Dive":
    st.markdown("# 🔬 Pipeline Deep Dive")

    STEP_TYPES = ["guardrail_pre","routing_classifier","action_group",
                  "agent_collaborator","routing_collaborator_roundtrip",
                  "llm_orchestration","guardrail_post"]
    STEP_LABELS = {
        "guardrail_pre": "Guardrail Pre",
        "routing_classifier": "Routing Classifier",
        "action_group": "Action Group API",
        "agent_collaborator": "Agent Collaborator",
        "routing_collaborator_roundtrip": "Sub-Agent Roundtrip",
        "llm_orchestration": "LLM Orchestration",
        "guardrail_post": "Guardrail Post"
    }

    tab1, tab2, tab3 = st.tabs(["Step Decomposition", "LLM Analysis", "Sub-Agent Roundtrips"])

    with tab1:
        avg_step = tdf[tdf["event_type"].isin(STEP_TYPES)].groupby(
            ["source","event_type"])["step_ms"].mean().reset_index()
        avg_step["step_s"] = avg_step["step_ms"] / 1000
        avg_step["source_label"] = avg_step["source"].map(ABBREV)

        fig = go.Figure()
        for st_type in STEP_TYPES:
            sub = avg_step[avg_step["event_type"]==st_type].set_index("source")
            vals = []
            for src in selected_sources:
                vals.append(sub.loc[src, "step_s"] if src in sub.index else 0)
            fig.add_trace(go.Bar(
                name=STEP_LABELS[st_type],
                x=[ABBREV[s] for s in selected_sources],
                y=vals
            ))
        fig = chart_layout(fig, "Avg Latency Stack by Pipeline Step",
                           "Each segment = avg time spent in that step per interaction")
        fig.update_layout(barmode="stack",
                          legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
        fig.update_xaxes(title_text="Source")
        fig.update_yaxes(title_text="Avg Time (s)")
        st.plotly_chart(fig, use_container_width=True)

        # LLM vs Non-LLM split
        col1, col2 = st.columns(2)
        llm_data = {"case_create":69.8,"case_update":54.3,"defects_search":28.3,
                    "firmware_recommendation":52.0,"license_management":49.3,"troubleshooting":56.8}
        avail_src = [s for s in selected_sources if s in llm_data]
        with col1:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name="LLM Time",
                x=[ABBREV[s] for s in avail_src], y=[llm_data[s] for s in avail_src]))
            fig2.add_trace(go.Bar(name="Non-LLM Overhead",
                x=[ABBREV[s] for s in avail_src], y=[100-llm_data[s] for s in avail_src]))
            fig2 = chart_layout(fig2, "LLM vs Non-LLM Share", "% of total wall-clock latency")
            fig2.update_layout(barmode="stack")
            fig2.update_xaxes(title_text="Source")
            fig2.update_yaxes(title_text="% of Latency")
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            step_summary = tdf[tdf["event_type"].isin(STEP_TYPES)].groupby(
                ["source_label","event_type"])["step_ms"].mean().unstack(fill_value=0)
            step_summary.columns = [STEP_LABELS.get(c,c) for c in step_summary.columns]
            step_summary_s = (step_summary / 1000).round(2)
            avail_labels = [ABBREV[s] for s in selected_sources if ABBREV[s] in step_summary_s.index]
            if avail_labels:
                heatmap_data = step_summary_s.loc[avail_labels]
                fig3 = go.Figure(data=go.Heatmap(
                    z=heatmap_data.values,
                    x=heatmap_data.columns.tolist(),
                    y=heatmap_data.index.tolist(),
                    colorscale="YlOrRd",
                    text=[[f"{v:.2f}s" for v in row] for row in heatmap_data.values],
                    texttemplate="%{text}", textfont={"size":10},
                    hoverongaps=False
                ))
                fig3 = chart_layout(fig3, "Step Latency Heatmap (seconds)", "Red = slower")
                fig3.update_xaxes(title_text="Pipeline Step", tickangle=-30)
                fig3.update_yaxes(title_text="Source")
                st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        llm_df = tdf[tdf["event_type"]=="llm_orchestration"].copy()

        with col1:
            fig4 = px.box(llm_df, x="source_label", y="step_ms", color="source_label",
                          color_discrete_map=COLOR_MAP, points="all",
                          labels={"source_label":"Source","step_ms":"LLM Time (ms)"})
            fig4 = chart_layout(fig4, "LLM Orchestration Latency Distribution", "All calls plotted")
            fig4.update_layout(showlegend=False)
            fig4.update_xaxes(title_text="Source")
            fig4.update_yaxes(title_text="LLM Time (ms)")
            st.plotly_chart(fig4, use_container_width=True)

        with col2:
            llm_depth = {"License Mgmt":1.6,"Case Create":2.8,"Case Update":4.5,
                         "Firmware Rec.":1.5,"Defects Search":2.0,"Troubleshooting":2.0}
            llm_ms = {"License Mgmt":4370,"Case Create":6639,"Case Update":4462,
                      "Firmware Rec.":6693,"Defects Search":11553,"Troubleshooting":18999}
            avail_labels2 = [ABBREV[s] for s in selected_sources]
            d_labels = [l for l in avail_labels2 if l in llm_depth]
            fig5 = go.Figure()
            fig5.add_trace(go.Bar(
                name="Avg LLM Calls", x=d_labels, y=[llm_depth[l] for l in d_labels],
                text=[f"{llm_depth[l]:.1f}" for l in d_labels], textposition="outside",
                marker_color=[COLOR_MAP.get(l,"#6366f1") for l in d_labels]
            ))
            fig5 = chart_layout(fig5, "Avg LLM Calls per Interaction",
                                "case_update fires 4.5 LLM calls — most complex flow")
            fig5.update_xaxes(title_text="Source")
            fig5.update_yaxes(title_text="Avg LLM Calls")
            st.plotly_chart(fig5, use_container_width=True)

        llm_tok = tdf[tdf["event_type"]=="llm_orchestration"].dropna(subset=["input_tokens"])
        if not llm_tok.empty:
            st.markdown("<div class='section-title'>LLM Token Load per Call</div>", unsafe_allow_html=True)
            fig6 = px.scatter(llm_tok, x="input_tokens", y="step_ms", color="source_label",
                              size="output_tokens", size_max=20,
                              color_discrete_map=COLOR_MAP,
                              labels={"input_tokens":"Input Tokens","step_ms":"LLM Time (ms)","source_label":"Source"},
                              trendline="ols")
            fig6 = chart_layout(fig6, "Input Tokens vs LLM Latency (bubble=output tokens)",
                                "Higher input load → longer LLM call")
            fig6.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5))
            st.plotly_chart(fig6, use_container_width=True)

    with tab3:
        st.markdown("<div class='section-title'>Sub-Agent Collaborator Round-Trip Latency</div>", unsafe_allow_html=True)
        collab_data = [
            ("License Assist", "License Mgmt", 9977),
            ("SupportCase Mgmt", "Case Create", 11332),
            ("Firmware Assist", "Firmware Rec.", 14871),
            ("SupportCase Mgmt", "Case Update", 20488),
            ("Case-Mgmt2", "Case Create", 23232),
            ("Case-Mgmt2", "Case Update", 32530),
            ("Defects-Mgmt", "Defects Search", 75946),
            ("Defects-Mgmt", "Troubleshooting", 129347),
        ]
        avail_label_set = {ABBREV[s] for s in selected_sources}
        collab_filt = [(c, src, ms) for c, src, ms in collab_data if src in avail_label_set]
        if collab_filt:
            c_df = pd.DataFrame(collab_filt, columns=["Collaborator","Source","Roundtrip_ms"])
            c_df["label"] = c_df["Collaborator"] + " → " + c_df["Source"]
            c_df = c_df.sort_values("Roundtrip_ms")
            fig7 = px.bar(c_df, x="Roundtrip_ms", y="label", orientation="h",
                          color="Source", color_discrete_map=COLOR_MAP,
                          text=[f"{v/1000:.1f}s" for v in c_df["Roundtrip_ms"]],
                          labels={"Roundtrip_ms":"Roundtrip (ms)","label":""})
            fig7.update_traces(textposition="outside", cliponaxis=False)
            fig7 = chart_layout(fig7, "Sub-Agent Roundtrip Latency",
                                "Defects-Mgmt in troubleshooting = 129s — critical bottleneck")
            fig7.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5))
            fig7.update_xaxes(title_text="Roundtrip (ms)")
            st.plotly_chart(fig7, use_container_width=True)

        st.markdown(
            "<div class='crit-box'>🔴 <b>Defects-Management-Agent</b> is a single point of failure. "
            "It accounts for 75.9s roundtrip in defects_search and 129.3s in troubleshooting. "
            "Root cause: synchronous blocking sub-agent with cascading internal LLM calls. "
            "Fix: async dispatch + timeout + semantic cache.</div>",
            unsafe_allow_html=True
        )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — COST & TOKENS
# ════════════════════════════════════════════════════════════════════════════
elif page == "💰 Cost & Tokens":
    st.markdown("# 💰 Cost & Token Analysis")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Input Tokens", f"{df['inputTokens'].sum():,}")
    c2.metric("Total Output Tokens", f"{df['outputTokens'].sum():,}")
    c3.metric("Total Session Cost", f"${df['totalCost'].sum():.4f}")
    c4.metric("Avg Output Ratio", f"{df['output_ratio'].mean():.3f}")

    col1, col2 = st.columns(2)
    with col1:
        cost_avg = df.groupby("source_label")[["inputCost","outputCost"]].mean().reset_index()
        labels_cost = [l for l in cost_avg["source_label"].tolist()]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Input Cost", x=cost_avg["source_label"],
                             y=cost_avg["inputCost"],
                             text=[f"${v:.3f}" for v in cost_avg["inputCost"]], textposition="inside"))
        fig.add_trace(go.Bar(name="Output Cost", x=cost_avg["source_label"],
                             y=cost_avg["outputCost"],
                             text=[f"${v:.3f}" for v in cost_avg["outputCost"]], textposition="inside"))
        fig = chart_layout(fig, "Avg Cost per Interaction", "Input tokens drive 72–93% of total cost")
        fig.update_layout(barmode="stack")
        fig.update_xaxes(title_text="Source")
        fig.update_yaxes(title_text="Avg Cost (USD)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(df, x="totalTokens", y="totalCost", color="source_label",
                          size="latency", size_max=25, color_discrete_map=COLOR_MAP,
                          trendline="ols",
                          labels={"totalTokens":"Total Tokens","totalCost":"Cost (USD)","source_label":"Source"})
        fig2 = chart_layout(fig2, "Cost vs Token Usage (bubble=latency)",
                            "Spearman r=0.99, p<0.0001 — near-perfect linear relationship")
        fig2.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        verbosity = df.groupby("source_label")["output_ratio"].mean().reset_index().sort_values("output_ratio", ascending=False)
        fig3 = px.bar(verbosity, x="source_label", y="output_ratio",
                      color="source_label", color_discrete_map=COLOR_MAP,
                      text=[f"{v:.3f}" for v in verbosity["output_ratio"]],
                      labels={"source_label":"Source","output_ratio":"Output/Input Ratio"})
        fig3.update_traces(textposition="outside", showlegend=False, cliponaxis=False)
        fig3 = chart_layout(fig3, "Response Verbosity (Output/Input Token Ratio)",
                            "defects_search: 5× more verbose than license_mgmt")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        session_cost = df.groupby("source_label")["totalCost"].sum().reset_index().sort_values("totalCost", ascending=True)
        fig4 = px.bar(session_cost, x="totalCost", y="source_label", orientation="h",
                      color="source_label", color_discrete_map=COLOR_MAP,
                      text=[f"${v:.3f}" for v in session_cost["totalCost"]],
                      labels={"totalCost":"Total Cost (USD)","source_label":"Source"})
        fig4.update_traces(textposition="outside", showlegend=False, cliponaxis=False)
        fig4 = chart_layout(fig4, "Total Session Cost by Source",
                            "case_create is most expensive overall at $0.726")
        st.plotly_chart(fig4, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — METADATA INTELLIGENCE
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Metadata Intelligence":
    st.markdown("# 🔍 Metadata Intelligence")
    st.markdown(
        "<div class='insight-box'>Extracted from <b>227 raw trace events</b> parsed from the OpenTelemetry "
        "output payload. Each event contains precise <code>startTime</code>, <code>endTime</code>, "
        "<code>totalTimeMs</code>, token usage, model config, collaborator names, and API paths.</div>",
        unsafe_allow_html=True
    )

    tab1, tab2, tab3 = st.tabs(["Guardrail Analysis", "Action Group APIs", "Routing Intelligence"])

    with tab1:
        grd = tdf[tdf["event_type"].isin(["guardrail_pre","guardrail_post"])]
        col1, col2 = st.columns(2)
        with col1:
            grd_avg = grd.groupby(["source_label","event_type"])["step_ms"].mean().reset_index()
            fig = px.bar(grd_avg, x="source_label", y="step_ms", color="event_type", barmode="group",
                         labels={"source_label":"Source","step_ms":"Avg Time (ms)","event_type":"Stage"},
                         color_discrete_sequence=["#6366f1","#a855f7"])
            fig = chart_layout(fig, "Guardrail Latency: Pre vs Post", "Consistent ~650–750ms per check")
            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            grd_per_int = grd.groupby(["interaction_id","source_label"]).agg(
                checks=("step_ms","count"), total_ms=("step_ms","sum")).reset_index()
            grd_src = grd_per_int.groupby("source_label")[["checks","total_ms"]].mean().reset_index()
            fig2 = px.bar(grd_src, x="source_label", y="total_ms",
                          text=[f"{v:.0f}ms" for v in grd_src["total_ms"]],
                          color="source_label", color_discrete_map=COLOR_MAP,
                          labels={"source_label":"Source","total_ms":"Avg Guardrail Time (ms)"})
            fig2.update_traces(textposition="outside", showlegend=False, cliponaxis=False)
            fig2 = chart_layout(fig2, "Avg Total Guardrail Time per Interaction",
                                "0 blocks in dataset — 100% pass-through overhead")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            "<div class='warn-box'>⚠️ <b>Key Finding:</b> All guardrail actions returned <code>NONE</code> — "
            "not a single interaction was blocked in this dataset. Yet guardrails add "
            "<b>2,500–3,200ms</b> of fixed overhead per interaction. "
            "Consider batching pre+post checks or introducing trusted-session bypass.</div>",
            unsafe_allow_html=True
        )

    with tab2:
        api_data = [
            ("license_management", "validate-product-line", "LAMBDA", "GET", 1096),
            ("license_management", "validate-order-number", "LAMBDA", "GET", 3095),
            ("firmware_recommendation", "get_recommendation", "LAMBDA", "GET", 4435),
            ("troubleshooting", "knowledge-base/query", "LAMBDA", "POST", 5097),
            ("case_update", "verify-case", "LAMBDA", "POST", 8580),
            ("case_update", "update-case", "LAMBDA", "POST", 10001),
            ("case_create", "verify-case", "LAMBDA", "POST", 12774),
            ("troubleshooting", "get-schema-sddm", "LAMBDA", "GET", 13695),
            ("troubleshooting", "execute-query-sddm", "LAMBDA", "POST", 37303),
            ("defects_search", "knowledge-base/query", "LAMBDA", "POST", 50636),
            ("case_create", "case-attachment", "RETURN_CONTROL", "POST", 0),
        ]
        api_df = pd.DataFrame(api_data, columns=["source","endpoint","exec_type","verb","ms_val"])
        api_df["source_label"] = api_df["source"].map(ABBREV)
        api_filt = api_df[api_df["source"].isin(selected_sources) & (api_df["ms_val"] > 0)].sort_values("ms_val")

        fig3 = px.bar(api_filt, x="ms_val", y="endpoint", orientation="h",
                      color="source_label", color_discrete_map=COLOR_MAP,
                      text=[f"{v/1000:.1f}s" for v in api_filt["ms_val"]],
                      labels={"ms_val":"Latency (ms)","endpoint":"API Endpoint","source_label":"Source"})
        fig3.update_traces(textposition="outside", cliponaxis=False)
        fig3 = chart_layout(fig3, "Action Group API Endpoint Latency",
                            "knowledge-base/query (50.6s) is the #1 API bottleneck")
        fig3.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5))
        st.plotly_chart(fig3, use_container_width=True)

        rc_data = [
            ("LAMBDA", "GET", "license_management", 3, 1096),
            ("LAMBDA", "GET", "firmware_recommendation", 2, 4435),
            ("LAMBDA", "POST", "troubleshooting", 3, 18698),
            ("LAMBDA", "POST", "case_create", 2, 8952),
            ("LAMBDA", "POST", "case_update", 3, 6414),
            ("RETURN_CONTROL", "POST", "case_create", 1, 0),
        ]
        rc_df2 = pd.DataFrame(rc_data, columns=["exec_type","verb","source","count","avg_ms"])
        rc_df2["source_label"] = rc_df2["source"].map(ABBREV)
        st.markdown("<div class='section-title'>Execution Type: LAMBDA vs RETURN_CONTROL</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='insight-box'><b>LAMBDA</b>: Action executed synchronously in AWS Lambda — "
            "adds full round-trip latency to the pipeline.<br>"
            "<b>RETURN_CONTROL</b>: Action deferred back to the client — "
            "zero Lambda latency, but requires client-side handling "
            "(used for <code>case-attachment</code> in case_create).</div>",
            unsafe_allow_html=True
        )
        st.dataframe(rc_df2[["source_label","exec_type","verb","count","avg_ms"]].rename(
            columns={"source_label":"Source","exec_type":"Exec Type","verb":"HTTP Verb",
                     "count":"# Calls","avg_ms":"Avg ms"}),
            use_container_width=True, hide_index=True
        )

    with tab3:
        rc_df = tdf[tdf["event_type"]=="routing_classifier"].dropna(subset=["step_ms"])
        col1, col2 = st.columns(2)
        with col1:
            rc_src = rc_df.groupby("source_label")["step_ms"].mean().reset_index().sort_values("step_ms")
            fig4 = px.bar(rc_src, x="step_ms", y="source_label", orientation="h",
                          color="source_label", color_discrete_map=COLOR_MAP,
                          text=[f"{v/1000:.2f}s" for v in rc_src["step_ms"]],
                          labels={"step_ms":"Avg Latency (ms)","source_label":"Source"})
            fig4.update_traces(textposition="outside", showlegend=False, cliponaxis=False)
            fig4 = chart_layout(fig4, "Routing Classifier Latency",
                                "Troubleshooting router is 6× slower than License Mgmt")
            st.plotly_chart(fig4, use_container_width=True)

        with col2:
            if not rc_df.empty and "input_tokens" in rc_df.columns:
                rc_tok = rc_df.dropna(subset=["input_tokens"])
                if len(rc_tok) > 0:
                    fig5 = px.scatter(rc_tok, x="input_tokens", y="step_ms", color="source_label",
                                      color_discrete_map=COLOR_MAP,
                                      labels={"input_tokens":"Input Tokens","step_ms":"Classifier Time (ms)","source_label":"Source"})
                    fig5 = chart_layout(fig5, "Classifier Token Load vs Latency",
                                        "Token load is similar — latency gap is routing ambiguity")
                    fig5.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5))
                    st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 — OPTIMIZATION PLAYBOOK
# ════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Optimization Playbook":
    st.markdown("# 🚨 Optimization Playbook")
    st.markdown(
        "<div class='insight-box'>Ranked optimization opportunities derived from statistical analysis "
        "of latency distributions, pipeline step profiling, and metadata trace extraction. "
        "Each item is backed by quantitative evidence from the log data.</div>",
        unsafe_allow_html=True
    )

    priorities = [
        ("🔴 P0 — Critical", [
            ("Defects-Management-Agent is a single point of failure",
             "75.9s roundtrip in defects_search; 129.3s in troubleshooting. "
             "Root cause: synchronous blocking sub-agent with cascading internal LLM calls.",
             "Implement async dispatch with 30s timeout. Add semantic result cache (Redis + embedding match). "
             "Pre-warm the agent on application start."),
            ("/knowledge-base/query takes 50.6s",
             "Blocking synchronous vector search call with no caching or fallback path. "
             "Accounts for 62% of defects_search total latency.",
             "Add an in-memory semantic cache layer. Use approximate nearest-neighbor (ANN) indexing. "
             "Return cached results for high-similarity queries (cosine sim > 0.92)."),
        ]),
        ("🟠 P1 — High", [
            ("/rma/execute-query-sddm takes 37.3s",
             "Dynamic SQL generation + execution against schema-on-demand model. "
             "Used in troubleshooting for RMA data retrieval.",
             "Pre-materialise the top 20 most common RMA query patterns as views. "
             "Add query result cache with 1-hour TTL."),
            ("case_update fires 4.5 LLM calls per interaction",
             "Sequential verify-case → update-case → confirm-case flow — each step is a separate LLM call. "
             "This is statistically the most LLM-call-heavy flow (p=0.0002 vs license_management).",
             "Merge verify + update into a single structured-output LLM call. "
             "Use tool-calling with strict JSON schema to eliminate the confirm step."),
        ]),
        ("🟡 P2 — Medium", [
            ("Guardrail fixed tax: 2,500–3,200ms per interaction, 0 blocks",
             "All guardrail actions returned NONE in this dataset — no interaction was ever blocked. "
             "Yet 3–4 checks run per interaction adding 2.5–3.2s of pure overhead.",
             "Batch pre+post guardrail into a single async check. "
             "Introduce trusted-session mode for authenticated users with clean history."),
            ("Routing classifier 6× slower for troubleshooting (7.75s vs 1.22s)",
             "Token load is nearly identical (~3,200–4,500 tokens) across all sources — "
             "the slowdown is not token-driven. Likely query ambiguity spanning multiple domains.",
             "Add a lightweight intent pre-classifier (< 500ms) before routing. "
             "Cache classifier decisions for repeated or similar queries."),
        ]),
        ("🟢 P3 — Low", [
            ("Context window grows +26–70% per turn",
             "Tokens accumulate across session turns as full history is passed each time. "
             "firmware_recommendation grows +70.3%/turn — fastest context explosion.",
             "Implement conversation summarization after turn 3. "
             "Use a rolling window that retains the last 2 turns + a running summary."),
            ("case_create RETURN_CONTROL action (case-attachment)",
             "One action in case_create uses RETURN_CONTROL, deferring to the client. "
             "If the client does not handle this, it creates silent failures.",
             "Add explicit client-side handler validation. "
             "Log RETURN_CONTROL events separately for monitoring."),
        ]),
    ]

    col1, col2 = st.columns(2)
    for i, (priority, items) in enumerate(priorities):
        col = col1 if i % 2 == 0 else col2
        with col:
            color = "#ef4444" if "P0" in priority else "#f59e0b" if "P1" in priority else "#eab308" if "P2" in priority else "#22c55e"
            st.markdown(f"<div style='color:{color};font-size:18px;font-weight:700;margin:20px 0 10px 0'>{priority}</div>",
                        unsafe_allow_html=True)
            for title, evidence, fix in items:
                with st.expander(f"**{title}**"):
                    st.markdown(f"**📊 Evidence:** {evidence}")
                    st.markdown(f"**🛠️ Fix:** {fix}")

    st.markdown("---")
    st.markdown("<div class='section-title'>Estimated Impact Summary</div>", unsafe_allow_html=True)

    impact_data = pd.DataFrame([
        {"Priority": "🔴 P0", "Optimization": "Cache knowledge-base/query", 
         "Current (avg)": "81.8s", "Expected After": "< 5s", "Latency Reduction": "94%", "Cost Impact": "-30%"},
        {"Priority": "🔴 P0", "Optimization": "Async Defects-Mgmt agent", 
         "Current (avg)": "129s (peak)", "Expected After": "< 30s", "Latency Reduction": "77%", "Cost Impact": "Neutral"},
        {"Priority": "🟠 P1", "Optimization": "Merge case_update LLM calls", 
         "Current (avg)": "36.7s", "Expected After": "~18s", "Latency Reduction": "~50%", "Cost Impact": "-40%"},
        {"Priority": "🟠 P1", "Optimization": "Materialise RMA query views", 
         "Current (avg)": "37.3s (api)", "Expected After": "< 2s", "Latency Reduction": "95%", "Cost Impact": "Neutral"},
        {"Priority": "🟡 P2", "Optimization": "Batch/async guardrails", 
         "Current (avg)": "2.8s overhead", "Expected After": "~0.7s", "Latency Reduction": "75%", "Cost Impact": "Neutral"},
        {"Priority": "🟡 P2", "Optimization": "Intent pre-classifier", 
         "Current (avg)": "7.75s router", "Expected After": "< 1.5s", "Latency Reduction": "80%", "Cost Impact": "Minimal"},
        {"Priority": "🟢 P3", "Optimization": "Rolling context window", 
         "Current (avg)": "+27–70%/turn", "Expected After": "+5–10%/turn", "Latency Reduction": "~15%", "Cost Impact": "-25%"},
    ])
    st.dataframe(impact_data, use_container_width=True, hide_index=True)
