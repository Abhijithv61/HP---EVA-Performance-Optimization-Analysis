
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="AVA Bot · Bottleneck Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Theme & CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Page background */
  .stApp { background: #0d1117; }
  [data-testid="stAppViewContainer"] { background: #0d1117; }
  [data-testid="stHeader"] { background: #0d1117; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #161b22;
    border-radius: 10px; padding: 4px;
    border: 1px solid #21262d;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 7px;
    color: #8b949e; font-size: 13px; font-weight: 500;
    padding: 8px 18px; border: none;
  }
  .stTabs [aria-selected="true"] {
    background: #21262d !important;
    color: #e6edf3 !important; font-weight: 600;
  }
  .stTabs [data-baseweb="tab"]:hover { color: #c9d1d9 !important; }

  /* Divider */
  hr { border-color: #21262d; margin: 24px 0; }

  /* Metric card */
  .card {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 10px; padding: 20px 22px;
  }
  .card-label {
    color: #8b949e; font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: .8px; margin-bottom: 6px;
  }
  .card-value { color: #e6edf3; font-size: 28px; font-weight: 700; line-height: 1; }
  .card-sub   { color: #6e7681; font-size: 12px; margin-top: 4px; }

  /* Severity badges */
  .badge {
    display: inline-block; border-radius: 5px;
    padding: 2px 10px; font-size: 12px; font-weight: 600;
  }
  .badge-crit  { background: rgba(248,81,73,.12);  color: #f85149; border: 1px solid rgba(248,81,73,.3);  }
  .badge-high  { background: rgba(210,153,34,.12); color: #d29922; border: 1px solid rgba(210,153,34,.3); }
  .badge-med   { background: rgba(88,166,255,.12); color: #58a6ff; border: 1px solid rgba(88,166,255,.3); }
  .badge-low   { background: rgba(63,185,80,.12);  color: #3fb950; border: 1px solid rgba(63,185,80,.3);  }

  /* Callout boxes */
  .callout {
    border-radius: 8px; padding: 14px 18px;
    font-size: 13px; line-height: 1.65; color: #c9d1d9;
  }
  .callout-red    { background: rgba(248,81,73,.07);  border-left: 3px solid #f85149; }
  .callout-yellow { background: rgba(210,153,34,.07); border-left: 3px solid #d29922; }
  .callout-blue   { background: rgba(88,166,255,.07); border-left: 3px solid #58a6ff; }
  .callout-green  { background: rgba(63,185,80,.07);  border-left: 3px solid #3fb950; }

  /* Fix step */
  .fix-step {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 8px; padding: 14px 18px; margin-bottom: 10px;
  }
  .fix-title { color: #e6edf3; font-size: 14px; font-weight: 600; margin-bottom: 4px; }
  .fix-desc  { color: #8b949e; font-size: 13px; line-height: 1.6; }

  /* Section header */
  .section-hdr {
    color: #e6edf3; font-size: 15px; font-weight: 600;
    margin: 22px 0 12px 0; letter-spacing: .2px;
  }

  /* Page title */
  .page-title { color: #e6edf3; font-size: 22px; font-weight: 700; margin-bottom: 4px; }
  .page-sub   { color: #6e7681; font-size: 13px; margin-bottom: 24px; }

  /* Table */
  .stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Shared chart style ───────────────────────────────────────────────────────
PAPER = "#0d1117"
PLOT  = "#161b22"
GRID  = "#21262d"
TEXT  = "#c9d1d9"
MUTED = "#6e7681"

def base_layout(fig, title="", subtitle=""):
    t = title + (f"<br><span style='font-size:13px;color:{MUTED};font-weight:400'>{subtitle}</span>" if subtitle else "")
    fig.update_layout(
        title=dict(text=t, font=dict(color=TEXT, size=15), x=0),
        paper_bgcolor=PAPER, plot_bgcolor=PLOT,
        font=dict(color=TEXT, size=12),
        margin=dict(t=70, b=50, l=10, r=10),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, size=11)),
        hoverlabel=dict(bgcolor="#21262d", font_color="#e6edf3", bordercolor="#30363d"),
    )
    return fig

COLORS = ["#58a6ff","#3fb950","#d29922","#f85149","#a371f7","#ffa657"]
LABELS = ["Defects Search","Troubleshooting","Case Update","Case Create","Firmware Rec.","License Mgmt"]
SOURCES= ["defects_search","troubleshooting","case_update","case_create","firmware_recommendation","license_management"]
CMAP   = dict(zip(LABELS, COLORS))


# ─── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    import ast, json, os
    try:
        df  = pd.read_csv("logs_consolidated.csv", parse_dates=["timestamp"])
    except:
        return None, None

    def sp(v):
        if pd.isna(v): return None
        try: return ast.literal_eval(v)
        except:
            try: return json.loads(v)
            except: return None

    df["output_parsed"] = df["output"].apply(sp)
    df["latency_per_1k"] = df["latency"] / df["totalTokens"] * 1000
    df["out_ratio"]      = df["outputTokens"] / df["inputTokens"]

    rows = []
    for _, r in df.iterrows():
        for ev in (r["output_parsed"] or []):
            et = ev.get("type","")
            tr = ev.get("trace",{})
            if et == "guardrail_trace":
                meta = tr.get("metadata",{})
                rows.append({"source":r["source"],"iid":r["id"],"etype":"guardrail",
                    "ms":meta.get("totalTimeMs"),"collab":None,"api":None,
                    "itok":None,"otok":None,"action":tr.get("action")})
            elif et == "routing_classifier_trace":
                mio=tr.get("modelInvocationOutput",{})
                meta=mio.get("metadata",{})
                if meta.get("totalTimeMs"):
                    u=meta.get("usage",{})
                    rows.append({"source":r["source"],"iid":r["id"],"etype":"routing_clf",
                        "ms":meta.get("totalTimeMs"),"collab":None,"api":None,
                        "itok":u.get("inputTokens"),"otok":u.get("outputTokens"),"action":None})
                obs=tr.get("observation",{})
                ac=obs.get("agentCollaboratorInvocationOutput",{})
                m2=ac.get("metadata",{})
                if m2.get("totalTimeMs"):
                    rows.append({"source":r["source"],"iid":r["id"],"etype":"collab_rt",
                        "ms":m2.get("totalTimeMs"),"collab":ac.get("agentCollaboratorName"),
                        "api":None,"itok":None,"otok":None,"action":None})
            elif et == "orchestration_trace":
                mio=tr.get("modelInvocationOutput",{})
                meta=mio.get("metadata",{})
                if meta.get("totalTimeMs"):
                    u=meta.get("usage",{})
                    rows.append({"source":r["source"],"iid":r["id"],"etype":"llm",
                        "ms":meta.get("totalTimeMs"),"collab":None,"api":None,
                        "itok":u.get("inputTokens"),"otok":u.get("outputTokens"),"action":None})
                obs=tr.get("observation",{})
                ac=obs.get("agentCollaboratorInvocationOutput",{})
                m2=ac.get("metadata",{})
                if m2.get("totalTimeMs"):
                    rows.append({"source":r["source"],"iid":r["id"],"etype":"agent_collab",
                        "ms":m2.get("totalTimeMs"),"collab":ac.get("agentCollaboratorName"),
                        "api":None,"itok":None,"otok":None,"action":None})
                ag=obs.get("actionGroupInvocationOutput",{})
                mag=ag.get("metadata",{})
                ai=tr.get("invocationInput",{}).get("actionGroupInvocationInput",{})
                if mag.get("totalTimeMs") is not None:
                    rows.append({"source":r["source"],"iid":r["id"],"etype":"action_grp",
                        "ms":mag.get("totalTimeMs"),"collab":None,
                        "api":ai.get("apiPath"),"itok":None,"otok":None,"action":None})

    tdf = pd.DataFrame(rows)
    for c in ["ms","itok","otok"]: tdf[c] = pd.to_numeric(tdf[c], errors="coerce")
    return df, tdf

df, tdf = load()


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
t1, t2, t3, t4, t5 = st.tabs([
    "  Overview  ",
    "  Sub-Agent & API  ",
    "  LLM & Tokens  ",
    "  Guardrail & Routing  ",
    "  Optimization Playbook  "
])


# ══════════════════════════════════════════
# TAB 1 · OVERVIEW
# ══════════════════════════════════════════
with t1:
    st.markdown("<div class='page-title'>AVA Bot · Bottleneck Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>22 interactions · 6 sources · 199 trace events · Staging environment</div>", unsafe_allow_html=True)

    # KPIs
    c1,c2,c3,c4,c5 = st.columns(5)
    kpis = [
        (c1, "Slowest Source",  "81.8s",  "Defects Search avg"),
        (c2, "Peak Latency",    "143.6s", "Troubleshooting max"),
        (c3, "Guardrail Tax",   "~2.8s",  "per interaction, 0 blocks"),
        (c4, "Token Growth",    "+70%",   "Firmware Rec. per turn"),
        (c5, "LLM Calls Peak",  "4.5×",   "per interaction (Case Update)"),
    ]
    for col, label, val, sub in kpis:
        col.markdown(
            f"<div class='card'><div class='card-label'>{label}</div>"
            f"<div class='card-value'>{val}</div>"
            f"<div class='card-sub'>{sub}</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Bottleneck summary table
    col_l, col_r = st.columns([1.1, 1])

    with col_l:
        st.markdown("<div class='section-hdr'>Bottleneck Inventory</div>", unsafe_allow_html=True)
        bn_df = pd.DataFrame([
            ["B1","Sub-Agent Roundtrip (Defects-Mgmt)",    "🔴 Critical", "75–129s",   "Async + Semantic Cache"],
            ["B2","Pipeline Step Imbalance",                "🔴 Critical", "Varies",    "Parallelise independent steps"],
            ["B3","Excessive LLM Calls (Case Update)",      "🟠 High",     "~18s waste","Merge to single structured call"],
            ["B4","Context Window Token Snowball",          "🟠 High",     "+27–70%/turn","Rolling summarisation"],
            ["B5","Guardrail Dead Weight",                  "🟡 Medium",   "2.5–3.2s",  "Async batch + session bypass"],
            ["B6","Routing Classifier Spike (Troubleshoot)","🟡 Medium",   "7.75s avg", "Intent pre-classifier + cache"],
        ], columns=["ID","Bottleneck","Severity","Impact","Fix"])
        st.dataframe(bn_df, use_container_width=True, hide_index=True,
                     column_config={
                         "ID": st.column_config.TextColumn(width="small"),
                         "Severity": st.column_config.TextColumn(width="small"),
                     })

    with col_r:
        st.markdown("<div class='section-hdr'>End-to-End Latency by Source</div>", unsafe_allow_html=True)
        src_lat = {"Defects Search":81.76,"Troubleshooting":79.69,"Case Update":36.72,
                   "Case Create":26.50,"Firmware Rec.":19.38,"License Mgmt":13.38}
        fig = go.Figure(go.Bar(
            x=list(src_lat.values()), y=list(src_lat.keys()), orientation="h",
            text=[f"{v:.1f}s" for v in src_lat.values()], textposition="outside",
            marker=dict(color=[CMAP[k] for k in src_lat.keys()], opacity=0.9),
        ))
        fig = base_layout(fig, "Avg Response Latency")
        fig.update_xaxes(title_text="Seconds")
        fig.update_yaxes(title_text="")
        fig.update_traces(cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-hdr'>Latency Stack — Where Time Goes</div>", unsafe_allow_html=True)

    step_data = {
        "Guardrail":     [0.671, 0.626, 0.714, 0.711, 0.806, 0.718],
        "Routing Clf":   [3.668, 7.752, 2.028, 1.542, 1.772, 1.218],
        "Sub-Agent RT":  [75.946, 69.872, 32.530, 23.232, 14.870, 9.977],
        "Agent Collab":  [0.0,    0.0,   20.488, 11.332, 0.0,   0.0  ],
        "Action Group":  [50.636, 18.698, 6.414,  8.952, 4.435, 1.329],
        "LLM":           [11.553, 24.455, 4.428,  6.738, 6.968, 4.427],
    }
    step_colors = ["#6e7681","#a371f7","#f85149","#ffa657","#d29922","#58a6ff"]
    fig2 = go.Figure()
    for (sname, vals), sc in zip(step_data.items(), step_colors):
        fig2.add_trace(go.Bar(name=sname, x=LABELS, y=vals, marker_color=sc, opacity=0.9))
    fig2 = base_layout(fig2, "Avg Latency Stack per Source", "Each segment = avg time in that pipeline step")
    fig2.update_layout(barmode="stack",
        legend=dict(orientation="h", y=-0.22, xanchor="center", x=0.5))
    fig2.update_xaxes(title_text="")
    fig2.update_yaxes(title_text="Seconds")
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════
# TAB 2 · SUB-AGENT & API
# ══════════════════════════════════════════
with t2:
    st.markdown("<div class='page-title'>Sub-Agent & API Bottlenecks</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Bottlenecks B1 & B2 · Root cause of worst-case latency</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='callout callout-red'>
    <strong>Finding:</strong> The <code>Defects-Management-Agent</code> sub-agent is called <strong>synchronously</strong>
    with no timeout or cache. It internally triggers a <code>/knowledge-base/query</code> action that alone takes
    <strong>50.6s</strong>, creating a cascading blocking chain that accounts for
    <strong>93% of Defects Search latency</strong> and <strong>87% of Troubleshooting latency</strong>.
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-hdr'>Sub-Agent Roundtrip Latency</div>", unsafe_allow_html=True)
        c_data = [
            ("License Assist","License Mgmt",9977),
            ("SupportCase","Case Create",11332),
            ("Firmware Assist","Firmware Rec.",14870),
            ("SupportCase","Case Update",20488),
            ("Case-Mgmt2","Case Create",23232),
            ("Case-Mgmt2","Case Update",32530),
            ("Defects-Mgmt","Defects Search",75946),
            ("Defects-Mgmt","Troubleshooting",129347),
        ]
        c_df = pd.DataFrame(c_data, columns=["Agent","Source","ms"])
        c_df["label"] = c_df["Agent"] + " → " + c_df["Source"]
        c_df = c_df.sort_values("ms")
        fig = px.bar(c_df, x="ms", y="label", orientation="h",
                     color="Source", color_discrete_map=CMAP,
                     text=[f"{v/1000:.1f}s" for v in c_df["ms"]],
                     labels={"ms":"Roundtrip (ms)","label":""})
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig = base_layout(fig, "Collaborator Roundtrip Times")
        fig.update_xaxes(title_text="ms")
        fig.update_layout(legend=dict(orientation="h", y=1.12, xanchor="center", x=0.5))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-hdr'>Sub-Agent Share of Total Latency</div>", unsafe_allow_html=True)
        sa_share = {
            "Defects Search":   (75946, 81760),
            "Troubleshooting":  (69872, 79690),
            "Case Update":      (32530, 36720),
            "Case Create":      (23232, 26500),
            "Firmware Rec.":    (14870, 19383),
            "License Mgmt":     (9977,  13380),
        }
        sa_pct  = [v[0]/v[1]*100 for v in sa_share.values()]
        sa_rest = [100-p for p in sa_pct]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Sub-Agent RT", x=list(sa_share.keys()), y=sa_pct,
                              text=[f"{p:.0f}%" for p in sa_pct], textposition="inside",
                              marker_color="#f85149", opacity=0.85))
        fig2.add_trace(go.Bar(name="Rest of Pipeline", x=list(sa_share.keys()), y=sa_rest,
                              marker_color="#21262d"))
        fig2 = base_layout(fig2, "Sub-Agent Share of Total Latency (%)")
        fig2.update_layout(barmode="stack",
            legend=dict(orientation="h", y=1.12, xanchor="center", x=0.5))
        fig2.update_xaxes(title_text="")
        fig2.update_yaxes(title_text="%")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div class='section-hdr'>Action Group API Latency</div>", unsafe_allow_html=True)
        api_data = [
            ("validate-product-line", "License Mgmt",    1096),
            ("validate-order-number", "License Mgmt",    3095),
            ("get_recommendation",    "Firmware Rec.",   4435),
            ("knowledge-base/query",  "Troubleshooting", 5097),
            ("verify-case",           "Case Update",     8580),
            ("update-case",           "Case Update",     10001),
            ("verify-case",           "Case Create",     12774),
            ("get-schema-sddm",       "Troubleshooting", 13695),
            ("execute-query-sddm",    "Troubleshooting", 37303),
            ("knowledge-base/query",  "Defects Search",  50636),
        ]
        a_df = pd.DataFrame(api_data, columns=["endpoint","source","ms"]).sort_values("ms")
        a_df["label"] = a_df["source"] + ": " + a_df["endpoint"]
        fig3 = px.bar(a_df, x="ms", y="label", orientation="h",
                      color="source", color_discrete_map=CMAP,
                      text=[f"{v/1000:.1f}s" for v in a_df["ms"]],
                      labels={"ms":"Latency (ms)","label":""})
        fig3.update_traces(textposition="outside", cliponaxis=False)
        fig3 = base_layout(fig3, "API Endpoint Latency")
        fig3.update_xaxes(title_text="ms")
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("<div class='section-hdr'>Fixes for B1 & B2</div>", unsafe_allow_html=True)
        fixes_b1 = [
            ("🔴 Async sub-agent dispatch",
             "Switch Defects-Management-Agent invocation from synchronous to async with a 30s hard timeout. "
             "Return a graceful fallback response on timeout rather than hanging."),
            ("🔴 Semantic cache for /knowledge-base/query",
             "Add Redis + vector embedding similarity cache in front of the knowledge base. "
             "Queries with cosine similarity > 0.92 to a cached query return instantly (< 100ms vs 50.6s)."),
            ("🟠 Parallelise case_create sub-agents",
             "Case-Mgmt2-Collaborator (23.2s) and SupportCase-Management-Agent (11.3s) are logically "
             "independent — run them concurrently. Expected saving: ~11s per interaction."),
            ("🟠 Materialise RMA query views",
             "Pre-compute the top 20 most frequent /rma/execute-query-sddm patterns as database views. "
             "Reduces 37.3s query time to < 2s for known patterns."),
            ("🟡 Pre-warm Defects-Management-Agent",
             "Send a no-op warm-up call on service startup to eliminate cold-start overhead. "
             "Schedule periodic keep-alive pings every 5 minutes."),
        ]
        for title, desc in fixes_b1:
            st.markdown(f"<div class='fix-step'><div class='fix-title'>{title}</div>"
                        f"<div class='fix-desc'>{desc}</div></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════
# TAB 3 · LLM & TOKENS
# ══════════════════════════════════════════
with t3:
    st.markdown("<div class='page-title'>LLM Calls & Token Bottlenecks</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Bottlenecks B3 & B4 · Excessive LLM calls and context window snowballing</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='callout callout-yellow'>
    <strong>Finding:</strong> <code>case_update</code> makes <strong>4.5 LLM calls per interaction</strong>
    in a sequential chain — each call waits for the last. Meanwhile token count grows <strong>+26–70% per turn</strong>
    across all sources, driving both latency (Spearman r=0.68, p=0.0005) and cost (r=0.99, p&lt;0.0001)
    linearly higher with every additional turn.
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-hdr'>LLM Calls per Interaction</div>", unsafe_allow_html=True)
        llm_calls = {"Case Update":4.5,"Case Create":2.75,"Defects Search":2.0,
                     "Troubleshooting":2.0,"License Mgmt":1.57,"Firmware Rec.":1.5}
        llm_ms    = {"Case Update":4462,"Case Create":6639,"Defects Search":11553,
                     "Troubleshooting":18999,"License Mgmt":4370,"Firmware Rec.":6693}
        sorted_calls = dict(sorted(llm_calls.items(), key=lambda x: x[1], reverse=True))
        fig = go.Figure(go.Bar(
            x=list(sorted_calls.keys()), y=list(sorted_calls.values()),
            text=[f"{v:.1f}" for v in sorted_calls.values()], textposition="outside",
            marker=dict(color=[CMAP[k] for k in sorted_calls.keys()], opacity=0.9)
        ))
        fig = base_layout(fig, "Avg LLM Calls / Interaction",
                          "case_update chains 4.5 sequential calls — most expensive flow")
        fig.update_xaxes(title_text="")
        fig.update_yaxes(title_text="Avg LLM Calls", range=[0, 5.5])
        fig.update_traces(cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-hdr'>Avg LLM Time per Call</div>", unsafe_allow_html=True)
        sorted_ms = dict(sorted(llm_ms.items(), key=lambda x: x[1], reverse=True))
        fig2 = go.Figure(go.Bar(
            x=list(sorted_ms.keys()), y=list(sorted_ms.values()),
            text=[f"{v/1000:.1f}s" for v in sorted_ms.values()], textposition="outside",
            marker=dict(color=[CMAP[k] for k in sorted_ms.keys()], opacity=0.9)
        ))
        fig2 = base_layout(fig2, "Avg Time per LLM Call (ms)",
                           "troubleshooting: 19s/call — large context dumps slow reasoning")
        fig2.update_xaxes(title_text="")
        fig2.update_yaxes(title_text="ms per Call")
        fig2.update_traces(cliponaxis=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<div class='section-hdr'>Token Growth Across Session Turns</div>", unsafe_allow_html=True)
        turn_data = {
            "Case Create":   [6929,17217,30386,35213,36699,44687,45130,45130],
            "Case Update":   [26402,41039],
            "Firmware Rec.": [7366,12547],
            "License Mgmt":  [4453,10005,13028,14776,23625,31723,31723],
        }
        fig3 = go.Figure()
        for label, tokens in turn_data.items():
            fig3.add_trace(go.Scatter(
                x=list(range(1,len(tokens)+1)), y=tokens,
                mode="lines+markers", name=label,
                line=dict(color=CMAP.get(label,"#58a6ff"), width=2),
                marker=dict(size=7),
                hovertemplate=f"{label}<br>Turn %{{x}}: %{{y:,}} tokens<extra></extra>"
            ))
        fig3.add_hrect(y0=35000, y1=46000, fillcolor="rgba(248,81,73,0.06)",
                       line_width=0, annotation_text="High-cost zone",
                       annotation_font_color="#f85149", annotation_position="top left")
        fig3 = base_layout(fig3, "Token Count per Session Turn",
                           "Full history appended each turn — context snowballs with no pruning")
        fig3.update_xaxes(title_text="Turn", dtick=1)
        fig3.update_yaxes(title_text="Total Tokens")
        fig3.update_layout(legend=dict(orientation="h", y=1.12, xanchor="center", x=0.5))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("<div class='section-hdr'>Fixes for B3 & B4</div>", unsafe_allow_html=True)
        fixes_b3 = [
            ("🟠 Merge case_update LLM calls",
             "Combine verify-case → update-case → confirm into a single LLM call with a strict "
             "JSON output schema (function-calling / tool-use). "
             "Eliminates ~2 round-trips, saving approximately 9–18s per interaction."),
            ("🟠 Parallelise get-case-url",
             "The GET /get-case-details-page-url call is independent of the update LLM step. "
             "Fire it concurrently rather than sequentially. Zero-cost ~3s saving."),
            ("🟠 Rolling context window summarisation",
             "After turn 3, replace full conversation history with: last 2 turns verbatim + "
             "a compressed summary of all earlier turns (200–300 tokens). "
             "Reduces case_create peak from 44,687 → ~13,000 tokens — 66% reduction."),
            ("🟡 Trim action group outputs before passing to LLM",
             "The /rma/get-schema-sddm response is passed raw into the LLM context. "
             "Extract only the relevant fields before injecting — reduces per-call input tokens "
             "by an estimated 30–40% for troubleshooting flows."),
            ("🟡 Output caching for repeated LLM patterns",
             "For license_management flows, the verify → lookup pattern is nearly identical across "
             "interactions. Cache structured LLM outputs keyed by (intent + product_line) for 24h TTL."),
        ]
        for title, desc in fixes_b3:
            st.markdown(f"<div class='fix-step'><div class='fix-title'>{title}</div>"
                        f"<div class='fix-desc'>{desc}</div></div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-hdr'>Token Growth Rate & Cost Correlation</div>", unsafe_allow_html=True)
    growth_df = pd.DataFrame([
        ["Case Create",  "+26.8%/turn", 6929,  44687, 38, "$0.091"],
        ["Case Update",  "+35.0%/turn", 26402, 41039, 56, "$0.116"],
        ["Firmware Rec.","+70.3%/turn", 7366,  12547, 26, "$0.034"],
        ["License Mgmt", "+33.0%/turn", 4453,  31723, 27, "$0.053"],
    ], columns=["Source","Avg Growth/Turn","Min Tokens","Max Tokens","Token Range (k)","Avg Cost"])
    st.dataframe(growth_df, use_container_width=True, hide_index=True)
    st.markdown(
        "<div class='callout callout-blue' style='margin-top:12px'>📈 "
        "<strong>Spearman r = 0.99 (p &lt; 0.0001)</strong> between token count and cost — "
        "every token saved is a direct cost reduction. "
        "Rolling summarisation on case_create alone would save an estimated "
        "<strong>~$0.04 per interaction</strong> from turn 4 onwards.</div>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════
# TAB 4 · GUARDRAIL & ROUTING
# ══════════════════════════════════════════
with t4:
    st.markdown("<div class='page-title'>Guardrail & Routing Bottlenecks</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Bottlenecks B5 & B6 · Fixed overhead with zero security value observed</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class='callout callout-yellow'>
        <strong>Guardrail (B5):</strong> Every interaction runs 3–4 guardrail checks adding
        <strong>2,500–3,200ms</strong> of fixed overhead. In this dataset,
        <strong>every single action returned NONE</strong> — not one interaction was flagged or blocked.
        This is pure overhead with zero security value observed.
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class='callout callout-blue'>
        <strong>Routing Classifier (B6):</strong> The troubleshooting router takes
        <strong>7,752ms avg</strong> (up to 12,106ms) vs 1,218ms for License Mgmt —
        despite nearly identical token loads (~3,200–4,300 tokens).
        The gap is routing <strong>ambiguity</strong>, not token volume.
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-hdr'>Guardrail Overhead per Interaction</div>", unsafe_allow_html=True)
        grd = {"Defects Search":2685,"Troubleshooting":2504,"Case Create":2643,
               "Case Update":2856,"Firmware Rec.":3224,"License Mgmt":2870}
        total_lat_ms = {"Defects Search":81760,"Troubleshooting":79690,"Case Create":26500,
                        "Case Update":36720,"Firmware Rec.":19383,"License Mgmt":13380}
        g_pct = {k: v/total_lat_ms[k]*100 for k,v in grd.items()}
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(grd.keys()), y=list(grd.values()),
            text=[f"{v}ms<br>({g_pct[k]:.1f}%)" for k,v in grd.items()],
            textposition="outside",
            marker=dict(color=[CMAP[k] for k in grd.keys()], opacity=0.85)
        ))
        fig.add_hline(y=2700, line_dash="dot", line_color="#6e7681",
                      annotation_text="avg baseline 2.7s", annotation_font_color="#6e7681")
        fig = base_layout(fig, "Avg Guardrail Time per Interaction", "0 blocks triggered — 100% overhead")
        fig.update_xaxes(title_text="")
        fig.update_yaxes(title_text="Guardrail Time (ms)")
        fig.update_traces(cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-hdr'>Routing Classifier Latency</div>", unsafe_allow_html=True)
        rc_lat = {"License Mgmt":1218,"Case Create":1542,"Firmware Rec.":1772,
                  "Case Update":2028,"Defects Search":3668,"Troubleshooting":7752}
        rc_tok = {"License Mgmt":3665,"Case Create":3741,"Firmware Rec.":3300,
                  "Case Update":3278,"Defects Search":3204,"Troubleshooting":3764}
        rc_sorted = dict(sorted(rc_lat.items(), key=lambda x:x[1]))
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=list(rc_sorted.values()), y=list(rc_sorted.keys()), orientation="h",
            text=[f"{v}ms" for v in rc_sorted.values()], textposition="outside",
            marker=dict(color=[CMAP[k] for k in rc_sorted.keys()], opacity=0.85)
        ))
        fig2 = base_layout(fig2, "Routing Classifier Latency",
                           "Token loads are similar — gap is query ambiguity, not size")
        fig2.update_xaxes(title_text="ms")
        fig2.update_yaxes(title_text="")
        fig2.update_traces(cliponaxis=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div class='section-hdr'>Guardrail vs Routing Token Load</div>", unsafe_allow_html=True)
        tok_df = pd.DataFrame({
            "Source": list(rc_tok.keys()),
            "Router Input Tokens": list(rc_tok.values()),
            "Classifier Latency (ms)": [rc_lat[k] for k in rc_tok.keys()],
        })
        fig3 = px.scatter(tok_df, x="Router Input Tokens", y="Classifier Latency (ms)",
                          color="Source", color_discrete_map=CMAP, size_max=18,
                          text="Source",
                          labels={"Router Input Tokens":"Input Tokens","Classifier Latency (ms)":"Latency (ms)"})
        fig3.update_traces(textposition="top center", marker=dict(size=12))
        fig3 = base_layout(fig3, "Token Load vs Classifier Latency",
                           "Similar tokens, very different times — ambiguity is the real driver")
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("<div class='section-hdr'>Fixes for B5 & B6</div>", unsafe_allow_html=True)
        fixes_b56 = [
            ("🟡 Async batch guardrail checks",
             "Move pre+post guardrail to a single async batch call. "
             "Run the guardrail check in parallel with routing classification rather than sequentially before it. "
             "Expected saving: 1.5–2.0s per interaction with no change to security coverage."),
            ("🟡 Trusted-session guardrail bypass",
             "For authenticated users with a clean interaction history (no flags in last 50 turns), "
             "skip guardrail pre-check or run at reduced sensitivity. "
             "All 82 guardrail events in this dataset returned NONE — bypass risk is low."),
            ("🟡 Intent pre-classifier before routing",
             "Add a lightweight intent tagger (< 500ms, can be a simple embedding classifier or small model) "
             "that labels the query domain before it reaches the routing classifier. "
             "The router then confirms rather than discovers, cutting ambiguity resolution time."),
            ("🟡 Cache routing decisions",
             "For semantically similar queries (cosine similarity > 0.95 to a cached query), "
             "reuse the routing decision directly. Troubleshooting queries repeat patterns frequently."),
        ]
        for title, desc in fixes_b56:
            st.markdown(f"<div class='fix-step'><div class='fix-title'>{title}</div>"
                        f"<div class='fix-desc'>{desc}</div></div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div class='callout callout-green'>✅ <strong>Quick Win:</strong> "
        "Batching guardrails + trusted-session bypass requires <strong>zero model changes</strong> — "
        "it's purely an infrastructure and orchestration config change. "
        "This delivers a <strong>~2–3s latency reduction on every single interaction</strong> "
        "across all 6 sources with minimal risk.</div>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════
# TAB 5 · OPTIMIZATION PLAYBOOK
# ══════════════════════════════════════════
with t5:
    st.markdown("<div class='page-title'>Optimization Playbook</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Ranked fixes · Estimated impact · Implementation effort</div>", unsafe_allow_html=True)

    # Expected latency after fixes
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-hdr'>Expected Latency After Fixes (by Priority)</div>", unsafe_allow_html=True)
        before = {"Defects Search":81.76,"Troubleshooting":79.69,"Case Update":36.72,
                  "Case Create":26.50,"Firmware Rec.":19.38,"License Mgmt":13.38}
        after_p0 = {"Defects Search":5.0,"Troubleshooting":15.0,"Case Update":36.72,
                    "Case Create":26.50,"Firmware Rec.":19.38,"License Mgmt":13.38}
        after_p1 = {"Defects Search":5.0,"Troubleshooting":15.0,"Case Update":18.0,
                    "Case Create":15.0,"Firmware Rec.":19.38,"License Mgmt":13.38}
        after_all = {"Defects Search":4.5,"Troubleshooting":12.0,"Case Update":14.0,
                     "Case Create":12.0,"Firmware Rec.":16.0,"License Mgmt":10.5}
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Current",   x=LABELS, y=[before[k]  for k in LABELS], marker_color="#f85149", opacity=0.7))
        fig.add_trace(go.Bar(name="After P0",  x=LABELS, y=[after_p0[k] for k in LABELS], marker_color="#d29922", opacity=0.8))
        fig.add_trace(go.Bar(name="After P0+P1",x=LABELS,y=[after_p1[k] for k in LABELS], marker_color="#3fb950", opacity=0.9))
        fig.add_trace(go.Bar(name="After All", x=LABELS, y=[after_all[k] for k in LABELS], marker_color="#58a6ff", opacity=0.9))
        fig = base_layout(fig, "Latency Reduction After Each Priority Tier")
        fig.update_layout(barmode="group",
            legend=dict(orientation="h", y=1.12, xanchor="center", x=0.5))
        fig.update_xaxes(title_text="")
        fig.update_yaxes(title_text="Latency (s)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-hdr'>Estimated Latency Reduction per Fix</div>", unsafe_allow_html=True)
        savings = [
            ("Semantic cache: knowledge-base/query", 50.6, "#f85149"),
            ("Async Defects-Mgmt agent",             45.0, "#f85149"),
            ("RMA query materialisation",            35.3, "#d29922"),
            ("Merge case_update LLM calls",          18.0, "#d29922"),
            ("Parallelise case_create sub-agents",   11.3, "#d29922"),
            ("Intent pre-classifier (troubleshoot)",  6.3, "#a371f7"),
            ("Guardrail async batch",                 2.1, "#a371f7"),
            ("Rolling context window (late turns)",   8.0, "#3fb950"),
        ]
        s_df = pd.DataFrame(savings, columns=["Fix","Saving_s","Color"]).sort_values("Saving_s")
        fig2 = go.Figure(go.Bar(
            x=s_df["Saving_s"], y=s_df["Fix"], orientation="h",
            text=[f"{v:.1f}s" for v in s_df["Saving_s"]], textposition="outside",
            marker=dict(color=s_df["Color"].tolist(), opacity=0.9)
        ))
        fig2 = base_layout(fig2, "Estimated Latency Saved per Optimization")
        fig2.update_xaxes(title_text="Seconds Saved")
        fig2.update_yaxes(title_text="")
        fig2.update_traces(cliponaxis=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-hdr'>Full Playbook — Ranked by Impact</div>", unsafe_allow_html=True)

    playbook = [
        ("🔴", "P0", "Semantic cache for /knowledge-base/query",
         "Defects Search, Troubleshooting", "~50s", "~30%", "Low",
         "Add Redis + embedding cache. Threshold cosine sim > 0.92 for cache hit."),
        ("🔴", "P0", "Async Defects-Management-Agent + 30s timeout",
         "Defects Search, Troubleshooting", "~45–100s peak", "Neutral", "Medium",
         "Switch from sync await to async dispatch. Graceful fallback on timeout."),
        ("🟠", "P1", "Materialise /rma/execute-query-sddm patterns",
         "Troubleshooting", "~35s", "Neutral", "Medium",
         "Pre-compute top 20 RMA query patterns as DB views. 1h cache TTL."),
        ("🟠", "P1", "Merge case_update LLM calls (4.5 → 2)",
         "Case Update", "~18s", "~40%", "Low",
         "Single structured JSON output call replaces 3-step sequential chain."),
        ("🟠", "P1", "Parallelise case_create sub-agent calls",
         "Case Create", "~11s", "Neutral", "Low",
         "Case-Mgmt2 and SupportCase agents are independent — run concurrently."),
        ("🟡", "P2", "Intent pre-classifier before routing",
         "Troubleshooting", "~6s", "Minimal", "Medium",
         "Lightweight domain tagger reduces routing ambiguity resolution time."),
        ("🟡", "P2", "Async batch guardrails + session bypass",
         "All sources", "~2s", "Neutral", "Low",
         "0 blocks observed. Move pre-check off critical path. Add trusted-session mode."),
        ("🟢", "P3", "Rolling context window summarisation",
         "Case Create, License Mgmt", "~8s late turns", "~25%", "Medium",
         "After turn 3: last 2 turns verbatim + compressed summary of earlier turns."),
        ("🟢", "P3", "Trim action group payloads before LLM injection",
         "Troubleshooting", "~3–5s", "~15%", "Low",
         "Extract relevant fields from /rma schema before passing to LLM context."),
        ("🟢", "P3", "Cache routing decisions for similar queries",
         "All sources", "~1–2s", "Minimal", "Low",
         "Reuse classifier output for cosine sim > 0.95. TTL: 1h."),
    ]
    pb_df = pd.DataFrame(playbook, columns=[
        "","Priority","Optimization","Affected Sources",
        "Latency Saving","Cost Saving","Effort","Implementation Note"])
    st.dataframe(pb_df.drop(columns=[""]), use_container_width=True, hide_index=True,
                 column_config={
                     "Priority": st.column_config.TextColumn(width="small"),
                     "Effort":   st.column_config.TextColumn(width="small"),
                 })

    st.markdown("<hr>", unsafe_allow_html=True)

    col3, col4, col5 = st.columns(3)
    with col3:
        st.markdown("""
        <div class='callout callout-red'>
        <strong>🔴 P0 — Implement first</strong><br>
        Two fixes, zero model changes needed. Infrastructure + caching only.
        Combined impact: eliminates the 75–129s worst-case latency entirely.
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='callout callout-yellow'>
        <strong>🟠 P1 — Next sprint</strong><br>
        Requires prompt / agent flow changes. Merge LLM calls and parallelise
        sub-agents. Biggest ROI per engineering effort after P0.
        </div>""", unsafe_allow_html=True)
    with col5:
        st.markdown("""
        <div class='callout callout-green'>
        <strong>🟢 P3 — Ongoing</strong><br>
        Rolling summarisation and payload trimming compound in value as session
        length grows. Prioritise for long-session use cases (case_create, license_mgmt).
        </div>""", unsafe_allow_html=True)
