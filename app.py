import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import ast, json, numpy as np

st.set_page_config(page_title="AVA Bot · Analysis", page_icon="⚡", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: #0d1117; }

  .stTabs [data-baseweb="tab-list"] {
    gap: 2px; background: #161b22; border-radius: 8px;
    padding: 3px; border: 1px solid #21262d; width: fit-content;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 6px; border: none;
    color: #6e7681; font-size: 13px; font-weight: 500; padding: 7px 20px;
  }
  .stTabs [aria-selected="true"] {
    background: #21262d !important; color: #e6edf3 !important; font-weight: 600;
  }

  .kpi { background:#161b22; border:1px solid #21262d; border-radius:8px; padding:18px 20px; }
  .kpi-val { color:#e6edf3; font-size:26px; font-weight:700; line-height:1; }
  .kpi-lbl { color:#6e7681; font-size:11px; font-weight:600; text-transform:uppercase;
             letter-spacing:.7px; margin-bottom:5px; }
  .kpi-sub { color:#484f58; font-size:11px; margin-top:3px; }

  .tag-r { display:inline-block; background:rgba(248,81,73,.1); color:#f85149;
           border:1px solid rgba(248,81,73,.25); border-radius:4px; padding:1px 8px;
           font-size:11px; font-weight:600; }
  .tag-y { display:inline-block; background:rgba(210,153,34,.1); color:#d29922;
           border:1px solid rgba(210,153,34,.25); border-radius:4px; padding:1px 8px;
           font-size:11px; font-weight:600; }
  .tag-g { display:inline-block; background:rgba(63,185,80,.1); color:#3fb950;
           border:1px solid rgba(63,185,80,.25); border-radius:4px; padding:1px 8px;
           font-size:11px; font-weight:600; }
  .tag-b { display:inline-block; background:rgba(88,166,255,.1); color:#58a6ff;
           border:1px solid rgba(88,166,255,.25); border-radius:4px; padding:1px 8px;
           font-size:11px; font-weight:600; }

  .note { background:#161b22; border-left:3px solid #30363d; border-radius:0 6px 6px 0;
          padding:11px 16px; font-size:13px; color:#8b949e; line-height:1.6; }
  .note b { color:#c9d1d9; }
  .note-r { border-left-color:#f85149; }
  .note-y { border-left-color:#d29922; }
  .note-g { border-left-color:#3fb950; }
  .note-b { border-left-color:#58a6ff; }

  hr { border:none; border-top:1px solid #21262d; margin:20px 0; }

  .stDataFrame thead tr th { background:#161b22 !important; color:#8b949e !important; font-size:12px; }
  .stDataFrame tbody tr td { font-size:13px; }
  .stDataFrame { border:1px solid #21262d; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# ── palette ───────────────────────────────────────────────────────────────────
BG, CARD, BORDER, TEXT, MUTED = "#0d1117","#161b22","#21262d","#c9d1d9","#484f58"
COLS = ["#58a6ff","#3fb950","#d29922","#f85149","#a371f7","#ffa657"]
SRCS = ["Defects Search","Troubleshooting","Case Update","Case Create","Firmware Rec.","License Mgmt"]
CM   = dict(zip(SRCS, COLS))

def chart(fig, title="", sub=""):
    t = title + (f"<br><span style='font-size:12px;color:{MUTED};font-weight:400'>{sub}</span>" if sub else "")
    fig.update_layout(
        title=dict(text=t, font=dict(color=TEXT, size=14), x=0),
        paper_bgcolor=BG, plot_bgcolor=CARD,
        font=dict(color=TEXT, size=12), margin=dict(t=65,b=45,l=10,r=10),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, size=11)),
        hoverlabel=dict(bgcolor="#21262d", font_color="#e6edf3"),
    )
    return fig

# ── data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    df = pd.read_csv("output_datasets/logs_consolidated.csv", parse_dates=["timestamp"])
    def sp(v):
        if pd.isna(v): return None
        try: return ast.literal_eval(v)
        except:
            try: return json.loads(v)
            except: return None
    df["op"] = df["output"].apply(sp)
    df["eff"] = df["latency"] / df["totalTokens"] * 1000
    df["oratio"] = df["outputTokens"] / df["inputTokens"]
    rows = []
    for _, r in df.iterrows():
        for ev in (r["op"] or []):
            et = ev.get("type",""); tr = ev.get("trace",{})
            if et == "guardrail_trace":
                m = tr.get("metadata",{})
                rows.append({"src":r["source"],"iid":r["id"],"etype":"grd",
                    "ms":m.get("totalTimeMs"),"collab":None,"itok":None})
            elif et == "routing_classifier_trace":
                mio = tr.get("modelInvocationOutput",{}); m = mio.get("metadata",{})
                if m.get("totalTimeMs"):
                    u = m.get("usage",{})
                    rows.append({"src":r["source"],"iid":r["id"],"etype":"rc",
                        "ms":m.get("totalTimeMs"),"collab":None,"itok":u.get("inputTokens")})
                obs = tr.get("observation",{}); ac = obs.get("agentCollaboratorInvocationOutput",{})
                m2 = ac.get("metadata",{})
                if m2.get("totalTimeMs"):
                    rows.append({"src":r["source"],"iid":r["id"],"etype":"crt",
                        "ms":m2.get("totalTimeMs"),"collab":ac.get("agentCollaboratorName"),"itok":None})
            elif et == "orchestration_trace":
                mio = tr.get("modelInvocationOutput",{}); m = mio.get("metadata",{})
                if m.get("totalTimeMs"):
                    u = m.get("usage",{})
                    rows.append({"src":r["source"],"iid":r["id"],"etype":"llm",
                        "ms":m.get("totalTimeMs"),"collab":None,"itok":u.get("inputTokens")})
                obs = tr.get("observation",{})
                ac = obs.get("agentCollaboratorInvocationOutput",{}); m2 = ac.get("metadata",{})
                if m2.get("totalTimeMs"):
                    rows.append({"src":r["source"],"iid":r["id"],"etype":"ac",
                        "ms":m2.get("totalTimeMs"),"collab":ac.get("agentCollaboratorName"),"itok":None})
    tdf = pd.DataFrame(rows)
    tdf["ms"] = pd.to_numeric(tdf["ms"], errors="coerce")
    tdf["itok"] = pd.to_numeric(tdf["itok"], errors="coerce")
    return df, tdf

df, tdf = load()

SRC_MAP = {"defects_search":"Defects Search","troubleshooting":"Troubleshooting",
           "case_update":"Case Update","case_create":"Case Create",
           "firmware_recommendation":"Firmware Rec.","license_management":"License Mgmt"}

# # ── architecture diagram builder ──────────────────────────────────────────────
# @st.cache_data
# def build_arch_diagram():
#     BG_D    = "#0d1117"
#     CARD_D  = "#161b22"
#     BORDER_S= "#f85149"
#     BORDER_A= "#3fb950"
#     BORDER_C= "#58a6ff"
#     BORDER_P= "#d29922"
#     TEXT_H  = "#e6edf3"
#     TEXT_M  = "#8b949e"
#     TEXT_S  = "#6e7681"

#     fig = go.Figure()
#     fig.update_layout(
#         width=1500, height=1000,
#         paper_bgcolor=BG_D, plot_bgcolor=BG_D,
#         margin=dict(l=20, r=20, t=55, b=20),
#         xaxis=dict(visible=False, range=[0, 16]),
#         yaxis=dict(visible=False, range=[0, 11]),
#         title=dict(text="AVA Bot — System Architecture",
#                    font=dict(color=TEXT_H, size=17), x=0.5)
#     )

#     def box(x, y, w, h, bc, fill=CARD_D):
#         fig.add_shape(type="rect", x0=x, y0=y, x1=x+w, y1=y+h,
#                       line=dict(color=bc, width=1.8), fillcolor=fill, layer="below")

#     def lbl(x, y, text, size=11, color=TEXT_H, bold=False, sub=None):
#         t = f"<b>{text}</b>" if bold else text
#         if sub:
#             t += f"<br><span style='font-size:8.5px;color:{TEXT_S}'>{sub}</span>"
#         fig.add_annotation(x=x, y=y, text=t, font=dict(size=size, color=color),
#                            showarrow=False, align="center",
#                            xanchor="center", yanchor="middle")

#     def arr(x0, y0, x1, y1, color="#484f58", w=1.2):
#         fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0,
#                            xref="x", yref="y", axref="x", ayref="y",
#                            showarrow=True, arrowhead=2, arrowsize=1,
#                            arrowwidth=w, arrowcolor=color, text="")

#     def badge(x, y, text, color):
#         fig.add_annotation(x=x, y=y, text=f" {text} ",
#                            font=dict(size=8, color=color),
#                            bgcolor="rgba(0,0,0,0)",
#                            bordercolor=color, borderwidth=1, borderpad=2,
#                            showarrow=False, xanchor="center", yanchor="middle")

#     # ── USER ─────────────────────────────────────────────────────────
#     box(7.25, 9.9, 1.5, 0.65, BORDER_C, "#0d1720")
#     lbl(8.0, 10.22, "👤  User", 12, TEXT_H, bold=True)
#     arr(8.0, 9.9, 8.0, 9.15, BORDER_C, w=1.8)

#     # ── SUPERVISOR ───────────────────────────────────────────────────
#     box(4.9, 8.1, 6.2, 0.82, BORDER_S, "#1c1215")
#     lbl(8.0, 8.51, "Supervising Agent", 13, TEXT_H, bold=True,
#         sub="Guardrail Pre/Post  ·  Router Classifier  ·  LLM Orchestration")
#     badge(5.8,  8.98, "Guardrail  0.7s", BORDER_P)
#     badge(8.0,  8.98, "Router Classifier  1.2–7.7s", "#a371f7")
#     badge(10.2, 8.98, "LLM Orchestration", BORDER_C)

#     # ── RMA-CASE SUB-SUPERVISOR ───────────────────────────────────────
#     box(0.25, 6.15, 3.6, 0.75, BORDER_S, "#1c1215")
#     lbl(2.05, 6.525, "RMA-Case Mgmt Supervisor", 10.5, TEXT_H, bold=True,
#         sub="Case-Management2-Collaborator")
#     arr(6.0, 8.1, 2.05, 6.9, BORDER_S, w=1.5)

#     # ── LEAF AGENTS (row 2) ───────────────────────────────────────────
#     leaf = [
#         (4.35,  "LicenseMgmt Agent",          BORDER_A),
#         (6.2,   "FirmwareRecom. Agent",        BORDER_A),
#         (8.05,  "LivechatSupport Agent",       BORDER_A),
#         (9.9,   "KnowledgeSearch Agent",       BORDER_A),
#         (11.75, "Defect/TroubleshootSupport Agent", "#f85149"),
#     ]
#     for cx, name, bc in leaf:
#         box(cx-0.85, 6.15, 1.7, 0.75, bc, CARD_D)
#         lbl(cx, 6.525, name, 9, TEXT_H)
#         arr(7.2 + (cx-8.0)*0.25, 8.1, cx, 6.9, bc, w=1.3)

#     # ── COLLABORATOR AGENTS (row 3) ───────────────────────────────────
#     collabs = [
#         (1.05,  "Case MgmtAgent",            BORDER_C),
#         (3.0,   "RMA MgmtAgent",             BORDER_C),
#         (4.9,   "license-assistcollaborator",BORDER_C),
#         (6.75,  "Firmware-Assistcollaborator",BORDER_C),
#         (8.9,   "SupportCase-MgmtAgent",     BORDER_C),
#         (10.85, "Defects-MgmtAgent",         "#f85149"),
#         (12.75, "Case-Mgmt2collaborator",    BORDER_C),
#     ]
#     for cx, name, bc in collabs:
#         box(cx-0.82, 4.1, 1.64, 0.76, bc, CARD_D)
#         lbl(cx, 4.48, name, 9, TEXT_H)

#     # arrows: sub-supervisor → its two agents
#     arr(1.4, 6.15, 1.05, 4.86, BORDER_C, w=1.3)
#     arr(2.7, 6.15, 3.0,  4.86, BORDER_C, w=1.3)
#     # leaf → collaborator
#     arr(4.35, 6.15, 4.9,  4.86, BORDER_C, w=1.2)
#     arr(6.2,  6.15, 6.75, 4.86, BORDER_C, w=1.2)
#     arr(8.05, 6.15, 8.9,  4.86, BORDER_C, w=1.2)
#     arr(9.9,  6.15, 10.85,4.86, "#f85149", w=1.4)
#     arr(11.75,6.15, 10.85,4.86, "#f85149", w=1.4)
#     arr(12.1, 6.15, 12.75,4.86, BORDER_C, w=1.2)

#     # ── ACTION GROUPS / APIs (row 4) ──────────────────────────────────
#     ags = [
#         (1.05,  "case_mgmt_actions",
#          ["/create-case", "/verify-case", "/update-case", "/get-url"], BORDER_P),
#         (3.0,   "past_cases_apis",
#          ["/rma/get-schema-sddm", "/rma/execute-query-sddm  37s"], BORDER_P),
#         (4.9,   "license_action_group",
#          ["/validate-product-line", "/validate-order-number", "/get-license-url"], BORDER_P),
#         (6.75,  "firmware_apis",
#          ["/central/get_recommendation"], BORDER_P),
#         (8.9,   "case_mgmt_actions",
#          ["/create-case", "/verify-case", "/case-attachment"], BORDER_P),
#         (10.85, "knowledge_base_apis",
#          ["/knowledge-base/query  ⚠ 50.6s"], "#f85149"),
#         (12.75, "case_mgmt_actions",
#          ["/verify-case", "/update-case"], BORDER_P),
#     ]
#     for cx, ag_name, apis, bc in ags:
#         h = 0.22 + 0.23 * len(apis)
#         y0 = 1.95
#         box(cx-0.88, y0, 1.76, h, bc, "#0d1520" if bc == "#f85149" else CARD_D)
#         lbl(cx, y0+h-0.14, ag_name, 8, bc, bold=True)
#         for i, api in enumerate(apis):
#             col = "#f85149" if "⚠" in api else TEXT_S
#             lbl(cx, y0+h-0.36-i*0.22, api.replace("⚠","").strip(), 7.5, col)
#         arr(cx, 4.1, cx, y0+h, bc, w=1.1)

#     # ── USE CASES (row 5) ─────────────────────────────────────────────
#     ucs = [
#         (1.05,  "case_createcase_update",         BORDER_A),
#         (3.0,   "troubleshooting",                  BORDER_P),
#         (4.9,   "license_mgmt",                     BORDER_A),
#         (6.75,  "firmware_rec.",                    BORDER_A),
#         (8.9,   "case_createcase_update",         BORDER_A),
#         (10.85, "defects_searchtroubleshooting",  "#f85149"),
#         (12.75, "case_updatetroubleshooting",     BORDER_A),
#     ]
#     for cx, uc, bc in ucs:
#         box(cx-0.74, 0.5, 1.48, 0.62, bc, CARD_D)
#         lbl(cx, 0.81, uc, 7.5, bc)
#         arr(cx, 1.95, cx, 1.12, bc, w=1.0)

#     # ── ROW LABELS ────────────────────────────────────────────────────
#     for y, txt in [(10.22,"User"),(8.51,"Supervisor"),(6.52,"Leaf Agents"),
#                    (4.48,"Collaborators"),(2.5,"Action Groups"),(0.81,"Use Cases")]:
#         fig.add_annotation(x=0.05, y=y, text=txt, font=dict(size=8, color=TEXT_S),
#                            showarrow=False, xanchor="left", yanchor="middle", textangle=-90)

#     # ── LEGEND ────────────────────────────────────────────────────────
#     box(13.9, 8.3, 1.75, 2.3, "#21262d", "#0d1117")
#     lbl(14.78, 10.68, "Legend", 9.5, TEXT_M, bold=True)
#     items = [
#         (BORDER_S, "Supervisor"),
#         (BORDER_A, "Agent / Collaborator"),
#         (BORDER_C, "Collab. (internal)"),
#         (BORDER_P, "Action Group / API"),
#         ("#f85149","⚠ Bottleneck"),
#     ]
#     for i, (c, txt) in enumerate(items):
#         yy = 10.3 - i*0.42
#         fig.add_shape(type="rect", x0=14.0, y0=yy-0.12, x1=14.22, y1=yy+0.12,
#                       line=dict(color=c, width=1.5), fillcolor=CARD_D)
#         lbl(14.85, yy, txt, 8.5, TEXT_M)

#     # ── BOTTLENECK CALLOUTS ───────────────────────────────────────────
#     fig.add_annotation(x=10.85, y=3.0, text="⚠ 50.6s avg · no cache",
#         font=dict(size=8, color="#f85149"), showarrow=False,
#         bgcolor="rgba(248,81,73,0.08)", bordercolor="#f85149", borderwidth=1, borderpad=2)
#     fig.add_annotation(x=10.85, y=5.5, text="75–129s roundtrip",
#         font=dict(size=8, color="#f85149"), showarrow=False,
#         bgcolor="rgba(248,81,73,0.07)", bordercolor="#f85149", borderwidth=1, borderpad=2)
#     fig.add_annotation(x=3.0, y=2.85, text="37s · no materialised views",
#         font=dict(size=7.5, color=BORDER_P), showarrow=False,
#         bgcolor="rgba(210,153,34,0.07)", bordercolor=BORDER_P, borderwidth=1, borderpad=2)
#     fig.add_annotation(x=8.0, y=7.63,
#         text="Guardrail: 2.5–3.2s · 0 blocks  |  Router: 1.2–7.7s (ambiguity-driven)",
#         font=dict(size=8, color=BORDER_P), showarrow=False)

#     return fig

# ══════════════════════════════════════════════════════════════════════════════
# TABS  — order: LLMs & Tokens | Architecture & Routing | System Architecture | Overview
# ══════════════════════════════════════════════════════════════════════════════
T2, T3, T1 = st.tabs([
    "  LLMs & Parameters  ",
    "  Architecture & Routing  ",
    # "  System Architecture  ",
    "  Optimization Overview  "
])

# # ─────────────────────────────────────────
# # TAB 4  SYSTEM ARCHITECTURE
# # ─────────────────────────────────────────
# with T4:
#     st.markdown("#### System Architecture")
#     st.markdown(
#         "<div style='color:#6e7681;font-size:13px;margin-bottom:20px'>"
#         "Derived from initial architecture diagram + 199 trace events · 22 interactions · 6 sources"
#         "</div>", unsafe_allow_html=True)

#     # 5-column layer legend
#     c1, c2, c3, c4, c5 = st.columns(5)
#     for col, lbl_txt, val, sub, cls in [
#         (c1, "Layer 1", "Supervising Agent",    "Guardrail · Router · LLM Orch.", "note note-r"),
#         (c2, "Layer 2", "5 Leaf Agents",        "Direct children of Supervisor",  "note note-g"),
#         (c3, "Layer 3", "7 Collaborators",      "Named sub-agents per domain",    "note note-b"),
#         (c4, "Layer 4", "7 Action Groups",      "API endpoints per collaborator", "note note-y"),
#         (c5, "Layer 5", "6 Use Cases",          "Source → agent chain mapping",   "note"),
#     ]:
#         col.markdown(
#             f"<div class='{cls}' style='padding:12px 14px'>"
#             f"<div style='color:#6e7681;font-size:10px;font-weight:600;text-transform:uppercase;"
#             f"letter-spacing:.6px;margin-bottom:4px'>{lbl_txt}</div>"
#             f"<div style='color:#e6edf3;font-size:14px;font-weight:700;line-height:1'>{val}</div>"
#             f"<div style='color:#484f58;font-size:11px;margin-top:3px'>{sub}</div>"
#             f"</div>", unsafe_allow_html=True)

#     st.markdown("<hr>", unsafe_allow_html=True)

#     arch_fig = build_arch_diagram()
#     st.plotly_chart(arch_fig, use_container_width=True)

#     st.markdown("<hr>", unsafe_allow_html=True)


# ─────────────────────────────────────────
# TAB 1  OVERVIEW
# ─────────────────────────────────────────
with T1:
    st.markdown("#### Optimization Overview")
    st.markdown("<div style='color:#6e7681;font-size:13px;margin-bottom:20px'>Ranked fixes across all bottlenecks · estimated latency & cost impact</div>", unsafe_allow_html=True)

    # col_l, col_r = st.columns(2)

    # with col_l:
    #     before  = [81.76, 79.69, 36.72, 26.50, 19.38, 13.38]
    #     after   = [4.5,   12.0,  14.0,  12.0,  16.0,  10.5 ]
    #     fig = go.Figure()
    #     fig.add_trace(go.Bar(name="Current", x=SRCS, y=before,
    #                          marker_color="#f85149", opacity=0.75,
    #                          text=[f"{v:.0f}s" for v in before], textposition="outside"))
    #     fig.add_trace(go.Bar(name="After All Fixes", x=SRCS, y=after,
    #                          marker_color="#3fb950", opacity=0.85,
    #                          text=[f"{v:.0f}s" for v in after], textposition="outside"))
    #     fig = chart(fig, "Current vs Projected Latency After Fixes",
    #                 "Defects Search: 81s → 4.5s · Troubleshooting: 79s → 12s")
    #     fig.update_layout(barmode="group",
    #         legend=dict(orientation="h", y=1.1, xanchor="center", x=0.5))
    #     fig.update_xaxes(title_text="")
    #     fig.update_yaxes(title_text="Seconds")
    #     fig.update_traces(cliponaxis=False)
    #     st.plotly_chart(fig, use_container_width=True)

    # with col_r:
    #     fixes   = ["Cache kb/query","Async Defects-Mgmt","Materialise RMA","Merge LLM calls",
    #                "Parallelise agents","Intent pre-clf","Async guardrails","Rolling ctx window"]
    #     savings = [50.6, 45.0, 35.3, 18.0, 11.3, 6.3, 2.1, 8.0]
    #     s_colors = ["#f85149","#f85149","#d29922","#d29922","#d29922","#a371f7","#a371f7","#3fb950"]
    #     fig2 = go.Figure(go.Bar(
    #         x=savings, y=fixes, orientation="h",
    #         text=[f"{v:.0f}s" for v in savings], textposition="outside",
    #         marker=dict(color=s_colors, opacity=0.88)
    #     ))
    #     fig2 = chart(fig2, "Latency Saved per Fix", "Sorted by impact · color = priority tier")
    #     fig2.update_xaxes(title_text="Seconds Saved")
    #     fig2.update_yaxes(title_text="", autorange="reversed")
    #     fig2.update_traces(cliponaxis=False)
    #     st.plotly_chart(fig2, use_container_width=True)

    # st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("<div style='color:#e6edf3;font-size:13px;font-weight:600;margin-bottom:12px'>Full Optimization Playbook</div>", unsafe_allow_html=True)
    opts = pd.DataFrame([
        ["🔴 P0","Semantic cache · /knowledge-base/query",   "Defects Search, Troubleshooting","−50s",  "−30%","Low",   "Redis + embedding cache · cosine sim > 0.92"],
        ["🔴 P0","Async Defects-Mgmt + 30s timeout",        "Defects Search, Troubleshooting","−100s peak","—",  "Medium","Async dispatch · graceful fallback on timeout"],
        ["🟠 P1","Materialise RMA query views",              "Troubleshooting",                "−35s",  "—",   "Medium","Pre-compute top 20 patterns as DB views · 1h TTL"],
        ["🟠 P1","Merge case_update LLM calls (4.5 → 2)",   "Case Update",                    "−18s",  "−40%","Low",   "Single structured JSON output call"],
        ["🟠 P1","Parallelise case_create sub-agents",       "Case Create",                    "−11s",  "—",   "Low",   "Case-Mgmt2 + SupportCase are independent — run concurrently"],
        ["🟡 P2","Intent pre-classifier",                   "Troubleshooting",                "−6s",   "—",   "Medium","Lightweight domain tagger before routing step"],
        ["🟡 P2","Async guardrails + session bypass",        "All sources",                    "−2s",   "—",   "Low",   "0 blocks observed · move off critical path"],
        ["🟢 P3","Rolling context window summarisation",     "Case Create, License Mgmt",      "−8s",   "−25%","Medium","Last 2 turns + compressed summary after turn 3"],
        ["🟢 P3","Trim action group payloads",               "Troubleshooting",                "−3s",   "−15%","Low",   "Extract relevant fields before LLM context injection"],
        ["🟢 P3","Cache routing decisions",                  "All sources",                    "−1s",   "—",   "Low",   "Reuse classifier output · cosine sim > 0.95 · 1h TTL"],
    ], columns=["Priority","Fix","Affected","Latency Δ","Cost Δ","Effort","How"])
    st.dataframe(opts, use_container_width=True, hide_index=True,
                 column_config={
                     "Priority": st.column_config.TextColumn(width="small"),
                     "Latency Δ": st.column_config.TextColumn(width="small"),
                     "Cost Δ":   st.column_config.TextColumn(width="small"),
                     "Effort":   st.column_config.TextColumn(width="small"),
                 })

    st.markdown("<hr>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown("<div class='note note-r'><b>🔴 P0 — Do first</b><br>Two infra changes, zero model edits. "
                "Async + cache eliminates the 75–129s worst-case latency entirely.</div>", unsafe_allow_html=True)
    c2.markdown("<div class='note note-y'><b>🟠 P1 — Next sprint</b><br>Agent flow changes. "
                "Merge LLM calls + parallelise sub-agents. Biggest ROI per engineering hour after P0.</div>", unsafe_allow_html=True)
    c3.markdown("<div class='note note-g'><b>🟢 P3 — Ongoing</b><br>Summarisation + payload trimming "
                "compound in value as session length grows. Prioritise for case_create + license_mgmt.</div>", unsafe_allow_html=True)



# ─────────────────────────────────────────
# TAB 2  LLMs & TOKENS
# ─────────────────────────────────────────
with T2:
    st.markdown("#### LLMs & Parameters")
    st.markdown("<div style=\'color:#6e7681;font-size:13px;margin-bottom:20px\'>LLM call depth, per-call latency, and context window growth</div>", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col, lbl_txt, val, sub in [
        (c1,"Peak LLM Calls","4.5×","Case Update per interaction"),
        (c2,"Slowest LLM Call","19s avg","Troubleshooting (large context)"),
        (c3,"Max Token Count","44,687","Case Create turn 6"),
        (c4,"Tokens ↔ Cost","r = 0.99","p < 0.0001 · near-perfect"),
    ]:
        col.markdown(f"<div class=\'kpi\'><div class=\'kpi-lbl\'>{lbl_txt}</div>"
                     f"<div class=\'kpi-val\'>{val}</div><div class=\'kpi-sub\'>{sub}</div></div>",
                     unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Summary table + Latency chart (NEW) ───────────────────────────────────
    summary_df = pd.DataFrame([
        ["license_management",    7, 13.4, 16630, 0.053],
        ["case_create",           8, 26.5, 27470, 0.091],
        ["case_update",           2, 36.7, 35721, 0.116],
        ["firmware_recommendation",2,19.4,  9957, 0.034],
        ["troubleshooting",       2, 79.7, 16375, 0.060],
        ["defects_search",        1, 81.8, 14590, 0.056],
    ], columns=["Source","Interactions","Avg Latency","Avg Tokens","Avg Cost"])

    tbl_col, chart_col = st.columns([1, 1.6])

    with tbl_col:
        st.markdown("<div style=\'color:#e6edf3;font-size:13px;font-weight:600;margin-bottom:10px\'>Summary</div>", unsafe_allow_html=True)
        display_df = summary_df.copy()
        display_df["Avg Latency"] = display_df["Avg Latency"].apply(lambda x: f"{x:.1f}s")
        display_df["Avg Tokens"] = display_df["Avg Tokens"].apply(lambda x: f"{x:,}")
        display_df["Avg Cost"]     = display_df["Avg Cost"].apply(lambda x: f"${x:.3f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True,
                     column_config={
                         "Source":          st.column_config.TextColumn(width="medium"),
                         "Interactions":    st.column_config.NumberColumn(width="small"),
                         "Avg Latency": st.column_config.TextColumn(width="small"),
                         "Avg Tokens":st.column_config.TextColumn(width="small"),
                         "Avg Cost":    st.column_config.TextColumn(width="small"),
                     })

    with chart_col:
        # Latency bar with error bars — mirrors the attached chart
        lat_err = {"license_management":2.1, "case_create":4.8,
                   "case_update":1.5, "firmware_recommendation":3.7,
                   "troubleshooting":63.8, "defects_search":0.0}
        src_order  = ["license_management","case_create","case_update",
                      "firmware_recommendation","troubleshooting","defects_search"]
        lbl_order  = ["License Mgmt","Case Create","Case Update",
                      "Firmware Rec.","Troubleshooting","Defects Search"]
        lat_vals   = [summary_df[summary_df["Source"]==s]["Avg Latency"].values[0] for s in src_order]
        err_vals   = [lat_err[s] for s in src_order]
        bar_clrs   = [CM[l] for l in lbl_order]

        fig_lat = go.Figure(go.Bar(
            x=lbl_order, y=lat_vals,
            text=[f"{v:.1f}s" for v in lat_vals], textposition="outside",
            marker=dict(color=bar_clrs, opacity=0.88),
            error_y=dict(type="data", array=err_vals, visible=True,
                         color="#6e7681", thickness=1.4, width=5),
        ))
        fig_lat = chart(fig_lat, "Avg Response Latency",
                        "Error bars = standard error · troubleshooting & defects_search are outliers")
        fig_lat.update_xaxes(title_text="Source")
        fig_lat.update_yaxes(title_text="Avg Latency (s)")
        fig_lat.update_traces(cliponaxis=False)
        st.plotly_chart(fig_lat, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1, col2 = st.columns(2)

    with col1:
        llm_calls = [4.5, 2.0, 2.75, 2.0, 1.5, 1.57]
        llm_ms    = [4462, 18999, 6639, 11553, 6693, 4370]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Avg Calls / Interaction", x=SRCS, y=llm_calls,
                             text=[f"{v:.1f}" for v in llm_calls], textposition="outside",
                             marker_color=[CM[s] for s in SRCS], opacity=0.88))
        fig = chart(fig, "LLM Calls per Interaction",
                    "case_update chains 4.5 sequential calls — most expensive flow")
        fig.update_yaxes(title_text="Calls", range=[0, 5.8])
        fig.update_xaxes(title_text="")
        fig.update_traces(cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure(go.Bar(
            x=SRCS, y=llm_ms,
            text=[f"{v//1000}s" for v in llm_ms], textposition="outside",
            marker=dict(color=[CM[s] for s in SRCS], opacity=0.88)
        ))
        fig2 = chart(fig2, "Avg Time per LLM Call",
                     "troubleshooting: 19s/call — large context dump slows reasoning")
        fig2.update_yaxes(title_text="ms per call")
        fig2.update_xaxes(title_text="")
        fig2.update_traces(cliponaxis=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        turn_data = {
            "Case Create":   [6929, 17217, 30386, 35213, 36699, 44687],
            "Case Update":   [26402, 41039],
            "Firmware Rec.": [7366, 12547],
            "License Mgmt":  [4453, 10005, 13028, 14776, 23625, 31723],
        }
        fig3 = go.Figure()
        for label_k, tokens in turn_data.items():
            fig3.add_trace(go.Scatter(
                x=list(range(1, len(tokens)+1)), y=tokens,
                mode="lines+markers", name=label_k,
                line=dict(color=CM.get(label_k,"#58a6ff"), width=2),
                marker=dict(size=6),
                hovertemplate=f"{label_k} · Turn %{{x}}: %{{y:,}} tokens<extra></extra>"
            ))
        fig3.add_hrect(y0=32000, y1=46000,
                       fillcolor="rgba(248,81,73,0.05)", line_width=0,
                       annotation_text="High-cost zone",
                       annotation_font_color="#f85149",
                       annotation_position="top left")
        fig3 = chart(fig3, "Token Growth Across Session Turns",
                     "Full history appended each turn — no pruning")
        fig3.update_xaxes(title_text="Turn", dtick=1)
        fig3.update_yaxes(title_text="Total Tokens")
        fig3.update_layout(legend=dict(orientation="h", y=1.1, xanchor="center", x=0.5))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("<div style='color:#e6edf3;font-size:13px;font-weight:600;margin-bottom:12px'>Key Findings</div>", unsafe_allow_html=True)
        findings = [
            ("note note-r",
             "<b>case_update: 4.5 sequential LLM calls</b><br>"
             "verify-case → get-url → update-case → confirm run one after another. "
             "Merging to a single structured-output call saves ~18s and ~40% cost."),
            ("note note-y",
             "<b>Troubleshooting: 19s per LLM call</b><br>"
             "Raw RMA schema dumps are injected into context before reasoning. "
             "Trimming to relevant fields reduces input tokens ~35% and speeds each call."),
            ("note note-y",
             "<b>Token snowball: +27–70% per turn</b><br>"
             "No context pruning in place. Firmware Rec. grows fastest (+70%/turn). "
             "Case Create hits 44k tokens by turn 6 — well into diminishing returns."),
            ("note note-b",
             "<b>Tokens drive cost perfectly (r=0.99)</b><br>"
             "Every token saved is a direct cost reduction. Rolling summarisation "
             "after turn 3 can cut case_create tokens by ~66% from turn 4 onwards."),
        ]
        for cls, txt in findings:
            st.markdown(f"<div class='{cls}' style='margin-bottom:10px'>{txt}</div>",
                        unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("<div style='color:#e6edf3;font-size:13px;font-weight:600;margin-bottom:10px'>Token Growth Rates</div>", unsafe_allow_html=True)
        tg = pd.DataFrame([
            ["Firmware Rec.", "+70.3%/turn", "7,366", "12,547"],
            ["Case Update",   "+35.0%/turn", "26,402","41,039"],
            ["License Mgmt",  "+33.0%/turn", "4,453", "31,723"],
            ["Case Create",   "+26.8%/turn", "6,929", "44,687"],
        ], columns=["Source","Growth/Turn","Min Tokens","Max Tokens"])
        st.dataframe(tg, use_container_width=True, hide_index=True)



# ─────────────────────────────────────────
# TAB 3  ARCHITECTURE & ROUTING
# ─────────────────────────────────────────
with T3:
    st.markdown("#### Agent Architecture, Guardrail & Routing")
    st.markdown("<div style=\'color:#6e7681;font-size:13px;margin-bottom:20px\'>Sub-agent latency, guardrail overhead, and routing classifier performance</div>", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col, lbl_txt, val, sub in [
        (c1,"Worst Sub-Agent","129s","Defects-Mgmt in Troubleshooting"),
        (c2,"Sub-Agent Share","93%","of Defects Search total latency"),
        (c3,"Guardrail Blocks","0 / 82","all actions returned NONE"),
        (c4,"Routing Slowdown","6.4×","Troubleshooting vs License Mgmt"),
    ]:
        col.markdown(f"<div class=\'kpi\'><div class=\'kpi-lbl\'>{lbl_txt}</div>"
                     f"<div class=\'kpi-val\'>{val}</div><div class=\'kpi-sub\'>{sub}</div></div>",
                     unsafe_allow_html=True)

    # ── NEW CHARTS 2 & 3: LLM Split + Latency Stack ───────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        # LLM vs Non-LLM Share (100% stacked)
        src_order = ["Case Create","Case Update","Defects Search","Firmware Rec.","License Mgmt","Troubleshooting"]
        llm_pct     = [70, 55, 28, 53, 50, 58]
        non_llm_pct = [30, 45, 72, 47, 50, 42]

        fig_split = go.Figure()
        fig_split.add_trace(go.Bar(
            name="Non-LLM Overhead", x=src_order, y=non_llm_pct,
            marker_color="#f85149", opacity=0.85,
            hovertemplate="%{x}<br>Non-LLM: %{y}%<extra></extra>",
        ))
        fig_split.add_trace(go.Bar(
            name="LLM Time", x=src_order, y=llm_pct,
            marker_color="#4ec9e1", opacity=0.88,
            hovertemplate="%{x}<br>LLM: %{y}%<extra></extra>",
        ))
        fig_split = chart(fig_split, "LLM vs Non-LLM Share of Total Latency",
                          "defects_search: 72% non-LLM — API/retrieval bottleneck")
        fig_split.update_layout(
            barmode="stack",
            legend=dict(orientation="h", y=1.1, xanchor="center", x=0.5),
        )
        fig_split.update_xaxes(title_text="Source")
        fig_split.update_yaxes(title_text="% of Total Latency", range=[0, 105])
        st.plotly_chart(fig_split, use_container_width=True)

    with col_b:
        # Latency Stack by Pipeline Step & Source
        stack_srcs  = ["Case Create","Case Update","Defects Search","Firmware Rec.","License Mgmt","Troubleshoot"]
        stack_steps = {
            "Guardrail (pre)":      [1.3, 1.4, 1.3, 1.6, 1.4, 1.3],
            "Routing Classifier":   [1.5, 2.0, 3.7, 1.8, 1.2, 7.8],
            "Action Group / API":   [14.5,15.0,50.6,14.9, 7.8,55.1],
            "Agent Collaborator":   [11.3,12.0, 0.0, 0.0, 0.0, 5.2],
            "Sub-Agent Roundtrip":  [23.2,32.5,75.9,14.9, 9.9,79.3],
            "LLM Orchestration":    [24.7,34.0,34.6,13.6,12.4,38.0],
            "Guardrail (post)":     [3.2, 3.3, 2.7, 3.6, 2.6, 3.1],
        }
        step_colors = {
            "Guardrail (pre)":     "#4ec9e1",
            "Routing Classifier":  "#f85149",
            "Action Group / API":  "#3fb950",
            "Agent Collaborator":  "#6e8ea6",
            "Sub-Agent Roundtrip": "#c9a84c",
            "LLM Orchestration":   "#c0392b",
            "Guardrail (post)":    "#7b241c",
        }
        fig_stack = go.Figure()
        for step, vals in stack_steps.items():
            fig_stack.add_trace(go.Bar(
                name=step, x=stack_srcs, y=vals,
                marker_color=step_colors[step], opacity=0.9,
                hovertemplate=f"{step}: %{{y:.1f}}s<extra></extra>",
            ))
        fig_stack = chart(fig_stack, "Latency Stack by Pipeline Step & Source",
                          "Sub-agent roundtrip dominates defects_search & troubleshooting")
        fig_stack.update_layout(
            barmode="stack",
            legend=dict(orientation="h", y=-0.22, xanchor="center", x=0.5,
                        font=dict(size=10)),
            margin=dict(b=100),
        )
        fig_stack.update_xaxes(title_text="Source")
        fig_stack.update_yaxes(title_text="Avg Time (s)")
        st.plotly_chart(fig_stack, use_container_width=True)


    st.markdown("<hr>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        c_data = [
            ("License Assist → License Mgmt",    9977),
            ("SupportCase → Case Create",        11332),
            ("Firmware Assist → Firmware Rec.",  14870),
            ("SupportCase → Case Update",        20488),
            ("Case-Mgmt2 → Case Create",         23232),
            ("Case-Mgmt2 → Case Update",         32530),
            ("Defects-Mgmt → Defects Search",    75946),
            ("Defects-Mgmt → Troubleshooting",  129347),
        ]
        cd = pd.DataFrame(c_data, columns=["agent","ms"])
        bar_colors = ["#6e7681","#6e7681","#6e7681","#d29922","#d29922","#d29922","#f85149","#f85149"]
        fig = go.Figure(go.Bar(
            x=cd["ms"], y=cd["agent"], orientation="h",
            text=[f"{v/1000:.0f}s" for v in cd["ms"]],
            textposition="outside",
            marker=dict(color=bar_colors, opacity=0.88)
        ))
        fig = chart(fig, "Sub-Agent Roundtrip Latency",
                    "Red = critical · Defects-Mgmt is the single point of failure")
        fig.update_xaxes(title_text="ms")
        fig.update_traces(cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        grd = {"License Mgmt":2870,"Case Create":2643,"Troubleshooting":2504,
               "Defects Search":2685,"Case Update":2856,"Firmware Rec.":3224}
        rc  = {"License Mgmt":1218,"Case Create":1542,"Firmware Rec.":1772,
               "Case Update":2028,"Defects Search":3668,"Troubleshooting":7752}
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="Guardrail",
            x=list(grd.keys()), y=list(grd.values()),
            text=[f"{v}ms" for v in grd.values()], textposition="inside",
            marker_color="#d29922", opacity=0.8
        ))
        fig2.add_trace(go.Bar(
            name="Routing Classifier",
            x=list(rc.keys()), y=list(rc.values()),
            text=[f"{v}ms" for v in rc.values()], textposition="inside",
            marker_color="#a371f7", opacity=0.8
        ))
        fig2 = chart(fig2, "Guardrail & Routing Classifier Overhead",
                     "Both are pure overhead — 0 blocks, same token load")
        fig2.update_layout(barmode="group",
            legend=dict(orientation="h", y=1.1, xanchor="center", x=0.5))
        fig2.update_xaxes(title_text="")
        fig2.update_yaxes(title_text="ms")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── NEW CHART 1: Action Group API Endpoint Latency (full width) ────────────
    api_labels = [
        "License Mgmt: validate-product-line",
        "License Mgmt: validate-order-number",
        "Firmware Rec.: get_recommendation",
        "Troubleshoot: knowledge-base/query",
        "Troubleshoot: get-schema-sddm",
        "Case Update: verify-case",
        "Case Update: update-case",
        "Case Create: verify-case",
        "Troubleshoot: execute-query-sddm",
        "Defects Search: knowledge-base/query",
    ]
    api_ms = [1100, 3100, 4400, 5100, 13700, 8600, 10000, 12800, 37300, 50600]
    api_src = [
        "License Mgmt","License Mgmt","Firmware Rec.","Troubleshooting",
        "Troubleshooting","Case Update","Case Update","Case Create",
        "Troubleshooting","Defects Search",
    ]
    api_colors = [CM[s] for s in api_src]

    fig_api = go.Figure(go.Bar(
        x=api_ms, y=api_labels, orientation="h",
        text=[f"{v/1000:.1f}s" for v in api_ms], textposition="outside",
        marker=dict(color=api_colors, opacity=0.88),
        customdata=api_src,
        hovertemplate="%{y}<br>Latency: %{text}<br>Source: %{customdata}<extra></extra>",
    ))
    # invisible legend traces for source grouping
    for src_name, clr in CM.items():
        fig_api.add_trace(go.Bar(
            x=[None], y=[None], name=src_name,
            marker_color=clr, showlegend=True
        ))
    fig_api = chart(fig_api, "Action Group API Endpoint Latency",
                    "knowledge-base/query takes 50.6s — primary bottleneck")
    fig_api.update_layout(
        legend=dict(orientation="h", y=1.08, xanchor="center", x=0.5,
                    title=dict(text="source  ", font=dict(color=MUTED, size=11))),
        xaxis=dict(title_text="Latency (ms)"),
        barmode="overlay",
    )
    fig_api.update_traces(cliponaxis=False)
    st.plotly_chart(fig_api, use_container_width=True)


    st.markdown("<hr>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<div style=\'color:#e6edf3;font-size:13px;font-weight:600;margin-bottom:12px\'>Key Findings</div>", unsafe_allow_html=True)
        arch_findings = [
            ("note note-r",
             "<b>Defects-Management-Agent: synchronous blocking chain</b><br>"
             "Called synchronously, no timeout, no cache. "
             "Internally triggers /knowledge-base/query (50.6s). "
             "Single root cause of the two slowest sources."),
            ("note note-r",
             "<b>case_create sub-agents run sequentially</b><br>"
             "Case-Mgmt2 (23.2s) and SupportCase (11.3s) are invoked one after the other "
             "despite being logically independent. Running in parallel saves ~11s."),
            ("note note-y",
             "<b>Guardrail: 2.5–3.2s fixed tax, zero value observed</b><br>"
             "82 guardrail events in the dataset. Every single one returned NONE. "
             "They run synchronously on the critical path before routing."),
            ("note note-b",
             "<b>Routing: same token load, 6× latency difference</b><br>"
             "Troubleshooting (7.75s) vs License Mgmt (1.22s) — token counts are similar "
             "(~3.2k–4.3k). Gap is query ambiguity spanning multiple agent domains."),
        ]
        for cls, txt in arch_findings:
            st.markdown(f"<div class=\'{cls}\' style=\'margin-bottom:10px\'>{txt}</div>",
                        unsafe_allow_html=True)

    with col4:
        st.markdown("<div style=\'color:#e6edf3;font-size:13px;font-weight:600;margin-bottom:12px\'>Fixes</div>", unsafe_allow_html=True)
        arch_fixes = [
            ("🔴", "Async Defects-Mgmt + semantic cache",
             "Async dispatch with 30s timeout. Redis cache on /knowledge-base/query "
             "(cosine sim > 0.92). Eliminates 50–129s worst-case latency."),
            ("🔴", "Materialise RMA query views",
             "/rma/execute-query-sddm (37s) → pre-computed DB views for top 20 patterns. "
             "Drops to < 2s for known queries."),
            ("🟠", "Parallelise case_create sub-agents",
             "Case-Mgmt2 and SupportCase are independent — "
             "fire concurrently. Saves ~11s at zero cost."),
            ("🟡", "Move guardrails off critical path",
             "Run pre-check async in parallel with routing rather than before it. "
             "Add trusted-session bypass for authenticated users. Saves ~2s per call."),
            ("🟡", "Intent pre-classifier for troubleshooting",
             "Lightweight domain tagger (< 500ms) before the routing step. "
             "Router confirms rather than discovers — cuts 7.75s down to ~1.5s."),
        ]
        for sev, title, desc in arch_fixes:
            st.markdown(
                f"<div style=\'background:#161b22;border:1px solid #21262d;border-radius:7px;"
                f"padding:13px 16px;margin-bottom:9px\'>"
                f"<div style=\'color:#e6edf3;font-size:13px;font-weight:600;margin-bottom:3px\'>"
                f"{sev} {title}</div>"
                f"<div style=\'color:#6e7681;font-size:12px;line-height:1.6\'>{desc}</div></div>",
                unsafe_allow_html=True
            )

