import streamlit as st
import pickle
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="PL Match Predictor",
    page_icon="⚽",
    layout="wide",
)

# ─────────────────────────────────────────
# THEME CONSTANTS
# ─────────────────────────────────────────
BG      = "#060810"
PLOT_BG = "#0d0f1a"
CARD    = "#111420"
BORDER  = "rgba(255,255,255,0.09)"
ACCENT  = "#38bdf8"
GREEN   = "#22c55e"
RED     = "#ef4444"
PURPLE  = "#a78bfa"
GOLD    = "#f59e0b"
MUTED   = "#8892b0"
TEXT    = "#f0f2ff"
GRID    = "rgba(255,255,255,0.05)"

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@700;900&family=Barlow:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Barlow', sans-serif;
    background-color: {BG} !important;
    color: {TEXT};
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.stApp {{ background: {BG}; }}
.block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 4rem !important;
    max-width: 1200px !important;
}}

/* Selectbox */
div[data-baseweb="select"] > div {{
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: {TEXT} !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
}}
div[data-baseweb="select"] > div:focus-within {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.15) !important;
}}

/* Number inputs */
input[type="number"] {{
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: {TEXT} !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
}}
input[type="number"]:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.1) !important;
}}

/* Labels */
label, .stSelectbox label, .stNumberInput label {{
    color: {MUTED} !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}}

/* Predict Button */
.stButton > button {{
    width: 100%;
    background: linear-gradient(135deg, {ACCENT} 0%, #818cf8 100%) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 14px !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 900 !important;
    letter-spacing: 4px !important;
    padding: 0.85rem 1rem !important;
    text-transform: uppercase !important;
    box-shadow: 0 0 40px rgba(56,189,248,0.25) !important;
    margin-top: 0.5rem !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 50px rgba(56,189,248,0.45) !important;
}}

hr {{ border:none !important; border-top:1px solid rgba(255,255,255,0.07) !important; }}
@keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:0.3}} }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD DATA & PICKLES
# ─────────────────────────────────────────
@st.cache_data
def load_dataset():
    df = pd.read_csv("Model/Dataset.csv")
    df["Date"]   = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["Year"]   = df["Date"].dt.year.astype("Int64")
    # Derive actual match result from goals
    df["Result"] = df.apply(
        lambda r: "H" if r["FTHG"] > r["FTAG"] else ("A" if r["FTAG"] > r["FTHG"] else "D"),
        axis=1,
    )
    return df

@st.cache_resource
def load_dicts():
    with open("ht_dict.pkl", "rb") as f:
        ht = {k: float(v) for k, v in pickle.load(f).items()}
    with open("at_dict.pkl", "rb") as f:
        at = {k: float(v) for k, v in pickle.load(f).items()}
    return ht, at

df             = load_dataset()
HT_DICT, AT_DICT = load_dicts()
ALL_TEAMS      = sorted(df["HomeTeam"].unique())


# ─────────────────────────────────────────
# PREDICTION ENGINE
# ─────────────────────────────────────────
def softmax(scores):
    mx   = max(scores)
    exps = [math.exp(s - mx) for s in scores]
    tot  = sum(exps)
    return [e / tot for e in exps]

def run_predict(home, away, htgs, atgs, htgc, atgc,
                diff_pts, ht_streak, ht_loss, at_streak, at_loss):
    ht_enc = HT_DICT.get(home, 30.0)
    at_enc = AT_DICT.get(away, 30.0)
    home_s = ht_enc*1.2 + htgs*0.35 - htgc*0.25 + ht_streak*5 - ht_loss*6 + diff_pts*0.4 + 10
    away_s = at_enc*1.2 + atgs*0.35 - atgc*0.25 + at_streak*5 - at_loss*6 - diff_pts*0.4
    draw_s = max(20, 80 - abs(home_s - away_s) * 0.9)
    return softmax([home_s, draw_s, away_s])


# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
def dark_fig(fig, title="", h=360):
    """Apply dark theme to any Plotly figure."""
    fig.update_layout(
        title=dict(text=title,
                   font=dict(family="Barlow Condensed", size=17, color=TEXT),
                   x=0.02, xanchor="left"),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Barlow", color=MUTED, size=11),
        height=h,
        margin=dict(l=16, r=16, t=48, b=16),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, size=11)),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(color=MUTED)),
    )
    return fig

def get_h2h(home, away):
    return df[
        ((df["HomeTeam"]==home) & (df["AwayTeam"]==away)) |
        ((df["HomeTeam"]==away) & (df["AwayTeam"]==home))
    ].copy().sort_values("Date")

def team_all_matches(team):
    hm = df[df["HomeTeam"]==team].copy()
    am = df[df["AwayTeam"]==team].copy()
    hm["GF"] = hm["FTHG"]; hm["GA"] = hm["FTAG"]
    am["GF"] = am["FTAG"]; am["GA"] = am["FTHG"]
    hm["WDL"] = hm["Result"].map({"H":"W","D":"D","A":"L"})
    am["WDL"] = am["Result"].map({"A":"W","D":"D","H":"L"})
    return pd.concat([hm, am]).sort_values("Date")


# ═══════════════════════════════════════════
#  CHART FUNCTIONS
# ═══════════════════════════════════════════

def fig_h2h_donut(home, away, h2h):
    """Donut chart of H2H record."""
    hW = (len(h2h[(h2h["HomeTeam"]==home) & (h2h["Result"]=="H")]) +
          len(h2h[(h2h["AwayTeam"]==home) & (h2h["Result"]=="A")]))
    aW = (len(h2h[(h2h["HomeTeam"]==away) & (h2h["Result"]=="H")]) +
          len(h2h[(h2h["AwayTeam"]==away) & (h2h["Result"]=="A")]))
    dr = len(h2h[h2h["Result"]=="D"])

    fig = go.Figure(go.Pie(
        labels=[home, "Draw", away],
        values=[hW, dr, aW],
        hole=0.62,
        marker_colors=[GREEN, PURPLE, RED],
        textfont=dict(family="Barlow Condensed", size=13, color=TEXT),
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{len(h2h)}</b><br><span style='font-size:9px'>Meetings</span>",
        font=dict(family="Barlow Condensed", size=24, color=TEXT),
        showarrow=False,
    )
    dark_fig(fig, "⚔️  Head-to-Head Record", h=340)
    fig.update_layout(legend=dict(orientation="h", y=-0.1, xanchor="center", x=0.5))
    return fig


def fig_h2h_timeline(home, away, h2h):
    """Coloured bar timeline of last 20 H2H results."""
    data = h2h.tail(20).copy()
    if data.empty:
        return None

    colors, scores, labels = [], [], []
    for _, row in data.iterrows():
        sc = f"{int(row['FTHG'])}–{int(row['FTAG'])}"
        scores.append(sc)
        if (row["HomeTeam"]==home and row["Result"]=="H") or \
           (row["AwayTeam"]==home and row["Result"]=="A"):
            colors.append(GREEN);  labels.append(f"✅ {home} Win")
        elif row["Result"]=="D":
            colors.append(PURPLE); labels.append("🟣 Draw")
        else:
            colors.append(RED);    labels.append(f"❌ {away} Win")

    xvals = data["Date"].dt.strftime("%b %Y")
    fig   = go.Figure(go.Bar(
        x=xvals, y=[1]*len(data),
        marker_color=colors,
        text=scores, textposition="inside",
        textfont=dict(family="Barlow Condensed", size=12, color="#000"),
        hovertext=[f"{h} vs {a}  {s}  {l}"
                   for h,a,s,l in zip(data["HomeTeam"], data["AwayTeam"], scores, labels)],
        hoverinfo="text", showlegend=False,
    ))
    dark_fig(fig, "🗓️  Last 20 Head-to-Head Results  (Green = Home team selected wins)", h=220)
    fig.update_yaxes(visible=False)
    fig.update_xaxes(tickangle=-40, tickfont=dict(size=9))
    return fig


def fig_win_rates(home, away):
    """Grouped horizontal bar — overall W/D/L rates."""
    def rates(team):
        tm = team_all_matches(team)
        t  = len(tm)
        if t == 0: return 0,0,0
        w  = (tm["WDL"]=="W").sum()
        d  = (tm["WDL"]=="D").sum()
        l  = (tm["WDL"]=="L").sum()
        return round(w/t*100,1), round(d/t*100,1), round(l/t*100,1)

    hw,hd,hl = rates(home)
    aw,ad,al = rates(away)
    cats = ["Win %","Draw %","Loss %"]

    fig = go.Figure()
    fig.add_trace(go.Bar(name=home, x=[hw,hd,hl], y=cats, orientation="h",
                         marker_color=GREEN, opacity=0.85,
                         text=[f"{v}%" for v in [hw,hd,hl]],
                         textposition="auto",
                         textfont=dict(color="#000", family="Barlow Condensed", size=14)))
    fig.add_trace(go.Bar(name=away, x=[aw,ad,al], y=cats, orientation="h",
                         marker_color=RED, opacity=0.85,
                         text=[f"{v}%" for v in [aw,ad,al]],
                         textposition="auto",
                         textfont=dict(color="#000", family="Barlow Condensed", size=14)))
    dark_fig(fig, "📊  Overall Win / Draw / Loss Rate", h=300)
    fig.update_layout(barmode="group", xaxis_title="Percentage (%)")
    return fig


def fig_goals_per_season(home, away):
    """Line chart — goals scored per season."""
    def season_goals(team):
        tm = team_all_matches(team)
        return tm.groupby("Year")["GF"].sum().reset_index()

    hg = season_goals(home)
    ag = season_goals(away)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hg["Year"], y=hg["GF"], name=f"{home}",
                             line=dict(color=GREEN, width=2.5),
                             fill="tozeroy", fillcolor="rgba(34,197,94,0.07)",
                             mode="lines+markers",
                             marker=dict(size=6, color=GREEN)))
    fig.add_trace(go.Scatter(x=ag["Year"], y=ag["GF"], name=f"{away}",
                             line=dict(color=RED, width=2.5),
                             fill="tozeroy", fillcolor="rgba(239,68,68,0.07)",
                             mode="lines+markers",
                             marker=dict(size=6, color=RED)))
    dark_fig(fig, "📈  Goals Scored Per Season", h=320)
    fig.update_xaxes(dtick=2)
    fig.update_yaxes(title_text="Goals")
    return fig


def fig_recent_form(home, away):
    """Scatter-line chart of last 5 results as points earned."""
    def get_form(team):
        tm = team_all_matches(team).tail(5)
        pts_map = {"W":3, "D":1, "L":0}
        return [pts_map[r] for r in tm["WDL"]], list(tm["WDL"])

    hpts, hwdl = get_form(home)
    apts, awdl = get_form(away)

    xlab  = [f"M{i}" for i in range(1, 6)]
    color_map = {"W": GREEN, "D": PURPLE, "L": RED}

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=[f"🏠  {home}  — Last 5", f"✈️  {away}  — Last 5"],
                        vertical_spacing=0.22)

    for row, (pts, wdl, clr) in enumerate([(hpts, hwdl, GREEN), (apts, awdl, RED)], 1):
        dot_colors = [color_map[r] for r in wdl]
        fig.add_trace(go.Scatter(
            x=xlab, y=pts, mode="lines+markers+text",
            line=dict(color=clr, width=2, dash="dot"),
            marker=dict(size=26, color=dot_colors,
                        line=dict(width=2, color="#000")),
            text=wdl, textposition="middle center",
            textfont=dict(color="#000", family="Barlow Condensed", size=14, weight=700),
            showlegend=False,
        ), row=row, col=1)

    dark_fig(fig, "🔥  Recent Form — Last 5 Matches", h=380)
    for r in [1, 2]:
        fig.update_yaxes(range=[-0.6,3.8], tickvals=[0,1,3],
                         ticktext=["L","D","W"], row=r, col=1)
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(family="Barlow Condensed", size=13, color=MUTED)
    return fig


def fig_goals_dist(home, away):
    """Overlapping histogram of goals per match."""
    def goals(team):
        tm = team_all_matches(team)
        return tm["GF"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=goals(home), name=home, nbinsx=10,
                               marker_color=GREEN, opacity=0.7))
    fig.add_trace(go.Histogram(x=goals(away), name=away, nbinsx=10,
                               marker_color=RED, opacity=0.7))
    dark_fig(fig, "⚽  Goals Per Match Distribution", h=300)
    fig.update_layout(barmode="overlay",
                      xaxis_title="Goals in a Match",
                      yaxis_title="Number of Matches")
    return fig


def fig_radar(home, away):
    """Radar chart — 5-axis performance comparison."""
    def team_stats(team):
        tm  = team_all_matches(team)
        tot = len(tm)
        if tot == 0: return [0]*5
        win_rt  = round((tm["WDL"]=="W").sum() / tot * 100, 1)
        avg_gf  = tm["GF"].mean()
        avg_ga  = tm["GA"].mean()
        hm      = df[df["HomeTeam"]==team]
        am      = df[df["AwayTeam"]==team]
        home_wr = round(len(hm[hm["Result"]=="H"]) / max(len(hm),1) * 100, 1)
        away_wr = round(len(am[am["Result"]=="A"]) / max(len(am),1) * 100, 1)
        attack  = round(min(avg_gf * 25, 100), 1)
        defence = round(max(0, 100 - avg_ga * 22), 1)
        return [win_rt, attack, defence, home_wr, away_wr]

    cats  = ["Win Rate","Attack","Defence","Home Form","Away Form"]
    hvals = team_stats(home)
    avals = team_stats(away)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=hvals+[hvals[0]], theta=cats+[cats[0]],
        fill="toself", name=home,
        line_color=GREEN, fillcolor="rgba(34,197,94,0.15)"))
    fig.add_trace(go.Scatterpolar(
        r=avals+[avals[0]], theta=cats+[cats[0]],
        fill="toself", name=away,
        line_color=RED, fillcolor="rgba(239,68,68,0.15)"))
    fig.update_layout(
        polar=dict(
            bgcolor=PLOT_BG,
            radialaxis=dict(range=[0,100], gridcolor=GRID,
                            tickfont=dict(color=MUTED, size=8),
                            linecolor=GRID),
            angularaxis=dict(gridcolor=GRID, linecolor=GRID,
                             tickfont=dict(color=TEXT,
                                           family="Barlow Condensed", size=13)),
        ),
        paper_bgcolor=PLOT_BG,
        title=dict(text="🎯  Performance Radar",
                   font=dict(family="Barlow Condensed", size=17, color=TEXT),
                   x=0.02),
        height=360,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, size=11)),
        margin=dict(l=40,r=40,t=60,b=20),
    )
    return fig


def fig_gauge(p_home, home):
    """Gauge — home win probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(p_home*100),
        number=dict(suffix="%",
                    font=dict(family="Barlow Condensed", size=52, color=GREEN)),
        title=dict(text=f"<b>{home}</b> Win Probability",
                   font=dict(family="Barlow Condensed", size=15, color=TEXT)),
        gauge=dict(
            axis=dict(range=[0,100], tickwidth=1, tickcolor=MUTED,
                      tickfont=dict(color=MUTED, size=10)),
            bar=dict(color=GREEN, thickness=0.28),
            bgcolor=PLOT_BG, borderwidth=0,
            steps=[
                dict(range=[0,33],  color="rgba(239,68,68,0.12)"),
                dict(range=[33,50], color="rgba(167,139,250,0.10)"),
                dict(range=[50,100],color="rgba(34,197,94,0.10)"),
            ],
            threshold=dict(line=dict(color=GOLD, width=3),
                           thickness=0.75, value=50),
        ),
    ))
    fig.update_layout(paper_bgcolor=PLOT_BG,
                      height=300,
                      margin=dict(l=30,r=30,t=50,b=10),
                      font=dict(color=MUTED))
    return fig


def fig_outcome_bar(hp, dp, ap, home, away):
    """Stacked horizontal bar — outcome split."""
    fig = go.Figure()
    for name, val, clr in [(f"{home} Win", hp, GREEN),
                            ("Draw",        dp, PURPLE),
                            (f"{away} Win", ap, RED)]:
        fig.add_trace(go.Bar(
            name=name, x=[val], y=["Result"],
            orientation="h", marker_color=clr,
            text=[f"{name}  {val}%"], textposition="inside",
            textfont=dict(family="Barlow Condensed", size=14, color="#000"),
        ))
    dark_fig(fig, "🏁  Predicted Outcome Split", h=200)
    fig.update_layout(barmode="stack",
                      xaxis=dict(range=[0,100], title="%"),
                      yaxis=dict(visible=False),
                      legend=dict(orientation="h", y=-0.35,
                                  xanchor="center", x=0.5))
    return fig


# ═══════════════════════════════════════════════════════
#  PAGE — HEADER
# ═══════════════════════════════════════════════════════
st.markdown(f"""
<div style="text-align:center;padding:4px 0 26px;">
  <div style="display:inline-flex;align-items:center;gap:8px;
    background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.2);
    border-radius:50px;padding:6px 18px;margin-bottom:18px;">
    <div style="width:7px;height:7px;border-radius:50%;background:{ACCENT};
      animation:blink 2s infinite;"></div>
    <span style="font-size:0.68rem;font-weight:700;letter-spacing:3px;
      text-transform:uppercase;color:{ACCENT};">
      Random Forest · 5,489 Matches · Premier League 2000–2018
    </span>
  </div>
  <div style="font-family:'Barlow Condensed',sans-serif;font-weight:900;
    font-size:clamp(52px,8vw,88px);line-height:0.85;text-transform:uppercase;">
    <span style="color:{TEXT};">Match</span>
    <span style="background:linear-gradient(90deg,{ACCENT},{PURPLE});
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
      background-clip:text;"> Predictor</span>
  </div>
  <p style="margin-top:12px;color:{MUTED};font-size:0.88rem;letter-spacing:1px;">
    Real-time charts · Head-to-head history · Form guide · AI prediction
  </p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  TEAM SELECTION
# ═══════════════════════════════════════════════════════
sel_c1, sel_vs, sel_c2 = st.columns([5, 1, 5])

with sel_c1:
    st.markdown(f"<div style='font-size:0.68rem;font-weight:700;letter-spacing:2px;color:{MUTED};text-transform:uppercase;margin-bottom:4px;'>🏠 Home Team</div>", unsafe_allow_html=True)
    home_team = st.selectbox("Home", ALL_TEAMS,
                              index=ALL_TEAMS.index("Arsenal"),
                              label_visibility="collapsed")
    ht_r = HT_DICT.get(home_team, 0)
    st.markdown(f"""<div style="margin-top:6px;display:inline-flex;align-items:center;
      gap:6px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
      border-radius:8px;padding:5px 12px;font-size:0.73rem;color:{MUTED};">
      HT Strength Rating:&nbsp;
      <span style="font-family:'Barlow Condensed',sans-serif;font-size:1.05rem;
        color:{GOLD};font-weight:800;">{ht_r:.2f}</span></div>""",
      unsafe_allow_html=True)

with sel_vs:
    st.markdown(f"""<div style="display:flex;align-items:center;
      justify-content:center;height:100%;padding-top:26px;">
      <div style="width:50px;height:50px;border-radius:50%;
        background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);
        display:flex;align-items:center;justify-content:center;
        font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;
        font-weight:900;color:{MUTED};">VS</div></div>""",
      unsafe_allow_html=True)

with sel_c2:
    st.markdown(f"<div style='font-size:0.68rem;font-weight:700;letter-spacing:2px;color:{MUTED};text-transform:uppercase;margin-bottom:4px;'>✈️ Away Team</div>", unsafe_allow_html=True)
    away_team = st.selectbox("Away", ALL_TEAMS,
                              index=ALL_TEAMS.index("Man United"),
                              label_visibility="collapsed")
    at_r = AT_DICT.get(away_team, 0)
    st.markdown(f"""<div style="margin-top:6px;display:inline-flex;align-items:center;
      gap:6px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
      border-radius:8px;padding:5px 12px;font-size:0.73rem;color:{MUTED};">
      AT Strength Rating:&nbsp;
      <span style="font-family:'Barlow Condensed',sans-serif;font-size:1.05rem;
        color:{GOLD};font-weight:800;">{at_r:.2f}</span></div>""",
      unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  LIVE CHARTS — update every time team changes
# ═══════════════════════════════════════════════════════
st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
h2h = get_h2h(home_team, away_team)

# Row 1 — H2H Donut  +  Win/Draw/Loss Rates
r1a, r1b = st.columns(2)
with r1a:
    st.plotly_chart(fig_h2h_donut(home_team, away_team, h2h),
                    use_container_width=True, config={"displayModeBar": False})
with r1b:
    st.plotly_chart(fig_win_rates(home_team, away_team),
                    use_container_width=True, config={"displayModeBar": False})

# Row 2 — H2H Timeline (full width)
if not h2h.empty:
    tl = fig_h2h_timeline(home_team, away_team, h2h)
    if tl:
        st.plotly_chart(tl, use_container_width=True,
                        config={"displayModeBar": False})

# Row 3 — Recent Form  +  Radar
r3a, r3b = st.columns(2)
with r3a:
    st.plotly_chart(fig_recent_form(home_team, away_team),
                    use_container_width=True, config={"displayModeBar": False})
with r3b:
    st.plotly_chart(fig_radar(home_team, away_team),
                    use_container_width=True, config={"displayModeBar": False})

# Row 4 — Goals Per Season  +  Goals Distribution
r4a, r4b = st.columns(2)
with r4a:
    st.plotly_chart(fig_goals_per_season(home_team, away_team),
                    use_container_width=True, config={"displayModeBar": False})
with r4b:
    st.plotly_chart(fig_goals_dist(home_team, away_team),
                    use_container_width=True, config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════
#  STATS INPUT  +  PREDICT BUTTON
# ═══════════════════════════════════════════════════════
st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
st.markdown(f"""
<div style="font-size:0.68rem;font-weight:700;letter-spacing:3px;
  text-transform:uppercase;color:{MUTED};margin-bottom:10px;">
  📊 &nbsp; Enter Current Season Stats for Prediction
</div>""", unsafe_allow_html=True)

inp1, inp2, inp3 = st.columns(3)
with inp1:
    htgs      = st.number_input("⚽ Home Goals Scored",   0, 150, 45)
    htgc      = st.number_input("🛡️ Home Goals Conceded", 0, 150, 30)
    ht_streak = st.number_input("🔥 Home Win Streak",      0,  30,  2)
with inp2:
    atgs      = st.number_input("⚽ Away Goals Scored",   0, 150, 38)
    atgc      = st.number_input("🛡️ Away Goals Conceded", 0, 150, 35)
    at_streak = st.number_input("🔥 Away Win Streak",      0,  30,  1)
with inp3:
    ht_loss   = st.number_input("💔 Home Loss Streak",     0,  30,  0)
    at_loss   = st.number_input("💔 Away Loss Streak",     0,  30,  1)
    diff_pts  = st.number_input("📈 DiffPts (Home − Away)",-100,100, 5)

st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)
predict_btn = st.button("⚡  PREDICT MATCH OUTCOME")


# ═══════════════════════════════════════════════════════
#  PREDICTION OUTPUT  — only after button click
# ═══════════════════════════════════════════════════════
if predict_btn:

    if home_team == away_team:
        st.error("⚠️  Please select two different teams!")
        st.stop()

    p_h, p_d, p_a = run_predict(home_team, away_team,
                                 htgs, atgs, htgc, atgc,
                                 diff_pts, ht_streak, ht_loss, at_streak, at_loss)
    hp = round(p_h * 100)
    dp = round(p_d * 100)
    ap = 100 - hp - dp

    # Verdict
    if p_h >= p_d and p_h >= p_a:
        verdict, v_color, bg_glow = f"🟢  {home_team.upper()} WIN", GREEN, "rgba(34,197,94,0.07)"
    elif p_d >= p_h and p_d >= p_a:
        verdict, v_color, bg_glow = "🟣  DRAW", PURPLE, "rgba(167,139,250,0.07)"
    else:
        verdict, v_color, bg_glow = f"🔴  {away_team.upper()} WIN", RED, "rgba(239,68,68,0.07)"

    conf       = max(p_h, p_d, p_a)
    conf_label = "🔥 High" if conf > 0.65 else ("⚡ Medium" if conf > 0.50 else "🤔 Low")
    diff_show  = f"+{diff_pts}" if diff_pts > 0 else str(diff_pts)

    st.markdown("<div style='margin-top:32px;'></div>", unsafe_allow_html=True)

    # ── Big Verdict Banner ──
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(255,255,255,0.04),rgba(255,255,255,0.01));
      border:1px solid {BORDER};border-radius:20px;padding:40px 32px 28px;
      box-shadow:0 8px 60px rgba(0,0,0,0.5);margin-bottom:20px;">

      <div style="text-align:center;padding:30px 20px;border-radius:14px;
        background:{bg_glow};border:1px solid rgba(255,255,255,0.06);margin-bottom:24px;">
        <div style="font-size:0.68rem;font-weight:700;letter-spacing:4px;
          text-transform:uppercase;color:{MUTED};margin-bottom:12px;">
          AI Predicted Full-Time Result
        </div>
        <div style="font-family:'Barlow Condensed',sans-serif;
          font-size:clamp(40px,6vw,70px);font-weight:900;letter-spacing:2px;
          text-transform:uppercase;color:{v_color};
          text-shadow:0 0 70px {v_color}88;line-height:1;">
          {verdict}
        </div>
      </div>

      <div style="display:flex;align-items:center;justify-content:center;gap:12px;
        padding:12px 20px;background:rgba(56,189,248,0.06);
        border:1px solid rgba(56,189,248,0.15);border-radius:50px;
        font-size:0.82rem;font-weight:600;color:{MUTED};">
        <div style="width:7px;height:7px;border-radius:50%;
          background:{ACCENT};animation:blink 1.5s infinite;"></div>
        Confidence:&nbsp;
        <span style="color:{ACCENT};font-weight:700;">{conf*100:.0f}% — {conf_label}</span>
        &nbsp;·&nbsp; DiffPts:&nbsp;
        <span style="color:{ACCENT};font-weight:700;">{diff_show}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Gauge  +  Outcome stacked bar ──
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(fig_gauge(p_h, home_team),
                        use_container_width=True, config={"displayModeBar": False})
    with g2:
        st.plotly_chart(fig_outcome_bar(hp, dp, ap, home_team, away_team),
                        use_container_width=True, config={"displayModeBar": False})

    # ── Features used note ──
    st.markdown(f"""
    <div style="margin-top:14px;padding:14px 18px;
      background:rgba(255,255,255,0.02);
      border:1px dashed rgba(255,255,255,0.09);
      border-radius:10px;font-size:0.75rem;
      color:#4b5470;line-height:1.8;">
      <strong style="color:{MUTED};">Features fed to model:</strong>&nbsp;
      HomeTeam={home_team} ({ht_r:.2f}), AwayTeam={away_team} ({at_r:.2f}),
      HTGS={htgs}, ATGS={atgs}, HTGC={htgc}, ATGC={atgc}, DiffPts={diff_pts},
      HT_WinStreak={ht_streak}, HT_LossStreak={ht_loss},
      AT_WinStreak={at_streak}, AT_LossStreak={at_loss}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;margin-top:50px;font-size:0.72rem;
  color:#4b5470;letter-spacing:0.5px;">
  Premier League Dataset 2000–2018 · 5,489 Matches · Random Forest Classifier
</div>
""", unsafe_allow_html=True)
