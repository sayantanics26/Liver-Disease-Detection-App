"""
Livo Health — Liver Disease Prediction Streamlit App
Run: streamlit run app.py
Requirements: streamlit requests Pillow
"""

import streamlit as st
import requests
import json
from PIL import Image
import io
import base64

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Livo Health",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── API Base URL ───────────────────────────────────────────────
API_BASE = "http://localhost:5000/api"   # ← change to your deployed URL

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
    background: #f8faf8;
    color: #111827;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 0 !important;
    max-width: 1140px;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* ── Top nav bar ── */
.livo-nav {
    background: #fff;
    border-bottom: 1px solid #e5e7eb;
    padding: 14px 40px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin: -1rem -2rem 0 -2rem;
    position: sticky;
    top: 0;
    z-index: 999;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.livo-logo-box {
    width: 36px; height: 36px;
    background: #16a34a;
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.livo-logo-text {
    font-size: 20px; font-weight: 800;
    letter-spacing: -0.5px; color: #111827;
}
.livo-logo-text span { color: #16a34a; }

/* ── Hero section ── */
.livo-hero {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 45%, #f0fdf4 100%);
    padding: 56px 40px 48px;
    text-align: center;
    margin: 0 -2rem 0 -2rem;
    border-bottom: 1px solid #d1fae5;
}
.livo-hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #fff;
    border: 1px solid #bbf7d0;
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 13px;
    color: #15803d;
    font-weight: 500;
    margin-bottom: 18px;
}
.livo-dot { width: 7px; height: 7px; border-radius: 50%; background: #16a34a; display: inline-block; }
.livo-title {
    font-size: 46px; font-weight: 800;
    letter-spacing: -1.5px; color: #111827;
    line-height: 1.1; margin-bottom: 14px;
}
.livo-title .green { color: #16a34a; }
.livo-subtitle {
    font-size: 18px; color: #6b7280;
    line-height: 1.65; margin-bottom: 8px;
    font-style: italic; font-weight: 500;
}
.livo-desc {
    font-size: 15px; color: #9ca3af;
    max-width: 540px; margin: 0 auto;
    line-height: 1.65;
}

/* ── Section heading ── */
.section-head {
    margin: 36px 0 24px;
}
.section-module {
    font-size: 11px; font-weight: 600;
    letter-spacing: .09em; text-transform: uppercase;
    color: #16a34a; margin-bottom: 6px;
}
.section-title {
    font-size: 24px; font-weight: 800;
    letter-spacing: -0.6px; color: #111827;
    margin-bottom: 6px;
}
.section-desc { font-size: 14px; color: #9ca3af; }

/* ── White cards ── */
.livo-card {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    margin-bottom: 16px;
}
.livo-card-header {
    display: flex; align-items: center; gap: 12px;
    padding-bottom: 16px;
    border-bottom: 1px solid #f3f4f6;
    margin-bottom: 20px;
}
.livo-card-icon {
    width: 40px; height: 40px;
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 10px; display: flex;
    align-items: center; justify-content: center;
    font-size: 18px;
}
.card-title { font-size: 15px; font-weight: 700; color: #111827; }
.card-sub   { font-size: 12px; color: #9ca3af; margin-top: 2px; }

/* ── Field labels ── */
.field-label {
    font-size: 13px; font-weight: 600;
    color: #374151; margin-bottom: 4px;
}
.field-ref {
    font-size: 11px; color: #9ca3af; margin-top: 2px;
}

/* ── Section divider ── */
.sec-div {
    font-size: 11px; font-weight: 600;
    letter-spacing: .08em; text-transform: uppercase;
    color: #9ca3af; padding: 4px 0 10px;
    border-bottom: 1px solid #f3f4f6;
    margin-bottom: 16px;
}

/* ── Result cards ── */
.result-wrap {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.result-bar-green  { height: 4px; background: linear-gradient(90deg, #22c55e, #16a34a); }
.result-bar-yellow { height: 4px; background: linear-gradient(90deg, #fbbf24, #d97706); }
.result-bar-red    { height: 4px; background: linear-gradient(90deg, #f87171, #dc2626); }
.result-inner { padding: 22px 24px; }

.risk-pct-green    { font-size: 52px; font-weight: 800; color: #16a34a; letter-spacing: -2px; line-height: 1; }
.risk-pct-yellow   { font-size: 52px; font-weight: 800; color: #d97706; letter-spacing: -2px; line-height: 1; }
.risk-pct-red      { font-size: 52px; font-weight: 800; color: #dc2626; letter-spacing: -2px; line-height: 1; }
.risk-unit { font-size: 22px; color: #9ca3af; }
.risk-label { font-size: 11px; color: #9ca3af; margin-top: 4px; }

.badge-low  { display:inline-block; padding:4px 14px; border-radius:20px; font-size:12px; font-weight:600; background:#f0fdf4; border:1px solid #bbf7d0; color:#15803d; }
.badge-med  { display:inline-block; padding:4px 14px; border-radius:20px; font-size:12px; font-weight:600; background:#fefce8; border:1px solid #fde68a; color:#b45309; }
.badge-high { display:inline-block; padding:4px 14px; border-radius:20px; font-size:12px; font-weight:600; background:#fef2f2; border:1px solid #fca5a5; color:#dc2626; }

.gauge-wrap { margin: 14px 0 6px; }
.gauge-bg   { height: 8px; border-radius: 4px; background: #f3f4f6; overflow: hidden; }

.marker-row {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 12px; background: #f9fafb;
    border-radius: 10px; border: 1px solid #f3f4f6;
    margin-bottom: 6px; font-size: 13px;
}
.mk-normal  { color: #16a34a; }
.mk-high    { color: #dc2626; }
.mk-low     { color: #d97706; }
.ms-normal  { padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; background:#f0fdf4; color:#15803d; }
.ms-high    { padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; background:#fef2f2; color:#dc2626; }
.ms-low     { padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; background:#fefce8; color:#b45309; }

.note-box {
    background: #f9fafb; border-radius: 12px;
    padding: 14px 16px; margin-top: 14px;
    font-size: 13px; color: #6b7280; line-height: 1.7;
    border-left: 3px solid #16a34a;
}
.disclaimer-box {
    background: #fafafa; border-top: 1px solid #f3f4f6;
    padding: 10px 24px; font-size: 11px; color: #d1d5db;
}

/* ── Upload zone ── */
.upload-hint {
    background: #f0fdf4; border: 2px dashed #86efac;
    border-radius: 16px; padding: 36px 24px;
    text-align: center; color: #15803d;
    font-size: 14px; margin-bottom: 16px;
}
.upload-hint .uh-icon { font-size: 40px; margin-bottom: 12px; }
.upload-hint .uh-title { font-weight: 700; font-size: 16px; margin-bottom: 6px; color: #111827; }
.upload-hint .uh-sub   { color: #9ca3af; font-size: 13px; }

/* ── US result ── */
.pred-normal   { font-size: 30px; font-weight: 800; color: #16a34a; letter-spacing: -1px; }
.pred-benign   { font-size: 30px; font-weight: 800; color: #d97706; letter-spacing: -1px; }
.pred-malignant{ font-size: 30px; font-weight: 800; color: #dc2626; letter-spacing: -1px; }

.prob-row { display:flex; align-items:center; gap:10px; margin-bottom:10px; font-size:13px; }
.prob-label { width: 74px; color: #6b7280; }
.prob-bg  { flex:1; height:8px; background:#f3f4f6; border-radius:4px; overflow:hidden; }
.prob-val { width:44px; text-align:right; font-weight:600; color:#374151; font-size:12px; }

.action-green  { background:#f0fdf4; border-radius:12px; padding:14px 16px; border-left:3px solid #16a34a; font-size:13px; color:#374151; line-height:1.7; margin-top:8px; }
.action-yellow { background:#fefce8; border-radius:12px; padding:14px 16px; border-left:3px solid #d97706; font-size:13px; color:#374151; line-height:1.7; margin-top:8px; }
.action-red    { background:#fef2f2; border-radius:12px; padding:14px 16px; border-left:3px solid #dc2626; font-size:13px; color:#374151; line-height:1.7; margin-top:8px; }

/* ── Idle state ── */
.idle-box {
    padding: 56px 24px; text-align: center;
    background: #fff; border: 1px solid #e5e7eb;
    border-radius: 16px;
}
.idle-icon { font-size: 52px; margin-bottom: 14px; opacity: 0.25; }
.idle-text { font-size: 14px; color: #9ca3af; line-height: 1.6; }

/* ── Quick fill buttons ── */
.stButton > button {
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    transition: all .15s !important;
}

/* ── Streamlit tab overrides ── */
.stTabs [data-baseweb="tab-list"] {
    background: #f9fafb;
    border-radius: 12px;
    padding: 4px;
    border: 1px solid #e5e7eb;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 22px !important;
    color: #6b7280 !important;
}
.stTabs [aria-selected="true"] {
    background: #fff !important;
    color: #16a34a !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
}
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* ── Number input ── */
.stNumberInput > label { font-size: 13px !important; font-weight: 600 !important; color: #374151 !important; }
.stSelectbox > label   { font-size: 13px !important; font-weight: 600 !important; color: #374151 !important; }
div[data-baseweb="input"] > div { border-radius: 10px !important; border-color: #d1d5db !important; }
div[data-baseweb="input"] > div:focus-within { border-color: #16a34a !important; box-shadow: 0 0 0 3px rgba(22,163,74,.12) !important; }

/* ── File uploader ── */
.stFileUploader > label { font-size: 13px !important; font-weight: 600 !important; }
[data-testid="stFileUploaderDropzone"] {
    border-radius: 14px !important;
    border: 2px dashed #86efac !important;
    background: #f0fdf4 !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: #16a34a !important;
    color: #fff !important;
    border: none !important;
    padding: 12px 24px !important;
    font-size: 15px !important;
}
.stButton > button[kind="primary"]:hover {
    background: #15803d !important;
}

/* ── Footer ── */
.livo-footer {
    border-top: 1px solid #e5e7eb;
    padding: 20px 40px;
    margin: 40px -2rem 0 -2rem;
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
    font-size: 12px;
    color: #9ca3af;
    background: #fff;
}

/* Smooth scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #86efac; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def call_api(endpoint, payload=None, files=None):
    """Call backend API. Returns (data, error_message)."""
    try:
        if files:
            r = requests.post(f"{API_BASE}{endpoint}", files=files, timeout=30)
        else:
            r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=15)
        r.raise_for_status()
        return r.json(), None
    except requests.ConnectionError:
        return None, "Cannot connect to the prediction server. Please ensure the backend is running."
    except requests.Timeout:
        return None, "Request timed out. The server may be busy."
    except Exception as e:
        return None, str(e)


def risk_config(level: str):
    lvl = level.lower()
    if lvl == "high":
        return {"bar": "result-bar-red",    "pct_cls": "risk-pct-red",    "badge": "badge-high", "gauge_color": "#ef4444", "icon": "🔴"}
    if lvl == "medium":
        return {"bar": "result-bar-yellow", "pct_cls": "risk-pct-yellow", "badge": "badge-med",  "gauge_color": "#f59e0b", "icon": "🟡"}
    return     {"bar": "result-bar-green",  "pct_cls": "risk-pct-green",  "badge": "badge-low",  "gauge_color": "#22c55e", "icon": "🟢"}


def marker_html(markers):
    rows = ""
    for m in markers:
        s = m["status"]
        icon  = "✅" if s == "Normal" else "🔺" if s == "Elevated" else "🔻"
        mcls  = "mk-normal" if s == "Normal" else "mk-high" if s == "Elevated" else "mk-low"
        mscls = "ms-normal" if s == "Normal" else "ms-high" if s == "Elevated" else "ms-low"
        unit  = f" {m['unit']}" if m['unit'] else ""
        rows += f"""
        <div class="marker-row">
            <span class="{mcls}">{icon}</span>
            <span style="flex:1;color:#374151">{m['name']}</span>
            <span style="font-weight:600;font-variant-numeric:tabular-nums">{m['value']}{unit}</span>
            <span class="{mscls}">{s}</span>
        </div>"""
    return rows


def gauge_html(pct, color):
    return f"""
    <div class="gauge-wrap">
        <div class="gauge-bg">
            <div style="height:100%;width:{pct}%;background:{color};border-radius:4px;transition:width 1s ease"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:9px;color:#d1d5db;margin-top:4px">
            <span>0</span><span>25</span><span>50</span><span>75</span><span>100</span>
        </div>
    </div>"""


# ═══════════════════════════════════════════════════════════════
#  NAV BAR
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="livo-nav">
    <div class="livo-logo-box">🫀</div>
    <div class="livo-logo-text">Livo <span>Health</span></div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="livo-hero">
    <div class="livo-hero-badge">
        <span class="livo-dot"></span>
        AI-Powered Liver Health Analysis
    </div>
    <div class="livo-title">Livo <span class="green">Health</span></div>
    <div class="livo-subtitle">Your Liver Health is Important too</div>
    <div class="livo-desc">
        Get fast, intelligent insights into your liver health through blood test analysis
        and ultrasound image interpretation.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════
tab_blood, tab_us = st.tabs(["🩸  Blood Test Analysis", "🔬  Ultrasound Scan Analysis"])


# ═══════════════════════════════════════════════════════════════
#  TAB 1 — BLOOD TEST
# ═══════════════════════════════════════════════════════════════
with tab_blood:
    st.markdown("""
    <div class="section-head">
        <div class="section-module">Module 01</div>
        <div class="section-title">Blood Test Analysis</div>
        <div class="section-desc">Enter your liver function test values to assess disease risk.</div>
    </div>""", unsafe_allow_html=True)

    col_form, col_result = st.columns([1.15, 0.85], gap="large")

    # ── FORM ──────────────────────────────────────────────────
    with col_form:
        st.markdown('<div class="livo-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="livo-card-header">
            <div class="livo-card-icon">📋</div>
            <div>
                <div class="card-title">Patient Lab Report</div>
                <div class="card-sub">Fill in your latest liver function test results</div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Quick fill buttons
        qc1, qc2, qc3 = st.columns(3)
        if qc1.button("↗ Sample: Disease", use_container_width=True):
            st.session_state.update({"qs_age":55,"qs_gender":"Male","qs_tbili":8.5,"qs_dbili":3.2,"qs_alp":560,"qs_sgpt":145,"qs_sgot":120,"qs_tp":5.8,"qs_alb":2.4,"qs_agr":0.62})
        if qc2.button("↗ Sample: Normal", use_container_width=True):
            st.session_state.update({"qs_age":35,"qs_gender":"Female","qs_tbili":0.7,"qs_dbili":0.2,"qs_alp":120,"qs_sgpt":22,"qs_sgot":18,"qs_tp":7.1,"qs_alb":4.0,"qs_agr":1.35})
        if qc3.button("✕ Clear", use_container_width=True):
            for k in ["qs_age","qs_gender","qs_tbili","qs_dbili","qs_alp","qs_sgpt","qs_sgot","qs_tp","qs_alb","qs_agr"]:
                st.session_state.pop(k, None)

        st.markdown('<div class="sec-div" style="margin-top:16px">Demographics</div>', unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        with d1:
            age = st.number_input("Age", min_value=1, max_value=110, value=int(st.session_state.get("qs_age", 45)), step=1, help="Patient age in years (4–90)")
        with d2:
            gender = st.selectbox("Gender", ["Male", "Female"], index=0 if st.session_state.get("qs_gender","Male")=="Male" else 1)

        st.markdown('<div class="sec-div">Bilirubin Panel</div>', unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1:
            tbili = st.number_input("Total Bilirubin (mg/dL)", min_value=0.0, max_value=100.0, value=float(st.session_state.get("qs_tbili", 1.0)), step=0.1, help="Normal: 0.2–1.2 mg/dL")
        with b2:
            dbili = st.number_input("Direct Bilirubin (mg/dL)", min_value=0.0, max_value=50.0, value=float(st.session_state.get("qs_dbili", 0.3)), step=0.1, help="Normal: 0.0–0.3 mg/dL")

        st.markdown('<div class="sec-div">Liver Enzymes</div>', unsafe_allow_html=True)
        e1, e2, e3 = st.columns(3)
        with e1:
            alp  = st.number_input("Alkaline Phosphatase (IU/L)", min_value=0, max_value=5000, value=int(st.session_state.get("qs_alp", 187)), step=1, help="Normal: 44–147 IU/L")
        with e2:
            sgpt = st.number_input("SGPT / ALT (IU/L)", min_value=0, max_value=3000, value=int(st.session_state.get("qs_sgpt", 35)), step=1, help="Normal: 7–56 IU/L")
        with e3:
            sgot = st.number_input("SGOT / AST (IU/L)", min_value=0, max_value=5000, value=int(st.session_state.get("qs_sgot", 30)), step=1, help="Normal: 10–40 IU/L")

        st.markdown('<div class="sec-div">Protein Panel</div>', unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        with p1:
            tp  = st.number_input("Total Proteins (g/dL)", min_value=0.0, max_value=12.0, value=float(st.session_state.get("qs_tp", 6.8)), step=0.1, help="Normal: 6.3–8.2 g/dL")
        with p2:
            alb = st.number_input("Albumin — ALB (g/dL)", min_value=0.0, max_value=6.0, value=float(st.session_state.get("qs_alb", 3.3)), step=0.1, help="Normal: 3.4–5.4 g/dL")
        with p3:
            agr = st.number_input("A/G Ratio", min_value=0.0, max_value=5.0, value=float(st.session_state.get("qs_agr", 0.9)), step=0.01, help="Normal: 1.1–2.1")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        analyze_btn = st.button("🔍  Analyze Blood Test", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)  # close card

    # ── RESULT ────────────────────────────────────────────────
    with col_result:
        if "blood_result" not in st.session_state:
            st.markdown("""
            <div class="idle-box">
                <div class="idle-icon">🫀</div>
                <div class="idle-text">Enter your lab values and click<br/><strong style='color:#6b7280'>Analyze Blood Test</strong><br/>to see your liver health assessment.</div>
            </div>""", unsafe_allow_html=True)

        if analyze_btn:
            with st.spinner("Analyzing your blood test results…"):
                data, err = call_api("/predict/clinical", payload={
                    "age": age, "gender": gender,
                    "total_bilirubin": tbili, "direct_bilirubin": dbili,
                    "alkaline_phosphotase": alp, "sgpt": sgpt, "sgot": sgot,
                    "total_proteins": tp, "albumin": alb, "ag_ratio": agr,
                })
            if err:
                st.error(f"⚠️ {err}")
            else:
                st.session_state["blood_result"] = data

        res = st.session_state.get("blood_result")
        if res and not analyze_btn:
            rc  = risk_config(res["risk_level"])
            pct = res["probability"]

            st.markdown(f"""
            <div class="result-wrap">
                <div class="{rc['bar']}"></div>
                <div class="result-inner">
                    <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px">
                        <div>
                            <div class="{rc['pct_cls']}">{pct}<span class="risk-unit">%</span></div>
                            <div class="risk-label">probability of liver disease</div>
                        </div>
                        <span class="{rc['badge']}">{res['risk_level']} Risk</span>
                    </div>
                    <div style="font-size:13px;color:#6b7280;margin-top:8px;font-weight:500">{res['prediction']}</div>
                    {gauge_html(pct, rc['gauge_color'])}
                </div>

                <div style="padding:0 24px 16px;border-top:1px solid #f3f4f6">
                    <div style="font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#9ca3af;padding:14px 0 10px">Lab Marker Status</div>
                    {marker_html(res.get("markers", []))}
                </div>

                <div style="padding:0 24px 16px">
                    <div style="font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#9ca3af;margin-bottom:8px">◈ Clinical Interpretation</div>
                    <div class="note-box">
                        {"Lab values appear within normal reference ranges. Liver health risk is low. Maintain a healthy lifestyle and schedule routine check-ups."
                          if res["risk_level"] == "Low" else
                         "Some markers show mild elevation. This may indicate early-stage hepatic stress. Consider consulting a doctor for further evaluation."
                          if res["risk_level"] == "Medium" else
                         "Several markers are significantly elevated. This pattern warrants prompt medical attention. Please consult a hepatologist as soon as possible."}
                    </div>
                </div>

                <div class="disclaimer-box">⚠ For informational purposes only · Always consult a qualified physician</div>
            </div>
            """, unsafe_allow_html=True)

        if analyze_btn and res:
            rc  = risk_config(res["risk_level"])
            pct = res["probability"]

            st.markdown(f"""
            <div class="result-wrap">
                <div class="{rc['bar']}"></div>
                <div class="result-inner">
                    <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px">
                        <div>
                            <div class="{rc['pct_cls']}">{pct}<span class="risk-unit">%</span></div>
                            <div class="risk-label">probability of liver disease</div>
                        </div>
                        <span class="{rc['badge']}">{res['risk_level']} Risk</span>
                    </div>
                    <div style="font-size:13px;color:#6b7280;margin-top:8px;font-weight:500">{res['prediction']}</div>
                    {gauge_html(pct, rc['gauge_color'])}
                </div>

                <div style="padding:0 24px 16px;border-top:1px solid #f3f4f6">
                    <div style="font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#9ca3af;padding:14px 0 10px">Lab Marker Status</div>
                    {marker_html(res.get("markers", []))}
                </div>

                <div style="padding:0 24px 16px">
                    <div style="font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#9ca3af;margin-bottom:8px">◈ Clinical Interpretation</div>
                    <div class="note-box">
                        {"Lab values appear within normal reference ranges. Liver health risk is low. Maintain a healthy lifestyle and schedule routine check-ups."
                          if res["risk_level"] == "Low" else
                         "Some markers show mild elevation. This may indicate early-stage hepatic stress. Consider consulting a doctor for further evaluation."
                          if res["risk_level"] == "Medium" else
                         "Several markers are significantly elevated. This pattern warrants prompt medical attention. Please consult a hepatologist as soon as possible."}
                    </div>
                </div>

                <div class="disclaimer-box">⚠ For informational purposes only · Always consult a qualified physician</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  TAB 2 — ULTRASOUND
# ═══════════════════════════════════════════════════════════════
with tab_us:
    st.markdown("""
    <div class="section-head">
        <div class="section-module">Module 02</div>
        <div class="section-title">Ultrasound Scan Analysis</div>
        <div class="section-desc">Upload a liver ultrasound image to detect Normal, Benign, or Malignant conditions.</div>
    </div>""", unsafe_allow_html=True)

    us_left, us_right = st.columns([1.15, 0.85], gap="large")

    with us_left:
        st.markdown("""
        <div class="upload-hint">
            <div class="uh-icon">🔬</div>
            <div class="uh-title">Upload Liver Ultrasound Image</div>
            <div class="uh-sub">Supports JPG · PNG · BMP · TIFF · WebP</div>
        </div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Select your ultrasound image",
            type=["jpg","jpeg","png","bmp","tiff","tif","webp"],
            label_visibility="collapsed"
        )

        if uploaded:
            img = Image.open(uploaded)
            st.markdown('<div class="livo-card" style="padding:0;overflow:hidden">', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="padding:10px 16px;border-bottom:1px solid #f3f4f6;display:flex;justify-content:space-between;align-items:center">
                <span style="font-size:12px;color:#6b7280">{uploaded.name} · {uploaded.size//1024} KB</span>
                <span style="font-size:11px;color:#9ca3af">{img.size[0]}×{img.size[1]}px</span>
            </div>""", unsafe_allow_html=True)
            st.image(img, use_container_width=True, clamp=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            scan_btn = st.button("🔍  Analyze Ultrasound Scan", type="primary", use_container_width=True)
        else:
            scan_btn = False

    with us_right:
        if not uploaded:
            st.markdown("""
            <div class="idle-box">
                <div class="idle-icon">🔬</div>
                <div class="idle-text">Upload a liver ultrasound image and click<br/><strong style='color:#6b7280'>Analyze Ultrasound Scan</strong><br/>to detect liver condition.</div>
            </div>""", unsafe_allow_html=True)

        if scan_btn and uploaded:
            with st.spinner("Analyzing ultrasound image…"):
                uploaded.seek(0)
                data, err = call_api(
                    "/predict/ultrasound",
                    files={"file": (uploaded.name, uploaded.read(), uploaded.type)}
                )
            if err:
                st.error(f"⚠️ {err}")
            else:
                st.session_state["us_result"] = data

        us_res = st.session_state.get("us_result") if uploaded else None

        if us_res:
            pred  = us_res.get("prediction", "Normal")
            conf  = us_res.get("confidence", 0)
            risk  = us_res.get("risk_level", "Low")
            proba = us_res.get("probabilities", {})
            note  = us_res.get("clinical_note", "")

            pred_cls  = {"Normal":"pred-normal","Benign":"pred-benign","Malignant":"pred-malignant"}.get(pred,"pred-normal")
            top_bar   = {"Normal":"result-bar-green","Benign":"result-bar-yellow","Malignant":"result-bar-red"}.get(pred,"result-bar-green")
            badge_cls = {"Normal":"badge-low","Benign":"badge-med","Malignant":"badge-high"}.get(pred,"badge-low")
            act_cls   = {"Normal":"action-green","Benign":"action-yellow","Malignant":"action-red"}.get(pred,"action-green")
            actions   = {
                "Normal":    "No immediate intervention required. Continue routine health screening. Repeat scan in 12 months if risk factors are present.",
                "Benign":    "Benign lesion detected. Recommend contrast-enhanced ultrasound (CEUS) or MRI. Follow-up scan advised in 3–6 months.",
                "Malignant": "URGENT: Findings suspicious for malignancy. Order triphasic CT/MRI liver. Refer to oncology for multidisciplinary evaluation and biopsy.",
            }

            # Probability bar HTML
            prob_bars = ""
            for cls, clr in [("Normal","#16a34a"),("Benign","#d97706"),("Malignant","#dc2626")]:
                v = proba.get(cls, 0)
                prob_bars += f"""
                <div class="prob-row">
                    <span class="prob-label">{cls}</span>
                    <div class="prob-bg">
                        <div style="height:100%;width:{v}%;background:{clr};border-radius:4px"></div>
                    </div>
                    <span class="prob-val">{v:.1f}%</span>
                </div>"""

            st.markdown(f"""
            <div class="result-wrap">
                <div class="{top_bar}"></div>
                <div style="padding:22px 24px;border-bottom:1px solid #f3f4f6">
                    <div style="font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#9ca3af;margin-bottom:8px">Scan Result</div>
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
                        <span class="{pred_cls}">{pred}</span>
                        <span class="{badge_cls}">{risk} Risk</span>
                    </div>
                    <div style="font-size:13px;color:#6b7280">Confidence: <strong style='color:#374151'>{conf:.1f}%</strong></div>
                </div>

                <div style="padding:16px 24px;border-bottom:1px solid #f3f4f6">
                    <div style="font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#9ca3af;margin-bottom:12px">Class Probabilities</div>
                    {prob_bars}
                </div>

                <div style="padding:16px 24px;border-bottom:1px solid #f3f4f6">
                    <div style="font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#9ca3af;margin-bottom:8px">◈ Radiological Note</div>
                    <p style="font-size:13px;color:#6b7280;line-height:1.7;margin:0">{note}</p>
                </div>

                <div style="padding:16px 24px">
                    <div style="font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#9ca3af;margin-bottom:8px">Recommended Action</div>
                    <div class="{act_cls}">{actions[pred]}</div>
                </div>

                <div class="disclaimer-box">⚠ Automated analysis only · All findings must be confirmed by a qualified radiologist</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  HOW IT WORKS  (no technical details)
# ═══════════════════════════════════════════════════════════════
st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;margin-bottom:28px">
    <div style="font-size:11px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:#9ca3af;margin-bottom:8px">Why Livo Health</div>
    <div style="font-size:26px;font-weight:800;letter-spacing:-.6px;color:#111827">Comprehensive Liver Health in Two Steps</div>
</div>
""", unsafe_allow_html=True)

hw1, hw2, hw3 = st.columns(3, gap="medium")

cards_info = [
    ("🩸", "Blood Test Analysis", "Enter your standard liver function test results — bilirubin, enzymes, and protein levels — to get an instant risk assessment and per-marker breakdown."),
    ("🔬", "Ultrasound Interpretation", "Upload your liver ultrasound scan and receive an AI-powered interpretation identifying normal liver, benign lesions, or suspicious findings."),
    ("📋", "Actionable Guidance", "Every result comes with a clear clinical interpretation, risk level indicator, and recommended next steps to help you take the right action."),
]

for col, (icon, title, desc) in zip([hw1, hw2, hw3], cards_info):
    with col:
        st.markdown(f"""
        <div class="livo-card" style="text-align:center">
            <div style="font-size:36px;margin-bottom:14px">{icon}</div>
            <div style="font-weight:700;font-size:15px;color:#111827;margin-bottom:8px">{title}</div>
            <div style="font-size:13px;color:#9ca3af;line-height:1.7">{desc}</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="livo-footer">
    <div><strong style="color:#6b7280">🫀 Livo Health</strong> — Your Liver Health is Important too</div>
    <div>⚠ For informational use only · Not a substitute for professional medical advice</div>
</div>
""", unsafe_allow_html=True)
