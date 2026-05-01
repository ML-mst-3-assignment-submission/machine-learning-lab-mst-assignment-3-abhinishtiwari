"""
app.py  (UPGRADED UI)
---------------------
Flask web app — shows ALL 5 ML models + ANN + RNN + Google Fact Check API.
Drop-in replacement: same backend logic, completely redesigned frontend.
"""

import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import joblib
import requests
from flask import Flask, render_template_string, request, jsonify

MAX_LEN = 100
API_KEY = "AIzaSyBV_AR3NID-5Lo861qZXI5clMG-qCy2Twg"

app = Flask(__name__)

# ─── Load models ──────────────────────────────────────────────────────────────

def load_models():
    ml_ready, dl_ready = False, False
    tfidf = trained_ml = tokenizer = ann = rnn = None

    if os.path.exists("models/tfidf_vectorizer.pkl") and os.path.exists("models/all_ml_models.pkl"):
        try:
            tfidf      = joblib.load("models/tfidf_vectorizer.pkl")
            trained_ml = joblib.load("models/all_ml_models.pkl")
            ml_ready   = True
            print("ML models loaded.")
        except Exception as e:
            print(f"ML load error: {e}")
    elif os.path.exists("models/best_ml_model.pkl"):
        try:
            best = joblib.load("models/best_ml_model.pkl")
            tfidf = best.named_steps.get("tfidf")
            clf   = best.named_steps.get("model")
            trained_ml = {"Best ML Model": clf}
            ml_ready   = True
            print("Fallback: loaded best_ml_model.pkl")
        except Exception as e:
            print(f"Fallback ML load error: {e}")
    else:
        print("No ML models found. Run Train.py first.")

    try:
        try:
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing.sequence import pad_sequences
        except ImportError:
            from keras.models import load_model
            from keras.preprocessing.sequence import pad_sequences

        if (os.path.exists("models/tokenizer.pkl") and
            os.path.exists("models/ann_model.keras") and
            os.path.exists("models/rnn_model.keras")):
            tokenizer = joblib.load("models/tokenizer.pkl")
            ann       = load_model("models/ann_model.keras")
            rnn       = load_model("models/rnn_model.keras")
            dl_ready  = True
            print("DL models loaded.")
        else:
            print("DL model files not found (run Train.py to generate them).")
    except Exception as e:
        print(f"DL load skipped: {e}")

    return tfidf, trained_ml, tokenizer, ann, rnn, ml_ready, dl_ready

tfidf, trained_ml, tokenizer, ann_model, rnn_model, ML_READY, DL_READY = load_models()

# ─── Fact Check ───────────────────────────────────────────────────────────────

def fact_check_api(query):
    url    = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": query, "key": API_KEY}
    try:
        r = requests.get(url, params=params, timeout=5)
        d = r.json()
        claims = d.get("claims", [])
        if not claims:
            return "No fact-check found"
        rv = claims[0]["claimReview"][0]
        return f"{rv['publisher']['name']}: {rv['textualRating']}"
    except Exception as e:
        return "API unavailable"

# ─── HTML ─────────────────────────────────────────────────────────────────────

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fake News Detector</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}

:root {
  --bg:       #eef4ff;
  --surface:  #ffffff;
  --surface2: #f6f8ff;
  --border:   rgba(15,23,42,0.12);
  --border2:  rgba(15,23,42,0.18);
  --text:     #0b1220;
  --muted:    rgba(11,18,32,0.62);
  --accent:   #2563eb;
  --accent2:  #7c3aed;
  --real-bg:  rgba(16,185,129,0.12);
  --real-bd:  rgba(16,185,129,0.22);
  --real-txt: #059669;
  --fake-bg:  rgba(220,38,38,0.10);
  --fake-bd:  rgba(220,38,38,0.18);
  --fake-txt: #dc2626;
}

html,body{height:100%}
body{
  font-family:'DM Sans',system-ui,sans-serif;
  background:radial-gradient(1200px 900px at 10% 0%, rgba(124,58,237,0.10), transparent 55%),
             radial-gradient(1100px 900px at 90% 20%, rgba(37,99,235,0.10), transparent 55%),
             var(--bg);
  color:var(--text);
  min-height:100vh;
  overflow-x:hidden;
}

/* ── Orb background ── */
.orb{position:fixed;border-radius:50%;filter:blur(80px);pointer-events:none;z-index:0}
.orb1{width:520px;height:520px;top:-160px;left:-120px;background:radial-gradient(circle,rgba(124,58,237,0.16),transparent 70%)}
.orb2{width:460px;height:460px;bottom:-160px;right:-120px;background:radial-gradient(circle,rgba(37,99,235,0.16),transparent 70%)}
.orb3{width:320px;height:320px;top:42%;left:48%;background:radial-gradient(circle,rgba(16,185,129,0.12),transparent 70%)}

/* ── Layout ── */
.shell{display:flex;min-height:100vh;position:relative;z-index:1}

/* ── Sidebar ── */
.sidebar{
  width:230px;flex-shrink:0;
  background:rgba(255,255,255,0.72);
  border-right:1px solid var(--border);
  backdrop-filter:blur(20px);
  display:flex;flex-direction:column;
  padding:0;
  position:sticky;top:0;height:100vh;
}
.logo{
  padding:26px 22px 22px;
  border-bottom:1px solid var(--border);
}
.logo-icon{font-size:22px;margin-bottom:6px}
.logo-title{font-family:'Syne',sans-serif;font-size:14px;font-weight:800;color:var(--accent2);letter-spacing:.05em}
.logo-sub{font-size:10px;color:var(--muted);letter-spacing:.12em;text-transform:uppercase;margin-top:1px}

.nav{padding:16px 12px;flex:1}
.nav-item{
  display:flex;align-items:center;gap:10px;
  width:100%;padding:10px 12px;
  border-radius:10px;border:none;
  background:transparent;color:var(--muted);
  font-family:'DM Sans',sans-serif;font-size:13px;
  cursor:pointer;text-align:left;
  transition:all .2s;margin-bottom:4px;
}
.nav-item:hover{background:rgba(15,23,42,0.04);color:var(--text)}
.nav-item.active{background:rgba(124,58,237,0.10);color:var(--accent2);font-weight:600}
.nav-icon{font-size:15px;width:18px;text-align:center}

.sidebar-footer{
  margin:0 12px 14px;padding:12px 14px;
  background:rgba(15,23,42,0.03);
  border-radius:10px;border:1px solid var(--border);
  font-size:10.5px;color:var(--muted);line-height:1.6;
}

/* ── Main ── */
.main{flex:1;padding:36px 44px;overflow-y:auto}

/* ── Top bar ── */
.topbar{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:32px;gap:16px}
.page-title{font-family:'Syne',sans-serif;font-size:26px;font-weight:800;line-height:1.1;margin-bottom:4px}
.gradient-text{background:linear-gradient(90deg,var(--accent2),var(--accent));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.page-sub{font-size:12px;color:var(--muted)}
.ai-badge{
  display:flex;align-items:center;gap:6px;
  background:rgba(37,99,235,0.08);border:1px solid rgba(37,99,235,0.18);
  border-radius:20px;padding:6px 14px;font-size:11.5px;color:var(--accent);
  flex-shrink:0;white-space:nowrap;
}
.badge-dot{width:6px;height:6px;border-radius:50%;background:var(--accent);animation:pulse 2s ease infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}

/* ── Warning banners ── */
.warning{
  background:rgba(251,191,36,0.16);border-left:3px solid #f59e0b;
  border-radius:8px;padding:11px 15px;
  color:#92400e;font-size:12.5px;margin-bottom:18px;line-height:1.5;
}
.warning code{background:rgba(15,23,42,0.06);padding:1px 6px;border-radius:4px;font-size:11.5px;color:inherit}

/* ── Input card ── */
.input-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:16px;padding:24px;margin-bottom:22px;
}
.field-label{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-bottom:10px}
textarea{
  width:100%;min-height:88px;
  background:rgba(15,23,42,0.02);
  border:1px solid var(--border);border-radius:10px;
  color:var(--text);font-family:'DM Sans',sans-serif;font-size:13.5px;
  padding:13px 15px;resize:vertical;outline:none;
  transition:border-color .25s;
}
textarea:focus{border-color:rgba(37,99,235,0.45)}
textarea::placeholder{color:rgba(11,18,32,0.38)}
.input-meta{display:flex;justify-content:space-between;align-items:center;margin-top:8px;font-size:11px;color:var(--muted)}
.analyse-btn{
  margin-top:14px;width:100%;padding:13px;
  background:linear-gradient(90deg,var(--accent),var(--accent2));
  border:none;border-radius:10px;
  color:#fff;font-family:'Syne',sans-serif;font-weight:700;font-size:14px;
  cursor:pointer;letter-spacing:.02em;
  transition:opacity .2s,transform .15s;
}
.analyse-btn:hover{opacity:.88;transform:translateY(-1px)}
.analyse-btn:disabled{opacity:.35;cursor:not-allowed;transform:none}

/* ── Loader ── */
.loader{display:none;align-items:center;justify-content:center;gap:8px;margin-top:14px;color:var(--muted);font-size:12.5px}
.loader.show{display:flex}
.spinner{width:16px;height:16px;border:2.5px solid rgba(15,23,42,0.18);border-top-color:var(--accent);border-radius:50%;animation:spin .75s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── Results ── */
#output{animation:fadeIn .4s ease}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}

/* ── Verdict banner ── */
.verdict-banner{
  border-radius:16px;padding:22px 26px;margin-bottom:20px;
  display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:20px;
}
.verdict-banner.real{background:var(--real-bg);border:1px solid var(--real-bd)}
.verdict-banner.fake{background:var(--fake-bg);border:1px solid var(--fake-bd)}
.verdict-left{display:flex;align-items:center;gap:16px}
.verdict-emoji{font-size:38px;line-height:1}
.verdict-label{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-bottom:3px}
.verdict-text{font-family:'Syne',sans-serif;font-size:34px;font-weight:900;line-height:1}
.verdict-text.real{color:var(--real-txt)}
.verdict-text.fake{color:var(--fake-txt)}
.verdict-sub{font-size:11.5px;color:var(--muted);margin-top:4px}
.verdict-right{display:flex;align-items:center;gap:12px}
.conf-ring{position:relative;width:76px;height:76px;flex-shrink:0}
.conf-ring svg{transform:rotate(-90deg)}
.conf-ring-label{
  position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;text-align:center;
}
.conf-ring-val{font-family:'Syne',sans-serif;font-size:17px;font-weight:800}
.conf-ring-unit{font-size:9px;color:var(--muted)}
.vote-pills{display:flex;gap:6px;margin-top:6px}
.pill{padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600}
.pill-real{background:rgba(16,185,129,0.14);color:var(--real-txt)}
.pill-fake{background:rgba(220,38,38,0.10);color:var(--fake-txt)}

/* ── Section title ── */
.section-title{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin:22px 0 12px}

/* ── Model cards ── */
.models-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(148px,1fr));gap:11px;margin-bottom:4px}
.model-card{
  border-radius:13px;padding:15px 13px;position:relative;overflow:hidden;
  transition:transform .2s;cursor:default;
}
.model-card:hover{transform:translateY(-2px)}
.model-card.real{background:linear-gradient(145deg, rgba(16,185,129,0.12), rgba(16,185,129,0.05));border:1px solid rgba(16,185,129,0.22)}
.model-card.fake{background:linear-gradient(145deg, rgba(220,38,38,0.10), rgba(220,38,38,0.04));border:1px solid rgba(220,38,38,0.18)}
.mc-ghost{position:absolute;top:-14px;right:-10px;font-size:52px;opacity:.07;pointer-events:none}
.mc-icon-name{font-size:10.5px;color:rgba(11,18,32,0.62);margin-bottom:5px}
.mc-verdict{font-family:'Syne',sans-serif;font-size:17px;font-weight:800;margin-bottom:7px}
.mc-verdict.real{color:var(--real-txt)}
.mc-verdict.fake{color:var(--fake-txt)}
.conf-bar-wrap{height:3px;background:rgba(15,23,42,0.08);border-radius:2px}
.conf-bar-fill{height:100%;border-radius:2px;transition:width .8s cubic-bezier(.4,0,.2,1)}
.mc-conf{font-size:10px;color:rgba(11,18,32,0.52);margin-top:5px}

/* ── Bottom grid ── */
.bottom-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:18px}
@media(max-width:700px){.bottom-grid{grid-template-columns:1fr}}

.info-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:13px;padding:16px 18px;
}
.info-card-label{font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-bottom:9px}
.info-card-body{font-size:13px;color:rgba(11,18,32,0.70);line-height:1.55}

.api-card{
  background:rgba(37,99,235,0.08);
  border:1px solid rgba(37,99,235,0.16);
  border-radius:13px;padding:16px 18px;
}
.api-card-label{font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:rgba(37,99,235,0.65);margin-bottom:9px}
.api-card-body{font-size:13px;color:rgba(37,99,235,0.90);line-height:1.55}

/* ── Prob distribution ── */
.prob-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:13px;padding:16px 18px;margin-top:14px;
}
.prob-row{margin-bottom:11px}
.prob-row:last-child{margin-bottom:0}
.prob-row-head{display:flex;justify-content:space-between;font-size:12.5px;margin-bottom:5px}
.prob-row-label{color:rgba(255,255,255,0.55)}
.prob-row-val{font-weight:600}
.prob-track{height:5px;background:rgba(15,23,42,0.08);border-radius:3px}
.prob-fill{height:100%;border-radius:3px;transition:width .9s cubic-bezier(.4,0,.2,1)}

/* ── History tab ── */
.hist-item{
  background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:13px 16px;margin-bottom:8px;
  display:flex;align-items:center;justify-content:space-between;
  cursor:pointer;transition:background .2s,border-color .2s;
}
.hist-item:hover{background:var(--surface2);border-color:var(--border2)}
.hist-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;margin-right:10px}
.hist-dot.real{background:var(--real-txt)}
.hist-dot.fake{background:var(--fake-txt)}
.hist-stmt{font-size:12.5px;color:rgba(11,18,32,0.72);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hist-meta{display:flex;align-items:center;gap:12px;flex-shrink:0}
.hist-verdict{font-size:11.5px;font-weight:700}
.hist-verdict.real{color:var(--real-txt)}
.hist-verdict.fake{color:var(--fake-txt)}
.hist-time{font-size:10.5px;color:var(--muted)}
.hist-empty{text-align:center;padding:48px 0;color:var(--muted);font-size:13.5px}

/* ── DL badge ── */
.dl-badge{
  display:inline-flex;align-items:center;gap:5px;
  background:rgba(124,58,237,0.10);border:1px solid rgba(124,58,237,0.18);
  border-radius:6px;padding:3px 9px;font-size:10.5px;color:rgba(124,58,237,0.92);
  margin-left:8px;vertical-align:middle;
}

footer{text-align:center;padding:20px;font-size:10.5px;color:rgba(11,18,32,0.42)}
</style>
</head>
<body>

<div class="orb orb1"></div>
<div class="orb orb2"></div>
<div class="orb orb3"></div>

<div class="shell">

  <!-- Sidebar -->
  <aside class="sidebar">
    <div class="logo">
      <div class="logo-icon">🔍</div>
      <div class="logo-title">Fake News</div>
      <div class="logo-sub">Detector</div>
    </div>
    <nav class="nav">
      <button class="nav-item active" id="nav-analyze" onclick="switchTab('analyze')">
        <span class="nav-icon">⚡</span> Analyze News
      </button>
      <button class="nav-item" id="nav-history" onclick="switchTab('history')">
        <span class="nav-icon">🕐</span> History
      </button>
      <button class="nav-item" id="nav-stats" onclick="switchTab('stats')">
        <span class="nav-icon">📊</span> Statistics
      </button>
    </nav>
    <div class="sidebar-footer">
      Uses 5 ML + 2 DL models with majority-vote ensemble &amp; Google Fact Check API
    </div>
  </aside>

  <!-- Main -->
  <main class="main">

    <!-- Top bar -->
    <div class="topbar">
      <div>
        <div class="page-title"><span class="gradient-text">🧠 Fake News Detection System</span></div>
        <div class="page-sub">ML + Deep Learning + Google Fact Check API</div>
      </div>
      
    </div>

    <!-- Warnings -->
    {% if not ml_ready %}
    <div class="warning">⚠️ Models not found. Run <code>python Train.py</code> first, then restart <code>python app.py</code>.</div>
    {% endif %}
    {% if ml_ready and not dl_ready %}
    <div class="warning">ℹ️ Deep learning models (ANN/RNN) not available. Run <code>python Train.py</code> to enable them. ML models are ready.</div>
    {% endif %}

    <!-- ─── ANALYZE TAB ─── -->
    <div id="tab-analyze">
      <div class="input-card">
        <div class="field-label">Enter News Statement</div>
        <textarea id="stmt" rows="3" placeholder="Paste or type a news headline or statement here…" oninput="updateChar(this)" onkeydown="if(event.ctrlKey&&event.key==='Enter')analyse()"></textarea>
        <div class="input-meta">
          <span>Ctrl + Enter to analyze</span>
          <span id="char-count">0 / 1000</span>
        </div>
        <button class="analyse-btn" id="btn" onclick="analyse()" {% if not ml_ready %}disabled{% endif %}>
          🚀 Analyze Statement
        </button>
        <div class="loader" id="loader">
          <div class="spinner"></div>
          Running all models…
        </div>
      </div>

      <div id="output"></div>
    </div>

    <!-- ─── HISTORY TAB ─── -->
    <div id="tab-history" style="display:none">
      <div class="section-title" style="margin-top:0">Recent History</div>
      <div id="hist-list"></div>
    </div>

    <!-- ─── STATS TAB ─── -->
    <div id="tab-stats" style="display:none">
      <div class="section-title" style="margin-top:0">Session Statistics</div>
      <div id="stats-content"></div>
    </div>

  </main>
</div>

<footer>PolitiFact Dataset &nbsp;·&nbsp; scikit-learn + TensorFlow + Flask &nbsp;·&nbsp; SMOTE balanced training</footer>

<script>
/* ─── Tab switching ─── */
let currentTab = 'analyze';
function switchTab(tab) {
  ['analyze','history','stats'].forEach(t => {
    document.getElementById('tab-'+t).style.display = t===tab ? 'block' : 'none';
    document.getElementById('nav-'+t).classList.toggle('active', t===tab);
  });
  currentTab = tab;
  if (tab === 'history') renderHistory();
  if (tab === 'stats')   renderStats();
}

/* ─── History ─── */
let history = JSON.parse(localStorage.getItem('fnd_history')||'[]');
function saveHistory() { localStorage.setItem('fnd_history', JSON.stringify(history.slice(0,30))); }

function renderHistory() {
  const el = document.getElementById('hist-list');
  if (!history.length) {
    el.innerHTML = '<div class="hist-empty">No analyses yet — go analyze some news!</div>';
    return;
  }
  el.innerHTML = history.map((h,i) => `
    <div class="hist-item" onclick="loadStatement('${escHtml(h.stmt)}')">
      <span class="hist-dot ${h.verdict.toLowerCase()}"></span>
      <span class="hist-stmt">${escHtml(h.stmt)}</span>
      <div class="hist-meta">
        <span class="hist-verdict ${h.verdict.toLowerCase()}">${h.verdict} · ${h.conf}%</span>
        <span class="hist-time">${h.time}</span>
      </div>
    </div>`).join('');
}

function loadStatement(stmt) {
  document.getElementById('stmt').value = stmt;
  updateChar(document.getElementById('stmt'));
  switchTab('analyze');
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/* ─── Stats ─── */
function renderStats() {
  const el = document.getElementById('stats-content');
  if (!history.length) {
    el.innerHTML = '<div class="hist-empty">No data yet.</div>';
    return;
  }
  const real = history.filter(h=>h.verdict==='Real').length;
  const fake = history.filter(h=>h.verdict==='Fake').length;
  const avgConf = Math.round(history.reduce((s,h)=>s+h.conf,0)/history.length);
  el.innerHTML = `
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px;margin-bottom:20px">
      ${statCard('Total Analyses', history.length, '#818cf8')}
      ${statCard('Real News', real, '#34d399')}
      ${statCard('Fake News', fake, '#f87171')}
      ${statCard('Avg Confidence', avgConf+'%', '#fbbf24')}
    </div>`;
}

function statCard(label, val, color) {
  return `<div style="background:var(--surface);border:1px solid var(--border);border-radius:13px;padding:18px;text-align:center">
    <div style="font-size:10px;letter-spacing:.1em;color:var(--muted);text-transform:uppercase;margin-bottom:8px">${label}</div>
    <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:${color}">${val}</div>
  </div>`;
}

/* ─── Char counter ─── */
function updateChar(el) {
  document.getElementById('char-count').textContent = el.value.length + ' / 1000';
}

/* ─── Radial ring ─── */
function ring(conf, color, size=76) {
  const r = (size-10)/2, circ = 2*Math.PI*r, dash = (conf/100)*circ;
  return `<div class="conf-ring" style="width:${size}px;height:${size}px">
    <svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
      <circle cx="${size/2}" cy="${size/2}" r="${r}" fill="none" stroke="rgba(15,23,42,0.12)" stroke-width="6"/>
      <circle cx="${size/2}" cy="${size/2}" r="${r}" fill="none" stroke="${color}" stroke-width="6"
        stroke-dasharray="${dash.toFixed(1)} ${circ.toFixed(1)}" stroke-linecap="round"/>
    </svg>
    <div class="conf-ring-label">
      <div class="conf-ring-val" style="color:${color}">${conf}%</div>
      <div class="conf-ring-unit">conf</div>
    </div>
  </div>`;
}

/* ─── Model card ─── */
const MC_ICONS = {
  "Logistic Regression":"📐","Decision Tree":"🌳","Random Forest":"🌲",
  "Naive Bayes":"📊","SVM":"⚡","ANN":"🧠","RNN":"🔁","Best ML Model":"🤖"
};
function modelCard(name, label, conf) {
  const isReal = label==='Real';
  const icon = MC_ICONS[name]||'🤖';
  const color = isReal ? '#6ee7b7' : '#fca5a5';
  return `<div class="model-card ${isReal?'real':'fake'}">
    <div class="mc-ghost">${icon}</div>
    <div class="mc-icon-name">${icon} ${name}</div>
    <div class="mc-verdict ${isReal?'real':'fake'}">${isReal?'✓':'✗'} ${label}</div>
    <div class="conf-bar-wrap">
      <div class="conf-bar-fill" style="width:${conf}%;background:${color}"></div>
    </div>
    <div class="mc-conf">${conf}% confidence</div>
  </div>`;
}

/* ─── Main analyse ─── */
async function analyse() {
  const stmt = document.getElementById('stmt').value.trim();
  if (!stmt) { alert('Please enter a statement.'); return; }
  document.getElementById('btn').disabled = true;
  document.getElementById('loader').classList.add('show');
  document.getElementById('output').innerHTML = '';

  try {
    const res = await fetch('/predict', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({statement: stmt})
    });
    const d = await res.json();

    if (d.error) {
      document.getElementById('output').innerHTML =
        `<div class="warning">❌ ${d.error}</div>`;
      return;
    }

    /* ── Verdict ── */
    const ov = d.overall;
    const isReal = ov.verdict==='Real';
    const color = isReal ? '#34d399' : '#f87171';
    const allConfs = [
      ...Object.values(d.ml_models).map(m=>m.confidence),
      ...(d.dl_models ? [d.dl_models.ANN?.probability||0, d.dl_models.RNN?.probability||0] : [])
    ];
    const avgConf = Math.round(allConfs.reduce((a,b)=>a+b,0)/allConfs.length);

    /* ── ML cards ── */
    let mlCards = Object.entries(d.ml_models)
      .map(([n,r]) => modelCard(n, r.label, r.confidence)).join('');

    /* ── DL cards ── */
    let dlSection = '';
    if (d.dl_models) {
      let dlCards = '';
      if (d.dl_models.ANN) dlCards += modelCard('ANN', d.dl_models.ANN.label, d.dl_models.ANN.probability);
      if (d.dl_models.RNN) dlCards += modelCard('RNN', d.dl_models.RNN.label, d.dl_models.RNN.probability);
      if (dlCards) dlSection = `
        <div class="section-title">Deep Learning Models <span class="dl-badge">🧠 DL</span></div>
        <div class="models-grid">${dlCards}</div>`;
    }

    /* ── Prob bars ── */
    const realPct = Math.round((ov.votes_real/ov.total)*100);
    const fakePct = 100-realPct;

    /* ── Save history ── */
    history.unshift({stmt: stmt.slice(0,80), verdict: ov.verdict, conf: avgConf, time: new Date().toLocaleString()});
    saveHistory();

    document.getElementById('output').innerHTML = `
      <!-- Verdict banner -->
      <div class="verdict-banner ${isReal?'real':'fake'}">
        <div class="verdict-left">
          <div class="verdict-emoji">${isReal?'✅':'❌'}</div>
          <div>
            <div class="verdict-label">Final Verdict</div>
            <div class="verdict-text ${isReal?'real':'fake'}">${ov.verdict.toUpperCase()}</div>
            <div class="verdict-sub">Majority vote across ${ov.total} models</div>
            <div class="vote-pills">
              <span class="pill pill-real">✓ Real: ${ov.votes_real}</span>
              <span class="pill pill-fake">✗ Fake: ${ov.votes_fake}</span>
            </div>
          </div>
        </div>
        <div class="verdict-right">
          ${ring(avgConf, color)}
          <div>
            <div style="font-family:'Syne',sans-serif;font-size:11px;color:var(--muted)">Confidence<br>Score</div>
          </div>
        </div>
      </div>

      <!-- ML Models -->
      <div class="section-title">Machine Learning Models</div>
      <div class="models-grid">${mlCards}</div>

      <!-- DL Models -->
      ${dlSection}

      <!-- Bottom info row -->
      <div class="bottom-grid">
        <div class="api-card">
          <div class="api-card-label">🌐 Google Fact Check API</div>
          <div class="api-card-body">${d.api}</div>
        </div>
        <div class="prob-card" style="margin-top:0">
          <div class="info-card-label">Probability Distribution</div>
          <div class="prob-row">
            <div class="prob-row-head">
              <span class="prob-row-label">Fake</span>
              <span class="prob-row-val" style="color:var(--fake-txt)">${fakePct}%</span>
            </div>
            <div class="prob-track"><div class="prob-fill" style="width:${fakePct}%;background:var(--fake-txt)"></div></div>
          </div>
          <div class="prob-row">
            <div class="prob-row-head">
              <span class="prob-row-label">Real</span>
              <span class="prob-row-val" style="color:var(--real-txt)">${realPct}%</span>
            </div>
            <div class="prob-track"><div class="prob-fill" style="width:${realPct}%;background:var(--real-txt)"></div></div>
          </div>
        </div>
      </div>

      <!-- Recent history snippet -->
      <div class="section-title" style="margin-top:22px">Recent History</div>
      <div id="inline-history"></div>
    `;

    /* Inline mini-history */
    const recentHTML = history.slice(0,4).map(h=>`
      <div class="hist-item" onclick="loadStatement('${escHtml(h.stmt)}')">
        <span class="hist-dot ${h.verdict.toLowerCase()}"></span>
        <span class="hist-stmt">${escHtml(h.stmt)}</span>
        <div class="hist-meta">
          <span class="hist-verdict ${h.verdict.toLowerCase()}">${h.verdict} · ${h.conf}%</span>
          <span class="hist-time">${h.time}</span>
        </div>
      </div>`).join('');
    document.getElementById('inline-history').innerHTML = recentHTML;

  } catch(e) {
    document.getElementById('output').innerHTML =
      `<div class="warning">❌ Request failed: ${e}</div>`;
  } finally {
    document.getElementById('btn').disabled = false;
    document.getElementById('loader').classList.remove('show');
  }
}
</script>
</body>
</html>
"""

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML, ml_ready=ML_READY, dl_ready=DL_READY)


@app.route("/predict", methods=["POST"])
def predict():
    if not ML_READY:
        return jsonify({"error": "Models not loaded. Run python Train.py first."})

    data = request.get_json()
    stmt = (data or {}).get("statement", "").strip()
    if not stmt:
        return jsonify({"error": "Empty statement."})

    try:
        # ── ML models ─────────────────────────────────────────────────────
        x_vec   = tfidf.transform([stmt])
        ml_preds = {}
        for name, clf in trained_ml.items():
            try:
                if "Naive" in name:
                    import scipy.sparse as sp
                    xnb = x_vec.copy(); xnb.data = np.abs(xnb.data)
                    pred  = int(clf.predict(xnb)[0])
                    proba = clf.predict_proba(xnb)[0]
                else:
                    pred  = int(clf.predict(x_vec)[0])
                    proba = clf.predict_proba(x_vec)[0]
                ml_preds[name] = {
                    "label":      "Real" if pred == 1 else "Fake",
                    "confidence": round(float(max(proba)) * 100, 1),
                }
            except Exception as e:
                ml_preds[name] = {"label": "Error", "confidence": 0.0}

        # ── DL models ─────────────────────────────────────────────────────
        dl_preds = None
        if DL_READY:
            try:
                from keras.preprocessing.sequence import pad_sequences
            except:
                from tensorflow.keras.preprocessing.sequence import pad_sequences

            seq    = tokenizer.texts_to_sequences([stmt])
            padded = pad_sequences(seq, maxlen=MAX_LEN)
            ann_p  = float(ann_model.predict(padded, verbose=0)[0][0])
            rnn_p  = float(rnn_model.predict(padded, verbose=0)[0][0])
            dl_preds = {
                "ANN": {"label": "Real" if ann_p > .5 else "Fake", "probability": round(ann_p*100,1)},
                "RNN": {"label": "Real" if rnn_p > .5 else "Fake", "probability": round(rnn_p*100,1)},
            }

        # ── Majority vote ─────────────────────────────────────────────────
        all_labels = [r["label"] for r in ml_preds.values()]
        if dl_preds:
            all_labels += [dl_preds["ANN"]["label"], dl_preds["RNN"]["label"]]
        votes_real = all_labels.count("Real")
        votes_fake = all_labels.count("Fake")
        overall    = "Real" if votes_real > votes_fake else "Fake"

        return jsonify({
            "ml_models": ml_preds,
            "dl_models": dl_preds,
            "overall":   {"verdict": overall, "votes_real": votes_real,
                          "votes_fake": votes_fake, "total": len(all_labels)},
            "api":       fact_check_api(stmt),
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Fake News Detector — Flask App (Upgraded UI)")
    print(f"  ML models: {'Ready' if ML_READY else 'NOT READY — run Train.py'}")
    print(f"  DL models: {'Ready' if DL_READY else 'Not loaded'}")
    print("  Open: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)