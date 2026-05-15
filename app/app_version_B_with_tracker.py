"""
PhoneCheck AI — Professional Edition v2
=========================================
8-capture camera flow:
  Front: 2 angles (flat, 45deg) — CNN on all, worst wins
  Back:  2 angles (flat, 45deg) — CNN on all, worst wins
  Edges: 4 photos (left, right, top, bottom) — Edge CNN
  2+ edges used — overall = USED override

Run: streamlit run app_ensemble.py
"""
import os, io, base64, numpy as np, requests, json, re, cv2
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force TF to CPU — save GPU for Ollama LLM
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "models")
IMG_SIZE = (224, 224)
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:8b")

CATEGORIES = ["damaged", "mint", "used"]
EDGE_CATEGORIES = ["mint", "used"]
CLASS_CONFIG = {
    "mint":    {"dot": "#34D399", "label": "MINT",    "tag": "Sell As-Is",      "desc": "Your phone looks great — no damage found. You can sell it at full price.",      "emoji": "✅", "resale_pct": 1.0},
    "used":    {"dot": "#FBBF24", "label": "USED",    "tag": "Discount 15-20%", "desc": "Some scratches or wear, but everything works fine. Discount the price a little.", "emoji": "⚡", "resale_pct": 0.82},
    "damaged": {"dot": "#F87171", "label": "DAMAGED", "tag": "Repair First",    "desc": "We found cracks or damage. Fix it first, then sell for more money.",              "emoji": "🔴", "resale_pct": 0.55},
}
GRADE_ORDER = {"damaged": 0, "used": 1, "mint": 2}

# Per-model CLAHE (matched to classifier A/B test results)
CLAHE_FRONT  = False   # Front: no gain, hurt scratched
CLAHE_BACK   = True    # Back: +4.6% overall, +10% cracked
CLAHE_SIDE   = False   # Merged edge: NO-CLAHE won (73.3% vs 71.1%)
CLAHE_TOPBOT = False   # Merged edge: NO-CLAHE won

# Confidence thresholds — "good"/"mint" must exceed this to be predicted
GOOD_THRESHOLD = 0.6   # 3-class: must be >60% confident for "mint" (good)
MINT_THRESHOLD = 0.6   # 2-class: must be >60% confident for "mint"

def apply_clahe_pil(pil_image):
    """Apply CLAHE to PIL image, return enhanced PIL image."""
    img = np.array(pil_image.convert('RGB'))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)

def apply_sharpen_pil(pil_image):
    """Apply UnsharpMask sharpening to PIL image — matches training preprocessing for front model."""
    from PIL import ImageFilter
    img = pil_image.convert('RGB')
    return img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

# ============================================================
# YOLO EDGE CROPPING (matches training pipeline)
# ============================================================
YOLO_EDGE_MODEL = None

def apply_sharpen_pil(pil_image):
    ...

# ============================================================
# YOLO EDGE CROPPING (matches training pipeline)
# ============================================================
YOLO_EDGE_MODEL = None

def yolo_crop_pil(pil_image, yolo_model):
    """YOLO crop: detect phone edge → crop with padding → return cropped PIL image."""
    if yolo_model is None:
        return pil_image
    ...
    return pil_image

@st.cache_resource
def load_yolo_model():
    """Load YOLOv11 edge detector for cropping phone edges before classification."""
    yolo_path = os.path.join(MODELS_DIR, "best_yolo_v22.pt")
    ...

def yolo_crop_pil(pil_image, yolo_model):
    """YOLO crop: detect phone edge → crop with padding → return cropped PIL image."""
    if yolo_model is None:
        return pil_image
    try:
        img_array = np.array(pil_image.convert('RGB'))
        results = yolo_model.predict(img_array, conf=0.25, verbose=False, device="cpu")
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_idx = int(boxes.conf.argmax())
            x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
            h, w = img_array.shape[:2]
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
            cropped = img_array[y1:y2, x1:x2]
            if cropped.size > 0:
                return Image.fromarray(cropped)
    except:
        pass
    return pil_image

@st.cache_resource
def load_yolo_model():
    """Load YOLOv11 edge detector for cropping phone edges before classification."""
    yolo_path = os.path.join(MODELS_DIR, "best_yolo_v22.pt")
    if not os.path.exists(yolo_path):
        return None
    try:
        # Force YOLO to CPU to avoid conflict with TF GPU
        import torch
        torch.cuda.is_available = lambda: False
        backup = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from ultralytics import YOLO
        model = YOLO(yolo_path)
        os.environ["CUDA_VISIBLE_DEVICES"] = backup
        return model
    except Exception as e:
        print(f"YOLO load failed: {e}")
        return None

# ============================================================
# VIEW GATE (front/back/other validation)
# ============================================================
VIEW_CATEGORIES = ["back", "front", "other"]
VIEW_THRESHOLD = 0.6  # Must be >60% confident to accept

@st.cache_resource
def load_view_gate():
    """View gate (front/back/other) — DISABLED for now.
    Returning None makes predict_view() short-circuit and trust the user's photo
    without any rejection logic firing in the capture flow."""
    return None
    # Original loader kept below for easy re-enable:
    # path = os.path.join(SCRIPT_DIR, "phone_side_classifier.keras")
    # if os.path.exists(path):
    #     return tf.keras.models.load_model(path)
    # return None

def predict_view(model, image):
    """Classify image as front/back/other. Returns (class, confidence)."""
    if model is None:
        return "front", 1.0  # No gate = trust user
    img = image.convert('RGB').resize(IMG_SIZE)
    preds = model.predict(np.expand_dims(np.array(img)/255.0, 0), verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx])
    return VIEW_CATEGORIES[idx], conf

CUSTOM_CSS = """
<link rel="manifest" href="app/static/manifest.json">
<meta name="theme-color" content="#6366F1">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="PhoneCheck">
<link rel="apple-touch-icon" href="app/static/icon-192.png">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">

<style>@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Source+Sans+3:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap'); :root { --bg-primary: #06090F; --bg-card: #0C1018; --bg-elevated: #111827; --border: #1A2035; --text-primary: #F0F2F5; --text-secondary:#94A3B8; --text-muted: #4A5568; --accent: #6366F1; --accent-dim: rgba(99,102,241,0.08); --mint: #34D399; --used: #FBBF24; --damaged: #F87171; --radius-sm: 8px; --radius-md: 12px; --radius-lg: 16px; --radius-xl: 20px; --space-xs: 4px; --space-sm: 8px; --space-md: 16px; --space-lg: 24px; --space-xl: 32px; --transition: 200ms ease-out; } .stApp { background: var(--bg-primary); font-family: 'Source Sans 3', sans-serif; color: var(--text-primary); } #MainMenu, footer, header { visibility: hidden; } section[data-testid="stSidebar"] { background: var(--bg-card); border-right: 1px solid var(--border); } .stMarkdown, .stText, p, span, li, td, th { font-family: 'Source Sans 3', sans-serif !important; } h1,h2,h3,h4,h5,h6 { font-family: 'Outfit', sans-serif !important; letter-spacing: -0.02em; } .stButton > button { font-family: 'Outfit', sans-serif !important; font-weight: 600; border-radius: var(--radius-sm); min-height: 48px; transition: all var(--transition); } .stButton > button[kind="primary"] { background: var(--accent); color: var(--bg-primary); border: none; box-shadow: 0 2px 12px rgba(99,102,241,0.2); } .stButton > button[kind="primary"]:hover { box-shadow: 0 4px 20px rgba(99,102,241,0.35); transform: translateY(-1px); } .stButton > button[kind="primary"]:active { transform: translateY(0); box-shadow: 0 1px 8px rgba(99,102,241,0.2); } .stProgress > div > div { background: linear-gradient(90deg, var(--accent), var(--mint)); border-radius: 4px; } [data-testid="stMetric"] { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-md); padding: var(--space-md); } [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; font-weight: 600; } .stAlert { border-radius: var(--radius-md); } ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: transparent; } ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; } .stSelectbox > div > div { min-height: 44px; border-radius: var(--radius-sm) !important; transition: border-color var(--transition); } .stTabs [data-baseweb="tab-list"] { gap: 4px; background: var(--bg-card); padding: 4px; border-radius: var(--radius-md); border: 1px solid var(--border); } .stTabs [data-baseweb="tab"] { border-radius: var(--radius-sm); font-family: 'Outfit', sans-serif !important; font-weight: 500; padding: 10px 16px; min-height: 40px; transition: all var(--transition); } .stTabs [aria-selected="true"] { background: var(--accent) !important; color: var(--bg-primary) !important; } .stTabs [data-baseweb="tab-highlight"] { display: none; } .stTabs [data-baseweb="tab-border"] { display: none; } [data-testid="stFileUploader"] { background: linear-gradient(135deg, rgba(99,102,241,0.03), rgba(52,211,153,0.03)); border: 2px dashed rgba(99,102,241,0.35); border-radius: var(--radius-lg); padding: var(--space-sm); transition: all var(--transition); } [data-testid="stFileUploader"]:hover { border-color: var(--accent); background: linear-gradient(135deg, rgba(99,102,241,0.06), rgba(52,211,153,0.06)); } [data-testid="stFileUploaderDropzone"] button { background: var(--accent) !important; color: var(--bg-primary) !important; font-family: 'Outfit', sans-serif !important; font-weight: 600 !important; border: none !important; border-radius: var(--radius-sm) !important; min-height: 44px !important; min-width: 140px !important; padding: 0 24px !important; transition: all var(--transition) !important; font-size: 14px !important; } [data-testid="stFileUploaderDropzone"] button:hover { box-shadow: 0 4px 16px rgba(99,102,241,0.3) !important; } [data-testid="stFileUploaderDropzone"] button:hover { box-shadow: 0 4px 16px rgba(99,102,241,0.3) !important; } [data-testid="stFileUploader"] small { color: var(--text-muted) !important; font-size: 10px !important; } .stMarkdown p, .stMarkdown div { word-wrap: break-word; overflow-wrap: break-word; } .stApp { -webkit-tap-highlight-color: transparent; } img { border-radius: var(--radius-sm); } .stTextInput > div > div > input { border-radius: var(--radius-sm) !important; min-height: 44px; font-family: 'Source Sans 3', sans-serif !important; } [data-testid="stSelectbox"] input { caret-color: transparent !important; cursor: pointer !important; }</style>
"""

REPAIR_COSTS = {
    "Apple": {
        "iPhone 16 Pro Max":{"screen":(280,350),"back":(180,280)},"iPhone 16 Pro":{"screen":(260,330),"back":(170,260)},
        "iPhone 16 Plus":{"screen":(220,280),"back":(150,230)},"iPhone 16":{"screen":(200,260),"back":(140,220)},
        "iPhone 16e":{"screen":(140,190),"back":(100,160)},"iPhone 15 Pro Max":{"screen":(250,330),"back":(160,250)},
        "iPhone 15 Pro":{"screen":(240,310),"back":(150,240)},"iPhone 15 Plus":{"screen":(200,260),"back":(140,210)},
        "iPhone 15":{"screen":(180,240),"back":(130,200)},"iPhone 14 Pro Max":{"screen":(240,320),"back":(150,250)},
        "iPhone 14 Pro":{"screen":(230,300),"back":(140,230)},"iPhone 14 Plus":{"screen":(190,250),"back":(130,200)},
        "iPhone 14":{"screen":(170,230),"back":(120,190)},"iPhone 13 Pro Max":{"screen":(210,280),"back":(130,220)},
        "iPhone 13 Pro":{"screen":(200,260),"back":(120,200)},"iPhone 13":{"screen":(160,220),"back":(100,170)},
        "iPhone 13 Mini":{"screen":(150,210),"back":(90,160)},"iPhone 12 Pro Max":{"screen":(180,240),"back":(110,190)},
        "iPhone 12 Pro":{"screen":(170,230),"back":(100,180)},"iPhone 12":{"screen":(150,200),"back":(90,160)},
        "iPhone 12 Mini":{"screen":(140,190),"back":(80,150)},"iPhone 11 Pro Max":{"screen":(160,220),"back":(100,180)},
        "iPhone 11 Pro":{"screen":(150,210),"back":(90,170)},"iPhone 11":{"screen":(100,150),"back":(80,140)},
        "iPhone SE (2022)":{"screen":(80,120),"back":(60,100)},"iPhone SE (2020)":{"screen":(70,110),"back":(50,90)},
        "iPhone XS Max":{"screen":(140,200),"back":(90,160)},"iPhone XS":{"screen":(130,190),"back":(80,150)},
        "iPhone XR":{"screen":(100,150),"back":(70,130)},"iPhone X":{"screen":(120,180),"back":(80,140)},
    },
    "Samsung": {
        "Galaxy S25 Ultra":{"screen":(240,320),"back":(120,200)},"Galaxy S25+":{"screen":(180,250),"back":(100,170)},
        "Galaxy S25":{"screen":(160,220),"back":(90,150)},"Galaxy S24 Ultra":{"screen":(230,310),"back":(110,190)},
        "Galaxy S24+":{"screen":(170,240),"back":(100,170)},"Galaxy S24":{"screen":(150,210),"back":(80,140)},
        "Galaxy S24 FE":{"screen":(90,140),"back":(60,110)},"Galaxy S23 Ultra":{"screen":(220,300),"back":(110,190)},
        "Galaxy S23+":{"screen":(160,230),"back":(90,160)},"Galaxy S23":{"screen":(140,200),"back":(80,140)},
        "Galaxy S23 FE":{"screen":(90,140),"back":(60,110)},"Galaxy S22 Ultra":{"screen":(200,280),"back":(100,180)},
        "Galaxy S22+":{"screen":(150,210),"back":(80,150)},"Galaxy S22":{"screen":(130,190),"back":(70,130)},
        "Galaxy S21 Ultra":{"screen":(180,250),"back":(90,160)},"Galaxy S21+":{"screen":(140,200),"back":(70,130)},
        "Galaxy S21":{"screen":(120,180),"back":(60,120)},"Galaxy S21 FE":{"screen":(100,160),"back":(60,110)},
        "Galaxy Z Fold6":{"screen":(350,500),"back":(150,250)},"Galaxy Z Fold5":{"screen":(330,470),"back":(140,230)},
        "Galaxy Z Fold4":{"screen":(300,430),"back":(130,210)},"Galaxy Z Flip6":{"screen":(200,300),"back":(100,170)},
        "Galaxy Z Flip5":{"screen":(190,280),"back":(90,160)},"Galaxy Z Flip4":{"screen":(170,260),"back":(80,150)},
        "Galaxy A55":{"screen":(80,130),"back":(40,80)},"Galaxy A54":{"screen":(70,120),"back":(35,75)},
        "Galaxy A53":{"screen":(60,110),"back":(30,70)},"Galaxy A34":{"screen":(60,100),"back":(30,60)},
        "Galaxy A25":{"screen":(50,90),"back":(25,55)},"Galaxy A15":{"screen":(40,70),"back":(20,45)},
    },
    "Google": {
        "Pixel 9 Pro XL":{"screen":(220,300),"back":(120,200)},"Pixel 9 Pro":{"screen":(200,280),"back":(110,190)},
        "Pixel 9":{"screen":(160,230),"back":(90,160)},"Pixel 8 Pro":{"screen":(180,260),"back":(100,180)},
        "Pixel 8":{"screen":(140,210),"back":(80,150)},"Pixel 8a":{"screen":(100,150),"back":(60,110)},
        "Pixel 7 Pro":{"screen":(160,230),"back":(90,160)},"Pixel 7":{"screen":(120,180),"back":(70,130)},
        "Pixel 7a":{"screen":(90,140),"back":(50,100)},"Pixel 6 Pro":{"screen":(140,210),"back":(80,150)},
        "Pixel 6":{"screen":(110,170),"back":(60,120)},"Pixel 6a":{"screen":(80,130),"back":(50,100)},
    },
    "OnePlus": {
        "OnePlus 13":{"screen":(180,260),"back":(90,160)},"OnePlus 12":{"screen":(160,240),"back":(80,150)},
        "OnePlus 12R":{"screen":(100,160),"back":(50,100)},"OnePlus 11":{"screen":(140,210),"back":(70,130)},
        "OnePlus Nord 4":{"screen":(70,120),"back":(35,70)},"OnePlus Nord CE4":{"screen":(50,90),"back":(30,60)},
    },
    "Xiaomi": {
        "Xiaomi 14 Ultra":{"screen":(180,260),"back":(90,160)},"Xiaomi 14 Pro":{"screen":(150,220),"back":(80,140)},
        "Xiaomi 14":{"screen":(120,180),"back":(60,120)},"Redmi Note 13 Pro":{"screen":(60,100),"back":(30,60)},
        "Redmi Note 13":{"screen":(40,80),"back":(20,50)},"Redmi Note 12":{"screen":(35,70),"back":(20,45)},
        "POCO F6":{"screen":(60,100),"back":(30,60)},"POCO X6 Pro":{"screen":(50,90),"back":(25,55)},
    },
    "Huawei": {
        "P60 Pro":{"screen":(180,260),"back":(90,160)},"P50 Pro":{"screen":(150,220),"back":(70,140)},
        "Mate 60 Pro":{"screen":(200,300),"back":(100,180)},"Nova 12":{"screen":(70,120),"back":(35,70)},
    },
}

# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_model(model_type):
    path = os.path.join(MODELS_DIR, f"phone_classifier_{model_type}.keras")
    if os.path.exists(path): return tf.keras.models.load_model(path), 3
    return None, None

@st.cache_resource
def load_edge_model(edge_type):
    path = os.path.join(MODELS_DIR, f"edge_classifier_{edge_type}.keras")
    if os.path.exists(path): return tf.keras.models.load_model(path)
    return None

# ============================================================
# PREDICTION
# ============================================================
def predict_phone(model, image, clahe_on=False, sharpen_on=False):
    img = image.convert('RGB')
    if sharpen_on:
        img = apply_sharpen_pil(img)
    img = img.resize(IMG_SIZE)
    if clahe_on:
        img = apply_clahe_pil(img)
    preds = model.predict(np.expand_dims(np.array(img)/255.0, 0), verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx])
    if CATEGORIES[idx] == "mint" and conf < GOOD_THRESHOLD:
        masked = preds.copy(); masked[idx] = -1
        idx = int(np.argmax(masked)); conf = float(preds[idx])
    return CATEGORIES[idx], conf, {CATEGORIES[i]: float(preds[i]) for i in range(3)}

def predict_edge(model, image, clahe_on=False, yolo_model=None):
    """Predict edge condition with YOLO crop + tiling. Worst tile wins."""
    # Step 1: YOLO crop the phone edge from the photo
    cropped = yolo_crop_pil(image, yolo_model)
    # Step 2: Split into 3 overlapping tiles (preserves fine detail)
    tiles = split_into_tiles_pil(cropped, num_tiles=3, overlap=0.15)
    # Step 3: Classify each tile
    tile_results = []
    for tile in tiles:
        img = tile.convert('RGB').resize(IMG_SIZE)
        if clahe_on:
            img = apply_clahe_pil(img)
        preds = model.predict(np.expand_dims(np.array(img)/255.0, 0), verbose=0)[0]
        idx = int(np.argmax(preds))
        conf = float(preds[idx])
        if EDGE_CATEGORIES[idx] == "mint" and conf < MINT_THRESHOLD:
            idx = 1; conf = float(preds[1])
        tile_results.append((EDGE_CATEGORIES[idx], conf, {EDGE_CATEGORIES[i]: float(preds[i]) for i in range(2)}))
    # Step 4: Worst tile wins — if ANY tile says "used", the edge is used
    worst = tile_results[0]
    for r in tile_results[1:]:
        if GRADE_ORDER.get(r[0], 2) < GRADE_ORDER.get(worst[0], 2):
            worst = r
    return worst

def worst_case(results):
    if not results: return None
    w = results[0]
    for r in results[1:]:
        if GRADE_ORDER.get(r[0], 2) < GRADE_ORDER.get(w[0], 2): w = r
    return w

# ============================================================
# HELPERS
# ============================================================
def get_brands(): return sorted(REPAIR_COSTS.keys())
def get_models(brand): return sorted(REPAIR_COSTS.get(brand, {}).keys())
def get_repair_estimate(brand, model, dtype):
    if brand in REPAIR_COSTS and model in REPAIR_COSTS[brand]: return REPAIR_COSTS[brand][model].get(dtype, (None, None))
    return (None, None)
def get_edge_repair_estimate(brand, phone_model):
    sc = get_repair_estimate(brand, phone_model, "screen")
    if sc[0]: return (max(20, int(sc[0]*0.12)), max(50, int(sc[1]*0.20)))
    return (20, 60)
def image_to_base64(image):
    buf = io.BytesIO(); img = image.convert('RGB'); img.thumbnail((512,512)); img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ============================================================
# UI COMPONENTS
# ============================================================
def section_label(text):
    st.markdown(f'<div style="font-family:Outfit,sans-serif;color:#64748B;font-size:12px;font-weight:500;letter-spacing:1px;text-transform:uppercase;margin:28px 0 12px 0;">{text}</div>', unsafe_allow_html=True)

def render_header():
    st.markdown("""<div id="pc-real-header" style="padding:40px 0 8px 0;">
    <div style="font-family:Outfit;font-size:28px;font-weight:800;color:#F0F2F5;letter-spacing:-0.03em;line-height:1;">PhoneCheck<span style="font-weight:300;color:#6366F1;">AI</span></div>
    <div style="font-family:'Source Sans 3';font-size:13px;color:#64748B;margin-top:6px;">Smartphone condition assessment</div>
    </div>""", unsafe_allow_html=True)

def render_grade_card(pred_class, confidence, all_probs, show_details, label=""):
    cfg = CLASS_CONFIG[pred_class]; pct = int(confidence * 100)
    prob_bars = ""
    if show_details:
        for cat in all_probs:
            c = CLASS_CONFIG.get(cat, {"dot":"#7A8599","label":cat.upper(),"emoji":""}); p = all_probs[cat]; w = int(p*100)
            prob_bars += f'<div style="display:flex;align-items:center;gap:8px;margin:6px 0;"><span style="font-family:Outfit;font-size:12px;color:#7A8599;width:65px;">{c.get("label",cat.upper())}</span><div style="flex:1;height:4px;background:#1A2035;border-radius:2px;overflow:hidden;"><div style="width:{w}%;height:100%;background:{c["dot"]};border-radius:2px;"></div></div><span style="font-family:IBM Plex Mono;font-size:12px;color:{c["dot"]};width:40px;text-align:right;">{p*100:.0f}%</span></div>'
        prob_bars = f'<div style="margin-top:16px;padding-top:12px;border-top:1px solid #1A203566;">{prob_bars}</div>'
    lbl = f'<div style="font-family:IBM Plex Mono;font-size:10px;color:#64748B;letter-spacing:1.5px;margin-bottom:14px;">{label}</div>' if label else ""
    st.markdown(f"""<div style="background:{cfg['dot']}08;border:1px solid {cfg['dot']}18;border-radius:14px;padding:24px;margin:12px 0;">{lbl}
    <div style="font-family:Outfit;font-size:36px;font-weight:900;color:{cfg['dot']};letter-spacing:-0.03em;line-height:1;">{cfg['label']}</div>
    <div style="display:flex;align-items:baseline;gap:8px;margin-top:6px;margin-bottom:16px;">
        <span style="font-family:IBM Plex Mono;font-size:13px;color:#94A3B8;">{pct}% confidence</span>
        <span style="font-family:Source Sans 3;font-size:13px;color:#64748B;">·</span>
        <span style="font-family:Source Sans 3;font-size:13px;color:{cfg['dot']}99;">{cfg['tag']}</span>
    </div>
    <div style="height:3px;background:{cfg['dot']}15;border-radius:2px;overflow:hidden;margin-bottom:16px;"><div style="width:{pct}%;height:100%;background:{cfg['dot']};border-radius:2px;"></div></div>
    <div style="font-family:Source Sans 3;font-size:14px;color:#C9D1D9;line-height:1.6;">{cfg['desc']}</div>{prob_bars}</div>""", unsafe_allow_html=True)

def render_cost(low, high, label, brand, model):
    mid = (low+high)/2
    st.markdown(f"""<div style="margin:16px 0;padding:20px;background:rgba(248,113,113,0.06);border:1px solid rgba(248,113,113,0.12);border-radius:14px;">
    <div style="font-family:Outfit;font-size:11px;font-weight:500;color:#F87171;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;">🔧 {label}</div>
    <div style="display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;">
        <span style="font-family:IBM Plex Mono;font-size:32px;font-weight:700;color:#F0F2F5;">€{mid:.0f}</span>
        <span style="font-family:IBM Plex Mono;font-size:13px;color:#7A8599;">€{low}–€{high} range</span>
    </div>
    <div style="font-family:Source Sans 3;font-size:12px;color:#64748B;margin-top:6px;">{brand} {model} · Third-party repair, EU average</div>
    </div>""", unsafe_allow_html=True)

def render_angle_dot(cls, name):
    cfg = CLASS_CONFIG[cls]
    st.markdown(f'<div style="display:flex;align-items:center;gap:6px;padding:3px 0;"><div style="width:6px;height:6px;border-radius:50%;background:{cfg["dot"]};"></div><span style="font-family:IBM Plex Mono;font-size:10px;color:#7A8599;">{name}</span><span style="font-family:IBM Plex Mono;font-size:10px;color:{cfg["dot"]};">{cfg["label"]}</span></div>', unsafe_allow_html=True)

def render_chat_msg(role, content):
    if role == "user":
        st.markdown(f'<div style="background:#111827;border-radius:16px;padding:14px 16px;margin:8px 0;"><div style="font-size:11px;color:#6366F1;font-weight:600;margin-bottom:6px;">You</div><div style="color:#F0F2F5;font-size:14px;line-height:1.6;">{content}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background:#0C1018;border-left:3px solid #6366F1;border-radius:16px;padding:14px 16px;margin:8px 0;"><div style="font-size:11px;color:#34D399;font-weight:600;margin-bottom:6px;">PhoneCheck AI</div><div style="color:#C9D1D9;font-size:14px;line-height:1.6;">{content}</div></div>', unsafe_allow_html=True)

# ============================================================
# OLLAMA
# ============================================================
def ollama_available():
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if r.status_code != 200: return False
        models = [m.get("name","") for m in r.json().get("models",[])]
        base = VISION_MODEL.split(":")[0]
        return any(VISION_MODEL in n or base in n for n in models)
    except: return False

def ollama_chat(system, text, image_b64=None, history=None, max_tokens=400):
    msgs = [{"role":"system","content":system}]
    if history: msgs.extend(history)
    um = {"role":"user","content":text}
    if image_b64: um["images"] = [image_b64]
    msgs.append(um)
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json={"model":VISION_MODEL,"messages":msgs,"stream":False,"options":{"temperature":0.3,"num_predict":max_tokens,"num_gpu":99}}, timeout=120)
        r.raise_for_status()
        content = r.json()["message"]["content"]
        # Strip Qwen3 thinking tags — model wraps reasoning in <think>...</think>
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        return content
    except requests.exceptions.Timeout: return "Request timed out."
    except requests.exceptions.ConnectionError: return "Cannot connect to Ollama."
    except Exception as e: return f"Error: {str(e)}"

def generate_analysis(image, pred_class, confidence, all_probs, area_type, brand=None, phone_model=None, repair_low=None, repair_high=None):
    """LLM adds specific damage details the CNN can't provide."""
    img_b64 = image_to_base64(image)
    area = "front screen" if area_type == "front" else "back glass"
    phone = f"{brand} {phone_model}" if brand and phone_model else "the phone"
    pct = int(confidence * 100)

    if pred_class == "damaged":
        # Ask LLM ONLY for specific damage location and type
        damage_detail = ollama_chat(
            "/no_think You are a phone repair technician. ONLY describe damage. Never mention phone brand, color, shape, or that someone is holding it. Never say 'appears to be' or 'modern smartphone'.",
            f"This {area} is cracked. Answer in exactly 2 short sentences:\n"
            f"1. Where is the damage? (exact location like top-left corner, center, along right edge)\n"
            f"2. What type? (spider-web crack, single line, shattered area, chip, hairline fracture)",
            image_b64=img_b64, max_tokens=300
        )
        # Filter out filler — strip sentences, don't discard everything
        filler_phrases = ['appears to be', 'being held', 'camera system', 'modern smartphone', 'silver', 'the image shows', 'i can see a', 'this is a photo']
        lines = [l.strip() for l in damage_detail.strip().split('\n') if l.strip()]
        lines = [l for l in lines if not all(f in l.lower() for f in ['smartphone']) and len(l) > 10]
        # Remove filler phrases from within sentences
        cleaned_lines = []
        for l in lines:
            skip = False
            for fp in filler_phrases:
                if l.lower().startswith(fp):
                    skip = True
                    break
            if not skip:
                cleaned_lines.append(l)
        damage_detail = ' '.join(cleaned_lines[:2]) if cleaned_lines else "Crack damage detected on the surface."

        repair_name = "Screen replacement" if area_type == "front" else "Back glass replacement"
        cost_line = f"\n**Repair:** {repair_name} — €{repair_low}–€{repair_high}" if repair_low and repair_high else f"\n**Repair:** {repair_name}"

        if confidence > 0.85:
            sev_emoji, severity = "🔴", "Severe"
            advice = "Repair before selling. Avoid using without protection."
        elif confidence > 0.7:
            sev_emoji, severity = "🟠", "Moderate"
            advice = "Still usable but repair within 1–2 weeks."
        else:
            sev_emoji, severity = "🟡", "Minor"
            advice = "A screen protector can prevent it from spreading."

        return f"""{sev_emoji} **{severity} Damage** — {area.title()}

{damage_detail}
{cost_line}

{advice}"""

    elif pred_class == "used":
        # Ask LLM ONLY about specific scratches/marks
        wear_detail = ollama_chat(
            "/no_think You are inspecting a phone for resale grading. Describe ONLY the visible scratches, scuffs, or wear marks you see — their exact location and severity. Do not describe the phone itself. Do not mention brand, color, camera module, buttons, or who is holding it. Be specific about what you actually see. Max 2 sentences.",
            f"Look closely at this {area}. It has cosmetic wear but no cracks. What specific scratches or marks can you see and exactly where are they located?",
            image_b64=img_b64, max_tokens=300
        )
        # Strip filler phrases instead of replacing entire response
        filler_phrases = ['appears to be', 'being held', 'camera system', 'modern smartphone', 'the image shows a', 'i can see a phone', 'this is a photo of', 'the phone is a']
        cleaned = wear_detail.strip()
        for fp in filler_phrases:
            cleaned = cleaned.lower().replace(fp, '')
        # Only fall back to template if response is essentially empty after cleaning
        cleaned_check = ''.join(c for c in cleaned if c.isalpha())
        if len(cleaned_check) < 15:
            wear_detail = "Light surface scratches from normal daily use."
        else:
            # Use original but take first 2 meaningful sentences
            sentences = [s.strip() for s in wear_detail.replace('\n', ' ').split('.') if len(s.strip()) > 10]
            sentences = [s for s in sentences if not any(f in s.lower() for f in filler_phrases)]
            wear_detail = '. '.join(sentences[:2]).strip()
            if wear_detail and not wear_detail.endswith('.'):
                wear_detail += '.'
            if len(wear_detail) < 15:
                wear_detail = "Light surface scratches from normal daily use."

        return f"""⚡ **Used Condition** — {area.title()}

{wear_detail}

**Sell as:** "Used — Good Condition" at 15–20% below mint price."""

    else:
        # Mint — no LLM needed
        return f"""✅ **Mint Condition** — {area.title()}

No scratches, cracks, or wear detected. Sell at full price as "Mint / Like New"."""

def generate_edge_analysis(edge_images_labels, used_edges, brand=None, phone_model=None, repair_low=None, repair_high=None):
    if not used_edges:
        return "✅ **All edges mint** — no dents, scuffs, or scratches on the frame."

    # Ask LLM about the worst edge
    worst_img = None
    for label, img in edge_images_labels:
        if label in used_edges: worst_img = img; break

    desc = ""
    if worst_img:
        desc = ollama_chat(
            "/no_think You inspect phone edges for resale. Describe ONLY the specific scuffs, dents, or scratches you see on the metal or plastic frame — their exact location and how deep they look. Do not describe the phone itself. One or two sentences max.",
            "Look closely at this phone edge. What specific wear marks, dents, or scratches can you see and exactly where are they?",
            image_b64=image_to_base64(worst_img), max_tokens=300
        )
        filler_phrases = ['appears to be', 'smartphone', 'modern', 'camera', 'the image shows', 'i can see a phone', 'this is a photo']
        sentences = [s.strip() for s in desc.replace('\n', ' ').split('.') if len(s.strip()) > 10]
        sentences = [s for s in sentences if not any(f in s.lower() for f in filler_phrases)]
        desc = '. '.join(sentences[:2]).strip()
        if desc and not desc.endswith('.'):
            desc += '.'
        if len(desc) < 15:
            desc = "Visible scuffs and scratches on the frame from daily use."

    edge_list = ", ".join(used_edges)
    if len(used_edges) >= 2:
        verdict = f"⚡ **{len(used_edges)} edges show wear** — graded as USED overall."
    else:
        verdict = f"⚡ **{edge_list} edge** shows minor wear."

    cost = f"Frame polishing: €{repair_low}–€{repair_high}" if repair_low else "Frame polishing: €20–60"

    return f"""{verdict}

{desc}

{cost}"""

def generate_chat(user_msg, context, history, img_b64=None):
    sys_prompt = (
        f"/no_think You are PhoneCheck AI, a friendly phone condition assistant. "
        f"You scanned a device with these results: {context} "
        f"Answer questions about the phone's condition, repair options, selling price, insurance, and warranty. "
        f"Use simple everyday language — the user is not technical. Be helpful and concise (max 80 words). "
        f"Use euro for currency. If you're not sure about something, say so honestly."
    )
    return ollama_chat(sys_prompt, user_msg, image_b64=img_b64 if not history else None, history=history, max_tokens=200)

# ============================================================
# OVERALL VERDICT
# ============================================================
def render_overall_verdict(front_v, back_v, edge_results, brand, phone_model):
    f_cls, b_cls = front_v[0], back_v[0]
    used_edges = [n for n, c, _ in edge_results if c == "used"]
    edge_override = len(used_edges) >= 2
    base = f_cls if GRADE_ORDER[f_cls] <= GRADE_ORDER[b_cls] else b_cls
    overall = "used" if edge_override and GRADE_ORDER[base] > GRADE_ORDER["used"] else base
    cfg = CLASS_CONFIG[overall]
    total_low, total_high, repairs = 0, 0, []
    if f_cls == "damaged" and phone_model:
        sl, sh = get_repair_estimate(brand, phone_model, "screen")
        if sl: total_low += sl; total_high += sh; repairs.append(f"Screen €{sl}–€{sh}")
    if b_cls == "damaged" and phone_model:
        bl, bh = get_repair_estimate(brand, phone_model, "back")
        if bl: total_low += bl; total_high += bh; repairs.append(f"Back €{bl}–€{bh}")
    if used_edges and phone_model:
        el, eh = get_edge_repair_estimate(brand, phone_model)
        if el: total_low += el; total_high += eh; repairs.append(f"Edge €{el}–€{eh}")

    if overall == "mint":
        verdict = "Your phone is in excellent condition. No damage anywhere."
        action = "Sell at full price."
        action_emoji = "💰"
        bg_tint = "rgba(52,211,153,0.05)"
        border_tint = "rgba(52,211,153,0.15)"
    elif overall == "used":
        verdict = "Your phone works perfectly but has some cosmetic wear."
        action = "Price it 15–20% below mint to sell quickly."
        action_emoji = "📦"
        bg_tint = "rgba(251,191,36,0.04)"
        border_tint = "rgba(251,191,36,0.12)"
    else:
        if f_cls == "damaged" and b_cls == "damaged":
            verdict = "Both the screen and back are damaged."
        elif f_cls == "damaged":
            verdict = "The screen is cracked. The back is " + ("fine" if b_cls == "mint" else "showing some wear") + "."
        else:
            verdict = "The back glass is cracked. The screen is " + ("fine" if f_cls == "mint" else "showing some wear") + "."
        action = "Repair before selling." + (f" Total cost: €{total_low}–€{total_high}." if total_low else "")
        action_emoji = "🔧"
        bg_tint = "rgba(248,113,113,0.04)"
        border_tint = "rgba(248,113,113,0.12)"
    if edge_override:
        verdict += f" ({len(used_edges)} worn edges also detected.)"

    f_cfg, b_cfg = CLASS_CONFIG[f_cls], CLASS_CONFIG[b_cls]
    e_color = "#FBBF24" if used_edges else "#34D399"
    e_label = f"{len(used_edges)} worn" if used_edges else "Mint"

    # Repair estimate
    repair_html = ""
    if repairs and total_low:
        mid = (total_low + total_high) / 2
        breakdown = "  ·  ".join(repairs)
        repair_html = f"""<div style="margin-top:24px;padding:20px;background:rgba(248,113,113,0.06);border-radius:12px;">
            <div style="font-family:Outfit;font-size:11px;font-weight:500;color:#F87171;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;">Repair Estimate</div>
            <div style="display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;">
                <span style="font-family:IBM Plex Mono;font-size:36px;font-weight:700;color:#F0F2F5;">€{mid:.0f}</span>
                <span style="font-family:IBM Plex Mono;font-size:13px;color:#7A8599;">€{total_low}–€{total_high} range</span>
            </div>
            <div style="font-family:Source Sans 3;font-size:12px;color:#7A8599;margin-top:6px;">{breakdown}</div>
        </div>"""

    st.markdown(f"""<div style="background:{bg_tint};border:1px solid {border_tint};border-radius:16px;padding:32px 28px;margin:8px 0;">
    <div style="font-family:Outfit;font-size:56px;font-weight:900;color:{cfg['dot']};letter-spacing:-0.04em;line-height:1;margin-bottom:4px;">{cfg['label']}</div>
    <div style="font-family:Outfit;font-size:14px;font-weight:500;color:{cfg['dot']}88;margin-bottom:20px;">{cfg['tag']}</div>
    <div style="font-family:Source Sans 3;font-size:16px;color:#C9D1D9;line-height:1.7;max-width:480px;">{verdict}</div>
    <div style="font-family:Outfit;font-size:15px;color:#F0F2F5;margin-top:10px;font-weight:600;">{action_emoji} {action}</div>
    <div style="display:flex;gap:28px;margin-top:24px;padding-top:20px;border-top:1px solid {border_tint};">
        <div>
            <div style="font-family:Source Sans 3;font-size:11px;color:#64748B;margin-bottom:4px;">Front</div>
            <div style="display:flex;align-items:center;gap:6px;">
                <div style="width:8px;height:8px;border-radius:50%;background:{f_cfg['dot']};"></div>
                <span style="font-family:Outfit;font-size:14px;font-weight:700;color:{f_cfg['dot']};">{f_cfg['label']}</span>
            </div>
        </div>
        <div>
            <div style="font-family:Source Sans 3;font-size:11px;color:#64748B;margin-bottom:4px;">Back</div>
            <div style="display:flex;align-items:center;gap:6px;">
                <div style="width:8px;height:8px;border-radius:50%;background:{b_cfg['dot']};"></div>
                <span style="font-family:Outfit;font-size:14px;font-weight:700;color:{b_cfg['dot']};">{b_cfg['label']}</span>
            </div>
        </div>
        <div>
            <div style="font-family:Source Sans 3;font-size:11px;color:#64748B;margin-bottom:4px;">Edges</div>
            <div style="display:flex;align-items:center;gap:6px;">
                <div style="width:8px;height:8px;border-radius:50%;background:{e_color};"></div>
                <span style="font-family:Outfit;font-size:14px;font-weight:700;color:{e_color};">{e_label}</span>
            </div>
        </div>
    </div>
    {repair_html}
    </div>""", unsafe_allow_html=True)
    return {'overall': overall, 'total_low': total_low, 'total_high': total_high, 'used_edges': used_edges, 'edge_override': edge_override}

# ============================================================
# MAIN — Mobile guided wizard
# ============================================================

STEPS_QUICK_CHECK = [
    {"key": "cf1", "num": 1,  "title": "Front Screen", "guide": "flat_front",
     "instruction": "Place the phone face-up on a flat surface.",
     "detail": "Camera directly above, entire screen visible and well-lit."},
    {"key": "cb1", "num": 2,  "title": "Back Glass", "guide": "flat_back",
     "instruction": "Flip the phone over. Place it back-up on a flat surface.",
     "detail": "Camera directly above, entire back panel visible."},
]

STEPS_QUICK = [
    {"key": "cf1", "num": 1,  "title": "Front Screen", "guide": "flat_front",
     "instruction": "Place the phone face-up on a flat surface.",
     "detail": "Camera directly above, entire screen visible and well-lit."},
    {"key": "cb1", "num": 2,  "title": "Back Glass", "guide": "flat_back",
     "instruction": "Flip the phone over. Place it back-up on a flat surface.",
     "detail": "Camera directly above, entire back panel visible."},
    {"key": "cel", "num": 3,  "title": "Left Edge", "guide": "edge_left",
     "instruction": "Hold the phone upright and photograph the left side.",
     "detail": "Show the full left edge. Look for dents, scuffs, or scratches."},
    {"key": "cer", "num": 4,  "title": "Right Edge", "guide": "edge_right",
     "instruction": "Now photograph the right side edge.",
     "detail": "Include the power button area."},
    {"key": "cet", "num": 5,  "title": "Top Edge", "guide": "edge_top",
     "instruction": "Photograph the top edge of the phone.",
     "detail": "Show the full top from left to right."},
    {"key": "ceb", "num": 6,  "title": "Bottom Edge", "guide": "edge_bottom",
     "instruction": "Finally, photograph the bottom edge (charging port side).",
     "detail": "Include the charging port area."},
]

STEPS_ACCURATE = [
    {"key": "cf1", "num": 1,  "title": "Front Screen — Flat", "guide": "flat_front",
     "instruction": "Place the phone face-up on a flat surface.",
     "detail": "Hold your camera directly above the phone, looking straight down. Make sure the entire screen is visible and well-lit."},
    {"key": "cf2", "num": 2,  "title": "Front Screen — Tilted 45\u00b0", "guide": "tilt45_front",
     "instruction": "Pick up the phone and tilt it ~45\u00b0 toward you.",
     "detail": "This angle catches glare on hairline cracks that aren't visible from above. Keep the screen in frame."},
    {"key": "cb1", "num": 3,  "title": "Back Glass — Flat", "guide": "flat_back",
     "instruction": "Flip the phone over. Place it back-up on a flat surface.",
     "detail": "Camera directly above, entire back panel visible. Check for cracks near the camera module."},
    {"key": "cb2", "num": 4,  "title": "Back Glass — Tilted 45\u00b0", "guide": "tilt45_back",
     "instruction": "Pick up and tilt ~45\u00b0 to catch back glass cracks.",
     "detail": "Back glass cracks often show as spider webs near corners. The tilt helps light catch them."},
    {"key": "cel", "num": 5,  "title": "Left Edge", "guide": "edge_left",
     "instruction": "Hold the phone upright and photograph the left side.",
     "detail": "Show the full left edge from top to bottom. Look for dents, scuffs, or scratches on the metal/plastic frame."},
    {"key": "cer", "num": 6,  "title": "Right Edge", "guide": "edge_right",
     "instruction": "Now photograph the right side edge.",
     "detail": "Include the power button area. Frame scratches and dents affect the 'used' vs 'mint' grade."},
    {"key": "cet", "num": 7,  "title": "Top Edge", "guide": "edge_top",
     "instruction": "Photograph the top edge of the phone.",
     "detail": "Show the full top from left to right. This area often has dents from drops."},
    {"key": "ceb", "num": 8, "title": "Bottom Edge", "guide": "edge_bottom",
     "instruction": "Finally, photograph the bottom edge (charging port side).",
     "detail": "Include the charging port area. Scuffs here are very common and affect grading."},
]

def render_positioning_guide(guide_type):
    """Animated 3D phone guides — tall proportions, high contrast, clear instructions."""
    a = "#6366F1"
    a2 = "#818CF8"
    d = "#475569"
    d2 = "#1E293B"
    t = "#94A3B8"
    uid = guide_type.replace("_","")

    if "flat" in guide_type:
        is_front = "front" in guide_type
        c = "#6366F1" if is_front else "#34D399"
        cl = "#A5B4FC" if is_front else "#6EE7B7"
        if is_front:
            detail = (
                f'<rect x="108" y="115" width="84" height="40" rx="2" fill="{c}" opacity="0.08"/>'
                f'<rect x="130" y="112" width="40" height="4" rx="2" fill="{cl}" opacity="0.45"/>'
                f'<rect x="138" y="150" width="24" height="2" rx="1" fill="{cl}" opacity="0.3"/>'
            )
        else:
            detail = (
                f'<rect x="108" y="114" width="24" height="24" rx="6" fill="{d2}" stroke="{cl}" stroke-width="1.5" opacity="0.6"/>'
                f'<circle cx="115" cy="121" r="5" fill="none" stroke="{cl}" stroke-width="1.5" opacity="0.55"/>'
                f'<circle cx="115" cy="121" r="2" fill="{cl}" opacity="0.25"/>'
                f'<circle cx="127" cy="121" r="5" fill="none" stroke="{cl}" stroke-width="1.5" opacity="0.55"/>'
                f'<circle cx="127" cy="121" r="2" fill="{cl}" opacity="0.25"/>'
                f'<circle cx="121" cy="133" r="3.5" fill="none" stroke="{cl}" stroke-width="1" opacity="0.4"/>'
                f'<circle cx="129" cy="134" r="1.8" fill="{cl}" opacity="0.4"/>'
            )
        lbl = "Screen facing up" if is_front else "Back panel facing up"
        svg = f"""<div style="text-align:center;padding:4px 0;"><svg viewBox="0 0 300 190" xmlns="http://www.w3.org/2000/svg" style="max-width:280px;">
<style>@keyframes fl{uid}{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-4px)}}}}.pf{uid}{{animation:fl{uid} 3s ease-in-out infinite}}</style>
<defs><linearGradient id="gf{uid}" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="{c}" stop-opacity="0.25"/><stop offset="100%" stop-color="{c}" stop-opacity="0.06"/></linearGradient></defs>
<line x1="40" y1="162" x2="260" y2="162" stroke="{d2}" stroke-width="1"/>
<g class="pf{uid}">
<ellipse cx="150" cy="161" rx="58" ry="5" fill="{c}" opacity="0.06"/>
<path d="M 85,140 L 215,140 L 232,110 L 102,110 Z" fill="url(#gf{uid})" stroke="{c}" stroke-width="1.8" stroke-linejoin="round"/>
<path d="M 85,140 L 85,148 L 215,148 L 215,140" fill="{d2}" fill-opacity="0.5" stroke="{d}" stroke-width="0.8" opacity="0.6"/>
<path d="M 215,140 L 215,148 L 232,118 L 232,110" fill="{d2}" fill-opacity="0.4" stroke="{d}" stroke-width="0.8" opacity="0.6"/>
{detail}
</g>
<g opacity="0.55"><line x1="150" y1="25" x2="150" y2="95" stroke="{t}" stroke-width="1" stroke-dasharray="3,4"/><polygon points="146,95 154,95 150,103" fill="{t}"/><circle cx="150" cy="18" r="8" fill="none" stroke="{t}" stroke-width="1"/><circle cx="150" cy="18" r="3" fill="{t}"/></g>
<text x="248" y="108" fill="{c}" font-size="16" font-weight="700" font-family="Outfit">0&#176;</text>
<text x="150" y="182" fill="{t}" font-size="10" text-anchor="middle" font-family="Source Sans 3">{lbl}</text>
</svg></div>"""

    elif "tilt45" in guide_type:
        is_front = "front" in guide_type
        c = "#6366F1" if is_front else "#34D399"
        cl = "#A5B4FC" if is_front else "#6EE7B7"
        if is_front:
            detail = (
                f'<rect x="108" y="115" width="84" height="40" rx="2" fill="{c}" opacity="0.08"/>'
                f'<rect x="130" y="112" width="40" height="4" rx="2" fill="{cl}" opacity="0.45"/>'
                f'<rect x="138" y="150" width="24" height="2" rx="1" fill="{cl}" opacity="0.3"/>'
            )
        else:
            detail = (
                f'<rect x="108" y="114" width="24" height="24" rx="6" fill="{d2}" stroke="{cl}" stroke-width="1.5" opacity="0.6"/>'
                f'<circle cx="115" cy="121" r="5" fill="none" stroke="{cl}" stroke-width="1.5" opacity="0.55"/>'
                f'<circle cx="115" cy="121" r="2" fill="{cl}" opacity="0.25"/>'
                f'<circle cx="127" cy="121" r="5" fill="none" stroke="{cl}" stroke-width="1.5" opacity="0.55"/>'
                f'<circle cx="127" cy="121" r="2" fill="{cl}" opacity="0.25"/>'
                f'<circle cx="121" cy="133" r="3.5" fill="none" stroke="{cl}" stroke-width="1" opacity="0.4"/>'
                f'<circle cx="129" cy="134" r="1.8" fill="{cl}" opacity="0.4"/>'
            )
        lbl = "Tilt bottom toward camera"
        svg = f"""<div style="text-align:center;padding:4px 0;"><svg viewBox="0 0 300 210" xmlns="http://www.w3.org/2000/svg" style="max-width:280px;">
<style>@keyframes fl{uid}{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-4px)}}}}.pf{uid}{{animation:fl{uid} 3s ease-in-out infinite}}</style>
<defs><linearGradient id="gt{uid}" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="{c}" stop-opacity="0.25"/><stop offset="100%" stop-color="{c}" stop-opacity="0.06"/></linearGradient></defs>
<line x1="40" y1="182" x2="260" y2="182" stroke="{d2}" stroke-width="1"/>
<g class="pf{uid}">
<ellipse cx="150" cy="181" rx="58" ry="5" fill="{c}" opacity="0.06"/>
<g transform="rotate(-25, 150, 148)">
<path d="M 85,140 L 215,140 L 232,110 L 102,110 Z" fill="url(#gt{uid})" stroke="{c}" stroke-width="1.8" stroke-linejoin="round"/>
<path d="M 85,140 L 85,148 L 215,148 L 215,140" fill="{d2}" fill-opacity="0.5" stroke="{d}" stroke-width="0.8" opacity="0.6"/>
<path d="M 215,140 L 215,148 L 232,118 L 232,110" fill="{d2}" fill-opacity="0.4" stroke="{d}" stroke-width="0.8" opacity="0.6"/>
{detail}
</g>
</g>
<path d="M 232,170 A 50,50 0 0,0 218,95" fill="none" stroke="#FBBF24" stroke-width="2" opacity="0.75"/>
<text x="240" y="135" fill="#FBBF24" font-size="16" font-weight="700" font-family="Outfit">45&#176;</text>
<g opacity="0.55" transform="rotate(-25, 150, 88)"><line x1="150" y1="15" x2="150" y2="80" stroke="{t}" stroke-width="1" stroke-dasharray="3,4"/><polygon points="146,80 154,80 150,88" fill="{t}"/><circle cx="150" cy="8" r="8" fill="none" stroke="{t}" stroke-width="1"/><circle cx="150" cy="8" r="3" fill="{t}"/></g>
<text x="150" y="202" fill="{t}" font-size="10" text-anchor="middle" font-family="Source Sans 3">{lbl}</text>
</svg></div>"""

    elif "tilt60" in guide_type:
        c = "#6366F1" if "front" in guide_type else "#34D399"
        svg = f"""<div style="text-align:center;padding:4px 0;"><svg viewBox="0 0 300 210" xmlns="http://www.w3.org/2000/svg" style="max-width:280px;">
<style>@keyframes s6{uid}{{0%,100%{{transform:rotate(0deg)}}50%{{transform:rotate(1.5deg)}}}}.p6{uid}{{animation:s6{uid} 4s ease-in-out infinite;transform-origin:130px 180px}}</style>
<line x1="40" y1="180" x2="260" y2="180" stroke="{d2}" stroke-width="1"/>
<g class="p6{uid}">
<path d="M 115,175 L 160,175 L 148,42 L 103,42 Z" fill="{c}" fill-opacity="0.15" stroke="{c}" stroke-width="1.8" stroke-linejoin="round"/>
<path d="M 115,175 L 119,178 L 164,178 L 160,175" fill="{d2}" fill-opacity="0.4" stroke="{d}" stroke-width="0.7" opacity="0.55"/>
<path d="M 160,175 L 164,178 L 152,45 L 148,42" fill="{d2}" fill-opacity="0.3" stroke="{d}" stroke-width="0.7" opacity="0.55"/>
</g>
<path d="M 168,173 A 60,60 0 0,1 152,62" fill="none" stroke="#F87171" stroke-width="2" opacity="0.8"/>
<text x="178" y="120" fill="#F87171" font-size="17" font-weight="700" font-family="Outfit">60&#176;</text>
<g opacity="0.5"><line x1="92" y1="8" x2="108" y2="32" stroke="{t}" stroke-width="1" stroke-dasharray="3,4"/><polygon points="104,30 112,30 109,38" fill="{t}"/><circle cx="89" cy="4" r="8" fill="none" stroke="{t}" stroke-width="1"/><circle cx="89" cy="4" r="3" fill="{t}"/></g>
<text x="148" y="200" fill="{t}" font-size="10" text-anchor="middle" font-family="Source Sans 3">Almost side-on view</text>
</svg></div>"""

    elif "edge" in guide_type:
        labels = {"edge_left": "LEFT", "edge_right": "RIGHT", "edge_top": "TOP", "edge_bottom": "BOTTOM"}
        label = labels.get(guide_type, "EDGE")
        hl = glow = arrow = ""
        px, py, pw, ph = 82, 18, 76, 164
        if "left" in guide_type:
            hl = f'<line x1="{px}" y1="{py+8}" x2="{px}" y2="{py+ph-8}" stroke="{a}" stroke-width="4.5" stroke-linecap="round"/>'
            glow = f'<line x1="{px}" y1="{py+8}" x2="{px}" y2="{py+ph-8}" stroke="{a2}" stroke-width="14" opacity="0.08" class="gp{uid}"/>'
            arrow = f'<g class="an{uid}"><line x1="32" y1="100" x2="70" y2="100" stroke="{a2}" stroke-width="1.5"/><polygon points="70,96 70,104 80,100" fill="{a2}"/></g>'
        elif "right" in guide_type:
            hl = f'<line x1="{px+pw}" y1="{py+8}" x2="{px+pw}" y2="{py+ph-8}" stroke="{a}" stroke-width="4.5" stroke-linecap="round"/>'
            glow = f'<line x1="{px+pw}" y1="{py+8}" x2="{px+pw}" y2="{py+ph-8}" stroke="{a2}" stroke-width="14" opacity="0.08" class="gp{uid}"/>'
            arrow = f'<g class="an{uid}"><line x1="208" y1="100" x2="170" y2="100" stroke="{a2}" stroke-width="1.5"/><polygon points="170,96 170,104 160,100" fill="{a2}"/></g>'
        elif "top" in guide_type:
            hl = f'<line x1="{px+8}" y1="{py}" x2="{px+pw-8}" y2="{py}" stroke="{a}" stroke-width="4.5" stroke-linecap="round"/>'
            glow = f'<line x1="{px+8}" y1="{py}" x2="{px+pw-8}" y2="{py}" stroke="{a2}" stroke-width="14" opacity="0.08" class="gp{uid}"/>'
            arrow = f'<g class="an{uid}"><line x1="120" y1="-2" x2="120" y2="12" stroke="{a2}" stroke-width="1.5"/><polygon points="116,12 124,12 120,19" fill="{a2}"/></g>'
        elif "bottom" in guide_type:
            hl = f'<line x1="{px+8}" y1="{py+ph}" x2="{px+pw-8}" y2="{py+ph}" stroke="{a}" stroke-width="4.5" stroke-linecap="round"/>'
            glow = f'<line x1="{px+8}" y1="{py+ph}" x2="{px+pw-8}" y2="{py+ph}" stroke="{a2}" stroke-width="14" opacity="0.08" class="gp{uid}"/>'
            arrow = f'<g class="an{uid}"><line x1="120" y1="202" x2="120" y2="190" stroke="{a2}" stroke-width="1.5"/><polygon points="116,190 124,190 120,183" fill="{a2}"/></g>'
        nd = "4px,0" if ("left" in guide_type or "right" in guide_type) else "0,4px"
        phone = (
            f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="14" fill="{a}" fill-opacity="0.04" stroke="{d}" stroke-width="2"/>'
            f'<rect x="{px+16}" y="{py+5}" width="{pw-32}" height="6" rx="3" fill="{d}" opacity="0.65"/>'
            f'<circle cx="120" cy="{py+8}" r="2.2" fill="{d}" opacity="0.45"/>'
            f'<line x1="{px+5}" y1="{py+17}" x2="{px+pw-5}" y2="{py+17}" stroke="{d}" stroke-width="0.7" opacity="0.3"/>'
            f'<line x1="{px+5}" y1="{py+ph-20}" x2="{px+pw-5}" y2="{py+ph-20}" stroke="{d}" stroke-width="0.7" opacity="0.25"/>'
            f'<rect x="{px+22}" y="{py+ph-14}" width="{pw-44}" height="3" rx="1.5" fill="{d}" opacity="0.4"/>'
            f'<line x1="{px+pw}" y1="{py+42}" x2="{px+pw}" y2="{py+55}" stroke="{d}" stroke-width="3" opacity="0.2"/>'
            f'<line x1="{px}" y1="{py+48}" x2="{px}" y2="{py+58}" stroke="{d}" stroke-width="3" opacity="0.15"/>'
            f'<line x1="{px}" y1="{py+63}" x2="{px}" y2="{py+73}" stroke="{d}" stroke-width="3" opacity="0.15"/>'
        )
        svg = f"""<div style="text-align:center;padding:4px 0;"><svg viewBox="0 0 240 205" xmlns="http://www.w3.org/2000/svg" style="max-width:200px;">
<style>@keyframes gp{uid}{{0%,100%{{opacity:0.04}}50%{{opacity:0.22}}}}@keyframes an{uid}{{0%,100%{{transform:translate(0,0)}}50%{{transform:translate({nd})}}}}.gp{uid}{{animation:gp{uid} 2s ease-in-out infinite}}.an{uid}{{animation:an{uid} 1.5s ease-in-out infinite}}</style>
{phone}{glow}{hl}{arrow}
<text x="120" y="108" fill="{a}" font-size="13" font-weight="600" text-anchor="middle" font-family="Outfit">{label}</text>
</svg></div>"""
    else:
        svg = ""

    if svg:
        st.markdown(svg, unsafe_allow_html=True)



def inject_splash_and_transitions():
    """Splash types title, shrinks it, fades overlay to reveal real header."""
    components.html("""
    <script>
    (function(){
        var pd = window.parent.document;

        /* CSS for block transitions (always inject) */
        if(!pd.getElementById('pc-css')){
            var css = pd.createElement('style');
            css.id = 'pc-css';
            css.textContent = '@keyframes pcSlL{from{opacity:0;transform:translateX(-40px)}to{opacity:1;transform:translateX(0)}} @keyframes pcSlR{from{opacity:0;transform:translateX(40px)}to{opacity:1;transform:translateX(0)}} [data-testid="stVerticalBlock"]>div{animation:pcSlL 0.4s cubic-bezier(0.16,1,0.3,1) both} [data-testid="stVerticalBlock"]>div:nth-child(even){animation-name:pcSlR} [data-testid="stVerticalBlock"]>div:nth-child(1){animation-delay:0s} [data-testid="stVerticalBlock"]>div:nth-child(2){animation-delay:0.03s} [data-testid="stVerticalBlock"]>div:nth-child(3){animation-delay:0.06s} [data-testid="stVerticalBlock"]>div:nth-child(4){animation-delay:0.09s} [data-testid="stVerticalBlock"]>div:nth-child(5){animation-delay:0.12s} [data-testid="stVerticalBlock"]>div:nth-child(6){animation-delay:0.15s} [data-testid="stVerticalBlock"]>div:nth-child(n+7){animation-delay:0.18s} .stProgress>div>div{transition:width 0.6s cubic-bezier(0.16,1,0.3,1)!important}';
            pd.head.appendChild(css);
        }

        /* Scroll memory */
        if(!pd._pcScr){
            pd._pcScr = true;
            pd.addEventListener('click', function(){ sessionStorage.setItem('_sy', window.parent.scrollY); }, true);
        }
        var sy = sessionStorage.getItem('_sy');
        if(sy) setTimeout(function(){ window.parent.scrollTo(0, parseInt(sy)); }, 100);

        /* File uploader button label — replace Streamlit's default "Browse files"/"Upload"
           text with "Take Photo". MutationObserver re-runs after every Streamlit re-render
           because the DOM is wiped and rebuilt on state changes. */
        if(!pd._pcUpLbl){
            pd._pcUpLbl = true;
            function fixUploadButtons(){
                var dropzones = pd.querySelectorAll('[data-testid="stFileUploaderDropzone"]');
                dropzones.forEach(function(dz){
                    var btn = dz.querySelector('button');
                    if(btn && btn.getAttribute('data-pc-labeled') !== '1'){
                        // Wipe all child content (icon spans, label divs, etc.)
                        btn.innerHTML = '<span style="font-family:Outfit,sans-serif;font-weight:600;font-size:14px;letter-spacing:0.5px;color:#06090F;">Take Photo</span>';
                        btn.setAttribute('data-pc-labeled', '1');
                    }
                });
            }
            fixUploadButtons();
            // Observe DOM for future re-renders (Streamlit rebuilds on state change)
            var obs = new MutationObserver(function(mutations){
                fixUploadButtons();
            });
            obs.observe(pd.body, {childList: true, subtree: true});
        }

        /* Splash only on fresh load */
        if(pd._pcSplash) return;
        pd._pcSplash = true;

        var ov = pd.createElement('div');
        ov.id = 'pc-splash';
        ov.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;background:#06090F;z-index:99999;display:flex;flex-direction:column;align-items:center;justify-content:center;pointer-events:none';

        var tw = pd.createElement('div');
        tw.style.cssText = 'text-align:center';

        var ti = pd.createElement('div');
        ti.style.cssText = 'font-family:Outfit,sans-serif;font-size:clamp(36px,9vw,56px);font-weight:800;color:#F0F2F5;letter-spacing:-0.03em;white-space:nowrap;display:inline';
        ti.innerHTML = '<span id="pct1"></span><span id="pct2" style="font-weight:300;color:#6366F1"></span>';

        var su = pd.createElement('div');
        su.style.cssText = 'font-family:Source Sans 3,sans-serif;font-size:15px;color:#64748B;margin-top:10px;opacity:0;transition:opacity 1s ease-out';
        su.textContent = 'Smartphone condition assessment';

        var bm = pd.createElement('div');
        bm.style.cssText = 'position:absolute;left:-10%;top:50%;width:120%;height:2px;background:linear-gradient(90deg,transparent,#6366F1 20%,#A5B4FC 50%,#6366F1 80%,transparent);transform:scaleX(0);transform-origin:left;transition:transform 0.7s cubic-bezier(0.16,1,0.3,1);opacity:0.9';

        var gl = pd.createElement('div');
        gl.style.cssText = 'position:absolute;left:50%;top:50%;width:0;height:0;border-radius:50%;background:radial-gradient(circle,rgba(99,102,241,0.25),transparent 70%);transform:translate(-50%,-50%);transition:width 1.4s ease-out,height 1.4s ease-out,opacity 1.8s ease-out';

        var dt = pd.createElement('div');
        dt.style.cssText = 'margin-top:28px;display:flex;gap:8px;justify-content:center;transition:opacity 0.6s ease-out';
        for(var i=0;i<3;i++){var d=pd.createElement('div');d.style.cssText='width:6px;height:6px;border-radius:50%;background:#1E293B;transition:background 0.3s';dt.appendChild(d);}

        tw.style.position='relative';
        tw.appendChild(gl);tw.appendChild(ti);tw.appendChild(bm);tw.appendChild(su);
        ov.appendChild(tw);ov.appendChild(dt);pd.body.appendChild(ov);

        var t1=pd.getElementById('pct1'),t2=pd.getElementById('pct2');
        var dotEls=dt.children,dti=0;
        var dotT=setInterval(function(){for(var x=0;x<3;x++)dotEls[x].style.background=x===dti%3?'#6366F1':'#1E293B';dti++;},400);

        var w1='PhoneCheck',w2='AI',ci=0,ph=1;
        function typ(){
            if(ph===1){if(ci<w1.length){t1.textContent+=w1[ci];ci++;setTimeout(typ,65+Math.random()*25);}else{ci=0;ph=2;setTimeout(typ,200);}}
            else if(ph===2){if(ci<w2.length){t2.textContent+=w2[ci];ci++;setTimeout(typ,110);}else{setTimeout(done,400);}}
        }
        function done(){
            clearInterval(dotT);dt.style.opacity='0';
            su.style.opacity='1';
            bm.style.transform='scaleX(1)';
            gl.style.width='400px';gl.style.height='400px';gl.style.opacity='0.4';
            setTimeout(function(){bm.style.transition='opacity 0.6s';bm.style.opacity='0';gl.style.opacity='0';},1000);
            /* Shrink title toward top-left */
            setTimeout(function(){
                tw.style.transition='transform 2.5s cubic-bezier(0.22,0.9,0.36,1)';
                tw.style.transform='scale(0.5) translateY(-60vh)';
                tw.style.opacity='0.6';
                ti.style.transition='opacity 2s ease-out';
                su.style.transition='opacity 1.5s ease-out';
            },2000);
            /* Fade entire overlay out */
            setTimeout(function(){
                ov.style.transition='opacity 1.2s ease-out';
                ov.style.opacity='0';
                setTimeout(function(){if(ov.parentNode)ov.remove();},1300);
            },3500);
        }
        setTimeout(typ,500);
    })();
    </script>
    """, height=0)

def main():
    st.set_page_config(page_title="PhoneCheck AI", page_icon="📱", layout="centered", initial_sidebar_state="collapsed")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    inject_splash_and_transitions()
    for k in ['chat_history', 'report_front', 'report_back', 'report_edge']:
        if k not in st.session_state: st.session_state[k] = [] if k == 'chat_history' else None

    # Load models
    front_model, _ = load_model("front"); back_model, _ = load_model("back")
    edge_model = load_edge_model("merged")
    view_gate = load_view_gate()
    yolo_model = load_yolo_model()
    is_ollama = ollama_available()

    # Sidebar
    show_probs = st.sidebar.checkbox("Show probabilities", value=False)
    loaded = [n for n, m in [("FRONT",front_model),("BACK",back_model),("EDGE",edge_model),("GATE",view_gate),("YOLO",yolo_model)] if m]
    st.sidebar.markdown(f'<div style="font-family:IBM Plex Mono;font-size:10px;color:#4A5568;">{" · ".join(loaded) or "NO MODELS"}</div>', unsafe_allow_html=True)

    # ===== HEADER =====
    st.markdown("""<div style="padding:24px 0 8px 0;">
    <div style="font-family:Outfit;font-size:28px;font-weight:800;color:#F0F2F5;letter-spacing:-0.03em;line-height:1;">PhoneCheck<span style="font-weight:300;color:#6366F1;">AI</span></div>
    <div style="font-family:'Source Sans 3';font-size:13px;color:#64748B;margin-top:6px;">Smartphone condition assessment</div>
    </div>""", unsafe_allow_html=True)
    items = [("FRONT",front_model),("BACK",back_model),("EDGE",edge_model),("GATE",view_gate),("YOLO",yolo_model),("AI",is_ollama)]
    model_count = sum(1 for _, m in items if m)
    st.markdown(f'<div style="font-family:Source Sans 3;font-size:12px;color:#64748B;padding:4px 0 12px 0;border-bottom:1px solid #1A2035;">{model_count} models loaded · <span style="color:#6366F1;">v2.1</span></div>', unsafe_allow_html=True)

    if not front_model and not back_model:
        st.error("No models found. Run `python phone_classifier.py`"); return

    # ===== DEVICE SELECTION =====
    section_label("SELECT YOUR DEVICE")
    dc1, dc2 = st.columns(2)
    with dc1: brand = st.selectbox("Brand", ["--"]+get_brands(), label_visibility="collapsed", key="brand")
    with dc2:
        phone_model = None
        if brand != "--":
            phone_model = st.selectbox("Model", ["--"]+get_models(brand), label_visibility="collapsed", key="model")
            if phone_model == "--": phone_model = None
        else:
            st.selectbox("Model", ["Select brand first"], disabled=True, label_visibility="collapsed", key="model_d")

    if not phone_model:
        st.markdown("""<style>
            @keyframes glow-pulse {
                0%, 100% { box-shadow: 0 0 12px rgba(99,102,241,0.4), 0 0 24px rgba(99,102,241,0.15); border-color: #6366F1; }
                50% { box-shadow: 0 0 20px rgba(99,102,241,0.7), 0 0 40px rgba(99,102,241,0.25); border-color: #A5B4FC; }
            }
            [data-testid="stSelectbox"] > div > div {
                animation: glow-pulse 2s ease-in-out infinite;
                border-radius: 8px;
            }
            [data-testid="stSelectbox"] input { caret-color: transparent !important; cursor: pointer !important; }
        </style>""", unsafe_allow_html=True)
        st.markdown("""<div style="text-align:center;padding:40px 16px;background:#0C1018;border:1px solid #1A2035;border-radius:12px;margin-top:16px;">
            <div style="width:48px;height:48px;border-radius:12px;background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.2);margin:0 auto 16px auto;display:flex;align-items:center;justify-content:center;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z" fill="#6366F1"/></svg>
            </div>
            <div style="font-family:Outfit;font-size:16px;font-weight:600;color:#F0F2F5;">Select your device to begin</div>
            <div style="font-family:Source Sans 3;font-size:13px;color:#7A8599;margin-top:8px;">Choose a brand and model above so we can estimate repair costs.</div>
        </div>""", unsafe_allow_html=True)
        return

    # ===== SCAN MODE =====
    section_label("SCAN MODE")
    cur_mode = st.session_state.get('_scan_mode', 'standard')

    MODE_DATA = [
        {"id": "quick_check", "name": "Quick", "sub": "2 images"},
        {"id": "standard",    "name": "Standard", "sub": "6 images"},
        {"id": "accurate",    "name": "Accurate", "sub": "8 images"},
    ]
    MODE_IDS = [m["id"] for m in MODE_DATA]

    # Animated segmented pill selector with sliding indicator — bigger pills
    active_idx = MODE_IDS.index(cur_mode) if cur_mode in MODE_IDS else 1
    indicator_left = f"calc({active_idx} * (100% / 3) + 3px)"
    indicator_width = "calc(100% / 3 - 4px)"

    seg_html = '<style>@keyframes pcFadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}</style>'
    seg_html += '<div style="animation:pcFadeUp 0.5s cubic-bezier(0.16,1,0.3,1) both;">'
    seg_html += f'<div id="pc-mode-track" style="position:relative;display:flex;background:#0C1018;border:1px solid #1A2035;border-radius:12px;padding:4px;">'
    # Sliding indicator
    seg_html += f'<div id="pc-mode-ind" style="position:absolute;top:4px;left:{indicator_left};width:{indicator_width};height:calc(100% - 8px);background:#6366F1;border-radius:10px;transition:left 0.3s cubic-bezier(0.16,1,0.3,1);z-index:0;pointer-events:none;"></div>'

    for i, m in enumerate(MODE_DATA):
        is_active = i == active_idx
        name_color = "#F0F2F5" if is_active else "#64748B"
        sub_color = "rgba(240,242,245,0.6)" if is_active else "#334155"
        name_weight = "700" if is_active else "500"
        seg_html += f'<div data-idx="{i}" data-mid="{m["id"]}" style="flex:1;text-align:center;padding:16px 8px;cursor:pointer;border-radius:10px;position:relative;z-index:1;" onclick="pcSlide(this,{i})">'
        seg_html += f'<div class="pc-mn" style="font-family:Outfit;font-size:15px;font-weight:{name_weight};color:{name_color};transition:color 0.25s,font-weight 0.25s;">{m["name"]}</div>'
        seg_html += f'<div class="pc-ms" style="font-family:IBM Plex Mono;font-size:11px;color:{sub_color};margin-top:4px;transition:color 0.25s;">{m["sub"]}</div>'
        seg_html += '</div>'

    seg_html += '</div></div>'

    st.markdown(seg_html, unsafe_allow_html=True)

    # Inject slide animation function into parent document
    components.html("""
    <script>
    (function(){
        var pd = window.parent;
        pd.pcSlide = function(el, idx) {
            var track = pd.document.getElementById('pc-mode-track');
            var ind = pd.document.getElementById('pc-mode-ind');
            if(!track || !ind) return;
            ind.style.left = 'calc(' + idx + ' * (100% / 3) + 3px)';
            var pills = track.querySelectorAll('[data-idx]');
            pills.forEach(function(p, i) {
                var mn = p.querySelector('.pc-mn'), ms = p.querySelector('.pc-ms');
                if(i === idx) { mn.style.color='#F0F2F5'; mn.style.fontWeight='700'; ms.style.color='rgba(240,242,245,0.6)'; }
                else { mn.style.color='#64748B'; mn.style.fontWeight='500'; ms.style.color='#334155'; }
            });
            var mid = el.getAttribute('data-mid');
            setTimeout(function(){
                var radios = pd.document.querySelectorAll('input[type=radio]');
                for(var r of radios) {
                    if(r.closest('[data-testid="stRadioGroup"]') && r.labels && r.labels[0] && r.labels[0].textContent.trim() === mid) {
                        r.click(); break;
                    }
                }
            }, 350);
        };
    })();
    </script>
    """, height=0)

    # Hidden radio that actually controls state (hidden via CSS)
    st.markdown('<style>[data-testid="stRadioGroup"] { position:absolute;opacity:0;height:0;overflow:hidden;pointer-events:none; }</style>', unsafe_allow_html=True)
    current_index = MODE_IDS.index(cur_mode) if cur_mode in MODE_IDS else 1
    selected_mode = st.radio("scan_mode_radio", options=MODE_IDS, index=current_index, label_visibility="collapsed", key="scan_mode_radio")

    # Handle mode change — reset confirmation
    ALL_STEPS = STEPS_QUICK_CHECK + STEPS_QUICK + STEPS_ACCURATE
    if selected_mode != cur_mode:
        for s in ALL_STEPS:
            st.session_state.pop(f"_img_{s['key']}", None)
            st.session_state.pop(f"cam_{s['key']}", None)
        st.session_state.pop('_analysis_done', None)
        st.session_state.pop('_scan_confirmed', None)
        st.session_state.pop('_case_warning_acknowledged', None)
        st.session_state['_scan_mode'] = selected_mode
        st.rerun()

    scan_mode = cur_mode
    if scan_mode == 'quick_check':
        STEPS = STEPS_QUICK_CHECK
    elif scan_mode == 'standard':
        STEPS = STEPS_QUICK
    else:
        STEPS = STEPS_ACCURATE
    total_steps = len(STEPS)

    # Confirmation gate — Start Scan button below the selector
    if not st.session_state.get('_scan_confirmed'):
        st.markdown('<div style="padding:12px 0 0 0;"></div>', unsafe_allow_html=True)
        if st.button("Start Scan", type="primary", use_container_width=True, key="start_scan_btn"):
            st.session_state['_scan_confirmed'] = True
            st.session_state['_just_started_scan'] = True
            st.rerun()
        return

    # ===== CASE / SCREEN PROTECTOR WARNING =====
    # Shown once between Start Scan and the first capture step. The model was trained on
    # bare phones — a case or screen protector hides scratches and cracks and produces
    # false "mint" readings. Dismiss-once via _case_warning_acknowledged.
    if not st.session_state.get('_case_warning_acknowledged'):
        # IMPORTANT: HTML must have NO leading whitespace on any line — Streamlit's markdown
        # parser treats 4+ space indents as code blocks even with unsafe_allow_html=True.
        warning_html = (
'<style>'
'@keyframes pcWarnIn { from { opacity: 0; transform: translateY(16px) scale(0.97); } to { opacity: 1; transform: translateY(0) scale(1); } }'
'@keyframes pcWarnIcon { 0%, 100% { transform: rotate(0deg); } 10%, 30% { transform: rotate(-8deg); } 20%, 40% { transform: rotate(8deg); } 50% { transform: rotate(0deg); } }'
'</style>'
'<div style="animation: pcWarnIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) both; background: linear-gradient(180deg, #0C1018 0%, #0A0E16 100%); border: 1px solid rgba(251, 191, 36, 0.25); border-radius: 16px; padding: 28px 24px; margin: 12px 0 16px 0; box-shadow: 0 12px 40px rgba(251, 191, 36, 0.08), 0 0 0 1px rgba(251, 191, 36, 0.05);">'
'<div style="display:flex;align-items:center;gap:14px;margin-bottom:16px;">'
'<div style="width: 44px; height: 44px; background: rgba(251, 191, 36, 0.12); border: 1px solid rgba(251, 191, 36, 0.25); border-radius: 12px; display: flex; align-items: center; justify-content: center; animation: pcWarnIcon 2.5s ease-in-out infinite; flex-shrink: 0;">'
'<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FBBF24" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
'</div>'
'<div style="flex:1;">'
'<div style="font-family:Outfit,sans-serif;font-size:18px;font-weight:700;color:#F0F2F5;letter-spacing:-0.01em;">Before you start</div>'
'<div style="font-family:Source Sans 3,sans-serif;font-size:13px;color:#FBBF24;margin-top:2px;">For an accurate condition check</div>'
'</div>'
'</div>'
'<div style="font-family:Source Sans 3,sans-serif;font-size:15px;color:#C9D1D9;line-height:1.65;margin-bottom:18px;">'
'Please <strong style="color:#F0F2F5;">remove the case and the screen protector</strong> from your phone before taking photos.'
'</div>'
'<div style="background:rgba(99,102,241,0.04);border:1px solid #1A2035;border-radius:10px;padding:14px 16px;margin-bottom:4px;">'
'<div style="font-family:Outfit,sans-serif;font-size:11px;font-weight:600;color:#6366F1;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:8px;">Why this matters</div>'
'<div style="font-family:Source Sans 3,sans-serif;font-size:13px;color:#94A3B8;line-height:1.6;">Cases and screen protectors hide scratches, cracks, and edge wear — leading to an inaccurate result.</div>'
'</div>'
'</div>'
        )
        st.markdown(warning_html, unsafe_allow_html=True)

        if st.button("Got it — my phone is bare", type="primary", use_container_width=True, key="case_warning_ack_btn"):
            st.session_state['_case_warning_acknowledged'] = True
            st.session_state['_just_started_scan'] = True
            st.rerun()
        return

    # ===== GUIDED CAPTURE =====

    # Auto-scroll to capture section after starting scan or moving to next step
    if st.session_state.pop('_just_started_scan', False):
        components.html("""
        <script>
        (function(){
            var attempts = 0;
            var maxAttempts = 20;
            function tryScroll() {
                attempts++;
                try {
                    var w = window.parent;
                    var d = w.document;
                    // Try finding the anchor
                    var el = d.getElementById('pc-capture-start');
                    if(!el) {
                        // Try finding any file uploader as fallback
                        var ups = d.querySelectorAll('[data-testid="stFileUploader"]');
                        if(ups.length > 0) el = ups[ups.length-1];
                    }
                    if(el) {
                        el.scrollIntoView({behavior:'smooth', block:'center'});
                        return;
                    }
                    // Not found yet — try again
                    if(attempts < maxAttempts) {
                        setTimeout(tryScroll, 300);
                    } else {
                        // Give up, scroll to bottom
                        w.scrollTo({top: d.body.scrollHeight, behavior:'smooth'});
                    }
                } catch(e) {}
            }
            setTimeout(tryScroll, 400);
        })();
        </script>
        """, height=0)

    captures = {}
    for step in STEPS:
        captures[step["key"]] = st.session_state.get(f"_img_{step['key']}")

    current_step_idx = 0
    for i, step in enumerate(STEPS):
        if captures[step["key"]] is None:
            current_step_idx = i
            break
    else:
        current_step_idx = total_steps  # all done

    captured_count = sum(1 for v in captures.values() if v is not None)

    st.markdown('<style>@keyframes pcCapFade{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}</style>', unsafe_allow_html=True)
    st.markdown('<div style="animation:pcCapFade 0.6s cubic-bezier(0.16,1,0.3,1) 0.15s both;">', unsafe_allow_html=True)
    st.markdown("---")
    st.progress(captured_count / total_steps)
    st.markdown(f'<div style="font-family:IBM Plex Mono;font-size:12px;color:#7A8599;text-align:center;margin:4px 0 16px 0;">{captured_count}/{total_steps} photos captured</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Step tracker — compact HTML-only horizontal dots
    if current_step_idx < total_steps:
        dots_html = ""
        for i, step in enumerate(STEPS):
            short = step["title"].split("—")[0].split("Screen")[0].split("Glass")[0].split("Edge")[0].strip()
            if not short:
                short = step["title"].split("—")[0].strip()
            if i < captured_count:
                dots_html += f'<div style="flex:1;text-align:center;"><div style="width:28px;height:28px;border-radius:50%;background:#34D399;margin:0 auto;display:flex;align-items:center;justify-content:center;font-size:12px;color:#06090F;font-weight:700;">{step["num"]}</div><div style="font-size:9px;color:#34D399;margin-top:3px;">Done</div></div>'
            elif i == current_step_idx:
                dots_html += f'<div style="flex:1;text-align:center;"><div style="width:28px;height:28px;border-radius:50%;border:2px solid #6366F1;background:rgba(99,102,241,0.15);margin:0 auto;display:flex;align-items:center;justify-content:center;font-size:12px;color:#6366F1;font-weight:700;">{step["num"]}</div><div style="font-size:9px;color:#6366F1;margin-top:3px;font-weight:600;">Now</div></div>'
            else:
                dots_html += f'<div style="flex:1;text-align:center;"><div style="width:28px;height:28px;border-radius:50%;background:#111827;border:1px solid #1E293B;margin:0 auto;display:flex;align-items:center;justify-content:center;font-size:11px;color:#334155;font-weight:600;">{step["num"]}</div><div style="font-size:9px;color:#334155;margin-top:3px;">{short}</div></div>'
        st.markdown(f'<div style="display:flex;gap:2px;padding:8px 0 16px 0;animation:pcCapFade 0.5s cubic-bezier(0.16,1,0.3,1) 0.25s both;">{dots_html}</div>', unsafe_allow_html=True)

    # Show current step
    if current_step_idx < total_steps:
        step = STEPS[current_step_idx]

        st.markdown(f"""<div style="animation:pcCapFade 0.5s cubic-bezier(0.16,1,0.3,1) 0.35s both;">
        <div style="background:#0C1018;border:1px solid #1E293B;border-radius:14px;padding:20px;margin-bottom:12px;">
            <div style="font-family:Outfit;font-size:18px;font-weight:700;color:#F0F2F5;text-align:center;margin-bottom:8px;">{step['title']}</div>
            <div style="font-size:14px;color:#818CF8;text-align:center;margin-bottom:4px;">{step['instruction']}</div>
            <div style="font-size:12px;color:#64748B;text-align:center;line-height:1.5;">{step['detail']}</div>
        </div></div>""", unsafe_allow_html=True)

        render_positioning_guide(step['guide'])

        st.markdown(f"""<div style="animation:pcCapFade 0.5s cubic-bezier(0.16,1,0.3,1) 0.45s both;">
        <div id="pc-capture-start" style="text-align:center;padding:12px;margin-bottom:8px;background:rgba(99,102,241,0.04);border:1px solid #1A2035;border-radius:12px;">
            <div style="font-family:Outfit;font-size:15px;font-weight:600;color:#6366F1;">Tap below to take photo</div>
            <div style="font-family:IBM Plex Mono;font-size:11px;color:#7A8599;margin-top:4px;">Choose Camera when prompted</div>
        </div></div>""", unsafe_allow_html=True)
        cam = st.file_uploader(f"Take photo: {step['title']}", type=['jpg','jpeg','png','webp','heic'], label_visibility="collapsed", key=f"cam_{step['key']}")

        if cam:
            # Store the image bytes in session state
            img_bytes = cam.getvalue()
            uploaded_img = Image.open(io.BytesIO(img_bytes))
            
            # ── INSTANT VIEW GATE CHECK for front/back photos ──
            expected_side = None
            if step['key'].startswith('cf'):
                expected_side = "front"
            elif step['key'].startswith('cb'):
                expected_side = "back"
            
            gate_rejected = False
            if expected_side and view_gate:
                view_cls, view_conf = predict_view(view_gate, uploaded_img)
                
                if view_cls == "other" and view_conf > VIEW_THRESHOLD:
                    # Not a phone at all
                    gate_rejected = True
                    st.markdown(f"""<div style="background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.2);border-radius:14px;padding:20px;margin:12px 0;">
                        <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                            <span style="font-size:20px;">🚫</span>
                            <div style="font-family:Outfit;font-size:15px;font-weight:600;color:#F87171;">No phone detected</div>
                        </div>
                        <div style="font-family:Source Sans 3;font-size:14px;color:#C9D1D9;line-height:1.6;">This image doesn't appear to contain a phone. Please retake with the phone clearly visible in the frame.</div>
                        <div style="font-family:IBM Plex Mono;font-size:11px;color:#7A8599;margin-top:8px;">Confidence: {view_conf:.0%}</div>
                    </div>""", unsafe_allow_html=True)
                    
                elif view_cls != expected_side and view_conf > VIEW_THRESHOLD:
                    # Wrong side
                    gate_rejected = True
                    st.markdown(f"""<div style="background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.2);border-radius:14px;padding:20px;margin:12px 0;">
                        <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                            <span style="font-size:20px;">🔄</span>
                            <div style="font-family:Outfit;font-size:15px;font-weight:600;color:#FBBF24;">Wrong side detected</div>
                        </div>
                        <div style="font-family:Source Sans 3;font-size:14px;color:#C9D1D9;line-height:1.6;">This looks like the <strong>{view_cls}</strong> of the phone, but we need the <strong>{expected_side}</strong>. Please retake with the correct side facing the camera.</div>
                        <div style="font-family:IBM Plex Mono;font-size:11px;color:#7A8599;margin-top:8px;">Confidence: {view_conf:.0%}</div>
                    </div>""", unsafe_allow_html=True)
            
            if gate_rejected:
                # Don't save this photo — clear the uploader and ask to retake
                st.session_state.pop(f"_img_{step['key']}", None)
                st.markdown(f"""<div style="text-align:center;padding:8px;background:#0C1018;border:1px solid #F87171;border-radius:10px;margin:8px 0;">
                    <span style="font-family:Outfit;font-size:14px;font-weight:600;color:#F87171;">Please retake this photo</span>
                </div>""", unsafe_allow_html=True)
            else:
                # Photo passed validation — save it
                st.session_state[f"_img_{step['key']}"] = img_bytes

                # Show captured image (small centered preview)
                col_pad1, col_img, col_pad2 = st.columns([2, 1, 2])
                with col_img:
                    st.image(uploaded_img, use_container_width=True)
                st.markdown(f"""<div style="text-align:center;padding:8px;background:#0C1018;border:1px solid #34D399;border-radius:10px;margin:8px 0;">
                    <span style="font-family:Outfit;font-size:14px;font-weight:600;color:#34D399;">Photo {step['num']} captured ✓</span>
                </div>""", unsafe_allow_html=True)

                # Next button
                if current_step_idx < total_steps - 1:
                    next_step = STEPS[current_step_idx + 1]
                    if st.button(f"Next → {next_step['title']}", type="primary", use_container_width=True):
                        st.session_state['_just_started_scan'] = True
                        st.rerun()
                else:
                    if st.button("🔍 Analyze Phone", type="primary", use_container_width=True):
                        st.rerun()

        # Start over — hidden in expander far from camera button
        if captured_count > 0:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("🔄 Start over (reset all photos)", use_container_width=True, key="reset_btn"):
                for s in STEPS:
                    st.session_state.pop(f"_img_{s['key']}", None)
                    st.session_state.pop(f"cam_{s['key']}", None)
                for k in ['_analysis_done','_scan_confirmed','_case_warning_acknowledged','_front_v','_back_v','_front_results','_back_results','_edge_results','report_front','report_back','report_edge','chat_history','_ck']:
                    st.session_state.pop(k, None)
                st.rerun()
        return

    # ===== ALL CAPTURED — ANALYZE =====

    # Load images from session state (handle all modes including quick_check with no edges)
    front_keys = [k for k in ["cf1","cf2"] if captures.get(k)]
    back_keys = [k for k in ["cb1","cb2"] if captures.get(k)]
    front_imgs = [Image.open(io.BytesIO(captures[k])) for k in front_keys]
    back_imgs = [Image.open(io.BytesIO(captures[k])) for k in back_keys]
    has_edges = all(captures.get(k) for k in ["cel","cer","cet","ceb"])
    edge_imgs = {}
    if has_edges:
        edge_imgs = {"Left": Image.open(io.BytesIO(captures["cel"])), "Right": Image.open(io.BytesIO(captures["cer"])),
                     "Top": Image.open(io.BytesIO(captures["cet"])), "Bottom": Image.open(io.BytesIO(captures["ceb"]))}

    # ── Run all predictions (with loading animation) ──
    # View gate already validated each photo at capture time
    if '_analysis_done' not in st.session_state or not st.session_state._analysis_done:
        # Full-page analysis animation via components.html
        components.html("""
        <script>
        (function(){
            var pd = window.parent.document;
            if(pd.getElementById('pc-analyze')) return;
            var ov = pd.createElement('div');
            ov.id = 'pc-analyze';
            ov.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;background:#06090F;z-index:99998;display:flex;flex-direction:column;align-items:center;justify-content:center;pointer-events:none';
            ov.innerHTML = '<style>@keyframes aspin{to{transform:rotate(360deg)}} @keyframes apulse{0%,100%{opacity:0.3}50%{opacity:1}} @keyframes acheck{from{opacity:0;transform:scale(0.5)}to{opacity:1;transform:scale(1)}} @keyframes afadeup{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}} .aring{width:64px;height:64px;border:3px solid #1E293B;border-top:3px solid #6366F1;border-radius:50%;animation:aspin 0.8s linear infinite;margin-bottom:32px} .astep{display:flex;align-items:center;gap:14px;padding:12px 0;max-width:280px;width:100%;animation:afadeup 0.4s ease-out both} .astep:nth-child(2){animation-delay:0.8s} .astep:nth-child(3){animation-delay:1.6s} .adot{width:10px;height:10px;border-radius:50%;flex-shrink:0;background:#1E293B;transition:background 0.5s ease-out,box-shadow 0.5s ease-out} .adot.go{background:#6366F1;box-shadow:0 0 8px rgba(99,102,241,0.4)} .adot.done{background:#34D399;box-shadow:0 0 8px rgba(52,211,153,0.4)} .alabel{font-family:Source Sans 3,sans-serif;font-size:14px;color:#64748B;transition:color 0.4s ease-out} .alabel.active{color:#F0F2F5} .alabel.done{color:#34D399}</style><div class="aring" id="aring"></div><div style="font-family:Outfit,sans-serif;font-size:20px;font-weight:700;color:#F0F2F5;margin-bottom:6px">Analyzing your phone</div><div style="font-family:Source Sans 3,sans-serif;font-size:13px;color:#64748B;margin-bottom:32px">Running AI models on your photos</div><div style="padding:0 20px"><div class="astep" id="as1"><div class="adot go" id="ad1"></div><span class="alabel active" id="al1">Checking front screen...</span></div><div class="astep" id="as2"><div class="adot" id="ad2"></div><span class="alabel" id="al2">Checking back panel</span></div><div class="astep" id="as3"><div class="adot" id="ad3"></div><span class="alabel" id="al3">Inspecting edges</span></div></div>';
            pd.body.appendChild(ov);
            setTimeout(function(){
                pd.getElementById('ad1').className='adot done';pd.getElementById('al1').className='alabel done';pd.getElementById('al1').textContent='Front screen checked';
                pd.getElementById('ad2').className='adot go';pd.getElementById('al2').className='alabel active';pd.getElementById('al2').textContent='Checking back panel...';
            },1200);
            setTimeout(function(){
                pd.getElementById('ad2').className='adot done';pd.getElementById('al2').className='alabel done';pd.getElementById('al2').textContent='Back panel checked';
                pd.getElementById('ad3').className='adot go';pd.getElementById('al3').className='alabel active';pd.getElementById('al3').textContent='Inspecting edges...';
            },2200);
            setTimeout(function(){
                pd.getElementById('ad3').className='adot done';pd.getElementById('al3').className='alabel done';pd.getElementById('al3').textContent='Edges inspected';
                pd.getElementById('aring').style.borderTopColor='#34D399';
            },3000);
        })();
        </script>
        """, height=0)

        # Run predictions — view gate already passed
        front_results = []
        if front_model:
            for img in front_imgs: front_results.append(predict_phone(front_model, img, clahe_on=CLAHE_FRONT, sharpen_on=True))
        front_v = worst_case(front_results) if front_results else ("mint", 0.5, {})

        back_results = []
        if back_model:
            for img in back_imgs: back_results.append(predict_phone(back_model, img, clahe_on=CLAHE_BACK))
        back_v = worst_case(back_results) if back_results else ("mint", 0.5, {})

        edge_result_list = []
        if has_edges:
            for name in ["Left","Right","Top","Bottom"]:
                if edge_model:
                    cls, conf, probs = predict_edge(edge_model, edge_imgs[name], clahe_on=False, yolo_model=yolo_model)
                else:
                    cls, conf, probs = ("mint", 0.5, {"mint":0.5,"used":0.5})
                edge_result_list.append((name, cls, conf))

        # Store in session state
        st.session_state._front_v = front_v
        st.session_state._back_v = back_v
        st.session_state._front_results = front_results
        st.session_state._back_results = back_results
        st.session_state._edge_results = edge_result_list
        st.session_state._analysis_done = True

        # Remove analysis overlay on next load
        import time
        time.sleep(1)  # Let animation finish
        st.rerun()

    # Retrieve cached results
    front_v = st.session_state._front_v
    back_v = st.session_state._back_v
    front_results = st.session_state._front_results
    back_results = st.session_state._back_results
    edge_result_list = st.session_state._edge_results
    used_edge_names = [n for n, c, _ in edge_result_list if c == "used"]

    # Remove analysis overlay with fade
    components.html("""
    <script>
    (function(){
        var pd = window.parent.document;
        var ov = pd.getElementById('pc-analyze');
        if(ov){
            ov.style.transition = 'opacity 0.8s ease-out';
            ov.style.opacity = '0';
            setTimeout(function(){ if(ov.parentNode) ov.remove(); }, 900);
        }
    })();
    </script>
    """, height=0)

    # Repair estimates
    fr_l = fr_h = br_l = br_h = er_l = er_h = None
    if front_v[0] == "damaged" and phone_model:
        fr_l, fr_h = get_repair_estimate(brand, phone_model, "screen")
    if back_v[0] == "damaged" and phone_model:
        br_l, br_h = get_repair_estimate(brand, phone_model, "back")
    if used_edge_names and phone_model:
        er_l, er_h = get_edge_repair_estimate(brand, phone_model)

    # ════════════════════════════════════════════════════
    # RESULTS HEADER
    # ════════════════════════════════════════════════════
    st.markdown(f"""<div style="text-align:center;padding:32px 0 8px 0;">
        <div style="font-family:Outfit;font-size:11px;font-weight:500;color:#64748B;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">Scan Complete</div>
        <div style="font-family:Outfit;font-size:22px;font-weight:700;color:#F0F2F5;">{brand} {phone_model}</div>
        <div style="font-family:Source Sans 3;font-size:13px;color:#64748B;margin-top:4px;">{
            "Quick Check — " + str(len(front_results)) + " front, " + str(len(back_results)) + " back" if scan_mode == "quick_check"
            else "Standard Scan — " + str(len(front_results)) + " front, " + str(len(back_results)) + " back, 4 edges" if scan_mode == "standard"
            else "Accurate Scan — " + str(len(front_results)) + " front, " + str(len(back_results)) + " back, 4 edges"
        }</div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div style="width:60px;height:2px;background:linear-gradient(90deg,#6366F1,#34D399);margin:8px auto 24px auto;border-radius:1px;"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════
    # SECTION 1: OVERALL VERDICT
    # ════════════════════════════════════════════════════
    overall_info = render_overall_verdict(front_v, back_v, edge_result_list, brand, phone_model)

    # ════════════════════════════════════════════════════
    # SECTION 2: DETAILED BREAKDOWN
    # ════════════════════════════════════════════════════
    st.markdown('<div style="margin-top:32px;padding-top:24px;border-top:1px solid #1E293B;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:Outfit;font-size:11px;font-weight:500;color:#64748B;letter-spacing:2px;text-transform:uppercase;margin-bottom:16px;">Detailed Breakdown</div>', unsafe_allow_html=True)

    f_cfg = CLASS_CONFIG[front_v[0]]
    b_cfg = CLASS_CONFIG[back_v[0]]

    if has_edges:
        edge_status = f"{len(used_edge_names)} worn" if used_edge_names else "All mint"
        tab_front, tab_back, tab_edges = st.tabs([
            f"Front — {f_cfg['label']}",
            f"Back — {b_cfg['label']}",
            f"Edges — {edge_status}"
        ])
    else:
        tab_front, tab_back = st.tabs([
            f"Front — {f_cfg['label']}",
            f"Back — {b_cfg['label']}"
        ])
        tab_edges = None

    # Front details
    with tab_front:
        st.image(front_imgs[0], use_container_width=True)
        n_front = len(front_results)
        angles = ["Flat 0°", "Tilted 45°"][:n_front]
        for i, (cls, conf, _) in enumerate(front_results):
            render_angle_dot(cls, angles[i])
        render_grade_card(front_v[0], front_v[1], front_v[2], show_probs, f"WORST OF {n_front} ANGLE{'S' if n_front>1 else ''}")
        if front_v[0] == "damaged" and fr_l:
            render_cost(fr_l, fr_h, "Cracked screen", brand, phone_model)

    # Back details
    with tab_back:
        st.image(back_imgs[0], use_container_width=True)
        n_back = len(back_results)
        angles_b = ["Flat 0°", "Tilted 45°"][:n_back]
        for i, (cls, conf, _) in enumerate(back_results):
            render_angle_dot(cls, angles_b[i])
        render_grade_card(back_v[0], back_v[1], back_v[2], show_probs, f"WORST OF {n_back} ANGLE{'S' if n_back>1 else ''}")
        if back_v[0] == "damaged" and br_l:
            render_cost(br_l, br_h, "Cracked back", brand, phone_model)

    # Edge details
    if has_edges and tab_edges is not None:
      with tab_edges:
        ec1, ec2 = st.columns(2)
        for i, (name, cls, conf) in enumerate(edge_result_list):
            ecfg = CLASS_CONFIG[cls]
            col = ec1 if i % 2 == 0 else ec2
            with col:
                st.image(edge_imgs[name], use_container_width=True)
                st.markdown(f'<div style="text-align:center;padding:6px 0 12px 0;"><div style="font-size:12px;color:#7A8599;">{name}</div><div style="font-size:16px;margin:4px 0;">{ecfg["emoji"]}</div><div style="font-family:Outfit;font-size:13px;font-weight:600;color:{ecfg["dot"]};">{ecfg["label"]} ({conf*100:.0f}%)</div></div>', unsafe_allow_html=True)
        if used_edge_names and er_l:
            render_cost(er_l, er_h, f"Edge polish ({', '.join(used_edge_names)})", brand, phone_model)

    # ════════════════════════════════════════════════════
    # SECTION 3: AI CONDITION REPORT
    # ════════════════════════════════════════════════════
    if is_ollama:
        cache_key = str(brand) + str(phone_model) + str(hash(str(captures.values())))
        if st.session_state.get('_ck') != cache_key:
            st.session_state._ck = cache_key
            st.session_state.report_front = st.session_state.report_back = st.session_state.report_edge = None
            st.session_state.chat_history = []

        # Generate reports in background
        if st.session_state.report_front is None:
            with st.spinner("AI is writing your detailed report..."):
                st.session_state.report_front = generate_analysis(front_imgs[0], front_v[0], front_v[1], front_v[2], "front", brand if brand!="--" else None, phone_model, fr_l, fr_h)
                st.session_state.report_back = generate_analysis(back_imgs[0], back_v[0], back_v[1], back_v[2], "back", brand if brand!="--" else None, phone_model, br_l, br_h)
                if has_edges:
                    st.session_state.report_edge = generate_edge_analysis([(n, edge_imgs[n]) for n in ["Left","Right","Top","Bottom"]], used_edge_names, brand if brand!="--" else None, phone_model, er_l, er_h)
                else:
                    st.session_state.report_edge = "Edges were not scanned in Quick Check mode."
                st.rerun()

        st.markdown('<div style="margin-top:32px;padding-top:24px;border-top:1px solid #1E293B;"></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-family:Outfit;font-size:11px;font-weight:500;color:#64748B;letter-spacing:2px;text-transform:uppercase;margin-bottom:16px;">AI Condition Report</div>', unsafe_allow_html=True)
        for key, title in [("report_front","Front Screen"),("report_back","Back Glass"),("report_edge","Edges")]:
            if st.session_state.get(key):
                st.markdown(f'<div style="font-family:IBM Plex Mono;font-size:11px;color:#6366F1;letter-spacing:1px;margin:20px 0 12px 0;padding-top:12px;border-top:1px solid #1A2035;">{title}</div>', unsafe_allow_html=True)
                st.markdown(st.session_state[key])

    # ════════════════════════════════════════════════════
    # SECTION 4: CHAT
    # ════════════════════════════════════════════════════
    if is_ollama:
        st.markdown('<div style="margin-top:32px;padding-top:24px;border-top:1px solid #1E293B;"></div>', unsafe_allow_html=True)
        st.markdown("""<div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
            <div style="width:28px;height:28px;border-radius:6px;background:#6366F1;display:flex;align-items:center;justify-content:center;font-size:14px;color:#F0F2F5;font-weight:700;">?</div>
            <div>
                <div style="font-family:Outfit;font-size:14px;font-weight:600;color:#F0F2F5;">Have questions about your results?</div>
                <div style="font-family:Source Sans 3;font-size:12px;color:#64748B;">Ask about repairs, selling price, or insurance</div>
            </div>
        </div>""", unsafe_allow_html=True)

        ctx = f"Front:{front_v[0]}. Back:{back_v[0]}. Edges:{','.join(n+'='+c for n,c,_ in edge_result_list)}. Override:{overall_info.get('edge_override')}. Overall:{overall_info['overall']}. Phone:{brand} {phone_model}."
        if overall_info['total_low']: ctx += f" Repair:{overall_info['total_low']}-{overall_info['total_high']}EUR."

        for msg in st.session_state.chat_history:
            render_chat_msg(msg["role"], msg["content"])

        chat_col1, chat_col2 = st.columns([5, 1])
        with chat_col1:
            q = st.text_input("q", placeholder="Should I repair or sell as-is?", label_visibility="collapsed", key="chat")
        with chat_col2:
            send = st.button("→", type="primary", use_container_width=True)

        if send and q:
            with st.spinner(""):
                resp = generate_chat(q, ctx, st.session_state.chat_history, image_to_base64(front_imgs[0]))
            st.session_state.chat_history.extend([{"role":"user","content":q},{"role":"assistant","content":resp}])
            st.rerun()

        if st.session_state.chat_history:
            if st.button("Clear chat", use_container_width=True, key="clr_chat"):
                st.session_state.chat_history = []
                st.rerun()

    # ════════════════════════════════════════════════════
    # SCAN AGAIN
    # ════════════════════════════════════════════════════
    st.markdown('<div style="margin-top:40px;padding-top:24px;border-top:1px solid #1E293B;"></div>', unsafe_allow_html=True)
    if st.button("🔄 Scan Another Phone", type="primary", use_container_width=True):
        for s in STEPS_QUICK_CHECK + STEPS_QUICK + STEPS_ACCURATE:
            st.session_state.pop(f"_img_{s['key']}", None)
            st.session_state.pop(f"cam_{s['key']}", None)
        for k in ['report_front','report_back','report_edge','chat_history','_analysis_done','_scan_confirmed','_case_warning_acknowledged','_front_v','_back_v','_front_results','_back_results','_edge_results','_ck']:
            st.session_state.pop(k, None)
        st.rerun()

    st.markdown('<div style="text-align:center;padding:48px 0 16px 0;"><div style="font-family:IBM Plex Mono;font-size:10px;color:#2A3545;letter-spacing:2px;">PHONECHECK AI &middot; BLOCK C 2026</div></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()