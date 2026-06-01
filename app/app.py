import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(
    page_title="FASHIONNET | CNN",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #f5f0e8 !important;
    color: #1a1a1a !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: #1a1a1a !important;
    border-right: 4px solid #ff3d00 !important;
}
[data-testid="stSidebar"] * { color: #f5f0e8 !important; }
[data-testid="stSidebar"] label { color: #aaa !important; font-size: 0.75rem !important; }
[data-testid="stSidebar"] .stSlider > div > div > div { background: #ff3d0044 !important; }
[data-testid="stSidebar"] .stSlider > div > div > div > div { background: #ff3d00 !important; }

.hero-wrap {
    background: #1a1a1a;
    padding: 0;
    margin: -20px -20px 0;
    display: flex;
    min-height: 260px;
    overflow: hidden;
}
.hero-left {
    flex: 1;
    padding: 48px 48px 36px;
    position: relative;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #ff3d00;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 6rem;
    color: #f5f0e8;
    line-height: 0.9;
    margin-bottom: 16px;
    letter-spacing: 2px;
}
.hero-title span { color: #ff3d00; }
.hero-desc {
    font-size: 0.9rem;
    color: #888;
    max-width: 380px;
    line-height: 1.7;
}
.hero-right {
    width: 340px;
    background: #ff3d00;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    padding: 32px;
    gap: 16px;
}
.hero-stat-box {
    background: #1a1a1a;
    width: 100%;
    padding: 14px 20px;
    border-radius: 4px;
}
.hero-stat-val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem;
    color: #ff3d00;
    line-height: 1;
}
.hero-stat-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: #888;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.sec {
    display: flex;
    align-items: center;
    gap: 0;
    margin: 36px 0 20px;
}
.sec-num {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    color: #ff3d00;
    line-height: 1;
    margin-right: 12px;
    opacity: 0.3;
}
.sec-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6rem;
    color: #1a1a1a;
    letter-spacing: 3px;
    text-transform: uppercase;
    border-bottom: 3px solid #ff3d00;
    padding-bottom: 4px;
}
.class-grid { display: grid; grid-template-columns: repeat(5,1fr); gap: 3px; margin: 16px 0; }
.class-card {
    background: #1a1a1a;
    padding: 12px;
    text-align: center;
    transition: background 0.2s;
}
.class-card:hover { background: #ff3d00; }
.class-icon { font-size: 1.6rem; }
.class-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    color: #888;
    letter-spacing: 1px;
    margin-top: 4px;
    text-transform: uppercase;
}
.stButton > button {
    font-family: 'Bebas Neue', sans-serif !important;
    background: #ff3d00 !important;
    color: #f5f0e8 !important;
    border: none !important;
    border-radius: 0 !important;
    letter-spacing: 4px !important;
    font-size: 1rem !important;
    width: 100% !important;
    padding: 16px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #1a1a1a !important;
    color: #ff3d00 !important;
    box-shadow: inset 0 0 0 2px #ff3d00 !important;
}
div[data-testid="stMetric"] {
    background: #1a1a1a !important;
    border-left: 4px solid #ff3d00 !important;
    border-radius: 0 !important;
    padding: 16px !important;
}
div[data-testid="stMetric"] label {
    font-family: 'JetBrains Mono', monospace !important;
    color: #666 !important;
    font-size: 0.65rem !important;
    letter-spacing: 2px !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'Bebas Neue', sans-serif !important;
    color: #f5f0e8 !important;
    font-size: 2rem !important;
}
[data-testid="stFileUploader"] {
    background: #1a1a1a !important;
    border: 2px dashed #ff3d0066 !important;
    border-radius: 4px !important;
}
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #f5f0e8; }
::-webkit-scrollbar-thumb { background: #ff3d00; }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = ['T-Shirt','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle Boot']
CLASS_ICONS = ['👕','👖','🧥','👗','🧣','👡','👔','👟','👜','👢']

# ── CNN Model (PyTorch) ──────────────────────────────────────────────────────
class FashionCNN(nn.Module):
    def __init__(self, filters1, filters2, dense_units, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, filters1, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(filters1, filters2, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            nn.Flatten(),
            nn.Linear(filters2 * 7 * 7, dense_units), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, 10)
        )

    def forward(self, x):
        return self.net(x)

# ── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    transform = transforms.ToTensor()
    train_ds = datasets.FashionMNIST(root='./data', train=True,  download=True, transform=transform)
    test_ds  = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Small subset — fast training
    train_idx = torch.arange(2500)
    test_idx  = torch.arange(500)

    X_train = torch.stack([train_ds[i][0] for i in train_idx])   # (2500,1,28,28)
    y_train = torch.tensor([train_ds[i][1] for i in train_idx])
    X_test  = torch.stack([test_ds[i][0]  for i in test_idx])
    y_test  = torch.tensor([test_ds[i][1]  for i in test_idx])

    return X_train, X_test, y_train, y_test

with st.spinner("Loading Fashion MNIST..."):
    X_train, X_test, y_train, y_test = load_data()

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-left">
        <div class="hero-eyebrow">▸ CNN · Computer Vision · Fashion Intelligence</div>
        <div class="hero-title">FASHION<span>NET</span></div>
        <div class="hero-desc">Classifying fashion items with Convolutional Neural Networks.
        Configure your model architecture and see real-time results.</div>
    </div>
    <div class="hero-right">
        <div class="hero-stat-box">
            <div class="hero-stat-val">{len(X_train):,}</div>
            <div class="hero-stat-lbl">Training Images</div>
        </div>
        <div class="hero-stat-box">
            <div class="hero-stat-val">10</div>
            <div class="hero-stat-lbl">Fashion Classes</div>
        </div>
        <div class="hero-stat-box">
            <div class="hero-stat-val">28×28</div>
            <div class="hero-stat-lbl">Image Size</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Class Grid
st.markdown('<div class="sec"><div class="sec-num">01</div><div class="sec-title">Fashion Classes</div></div>', unsafe_allow_html=True)
st.markdown('<div class="class-grid">' +
    ''.join([f'<div class="class-card"><div class="class-icon">{CLASS_ICONS[i]}</div><div class="class-name">{CLASS_NAMES[i]}</div></div>' for i in range(10)]) +
    '</div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown('<p style="font-family:Bebas Neue;font-size:1.4rem;color:#ff3d00;letter-spacing:3px;padding:12px 0 4px;">CONFIG</p>', unsafe_allow_html=True)
st.sidebar.markdown('<div style="height:3px;background:#ff3d00;margin-bottom:20px;"></div>', unsafe_allow_html=True)
epochs       = st.sidebar.slider("Epochs",       1,   8,   3)
batch_size   = st.sidebar.selectbox("Batch Size", [64, 128], index=0)
dropout_rate = st.sidebar.slider("Dropout",      0.1, 0.4, 0.2)
filters1     = st.sidebar.slider("Conv1 Filters", 8,  32,  16)
filters2     = st.sidebar.slider("Conv2 Filters", 16, 64,  32)
dense_units  = st.sidebar.slider("Dense Units",  32,  128, 64)
lr           = st.sidebar.select_slider("Learning Rate", [0.0001, 0.001, 0.01], value=0.001)

# ── Train ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def build_and_train(epochs, batch_size, dropout_rate, filters1, filters2, dense_units, lr):
    device = torch.device('cpu')
    model  = FashionCNN(filters1, filters2, dense_units, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(X_train, y_train)
    loader   = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

    # 10 % validation split
    val_size   = int(0.1 * len(X_train))
    train_size = len(X_train) - val_size
    X_tr, X_val = X_train[:train_size], X_train[train_size:]
    y_tr, y_val = y_train[:train_size], y_train[train_size:]

    tr_ds  = TensorDataset(X_tr, y_tr)
    tr_ldr = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    patience_cnt  = 0
    best_state    = None

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for xb, yb in tr_ldr:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            correct    += (out.argmax(1) == yb).sum().item()
            total      += len(xb)

        train_acc  = correct / total
        train_loss = total_loss / total

        # Validation
        model.eval()
        with torch.no_grad():
            val_out  = model(X_val.to(device))
            val_loss = criterion(val_out, y_val.to(device)).item()
            val_acc  = (val_out.argmax(1) == y_val.to(device)).float().mean().item()

        history['accuracy'].append(train_acc)
        history['loss'].append(train_loss)
        history['val_accuracy'].append(val_acc)
        history['val_loss'].append(val_loss)

        # Early stopping (patience=2)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 2:
                break

    if best_state:
        model.load_state_dict(best_state)

    return model, history

with st.spinner("🧠 Training FashionNet..."):
    model, history = build_and_train(epochs, batch_size, dropout_rate,
                                     filters1, filters2, dense_units, lr)

# ── Evaluate ──────────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    logits = model(X_test)
    y_pred = logits.argmax(1).numpy()
    loss_val = nn.CrossEntropyLoss()(logits, y_test).item()
    accuracy = (y_pred == y_test.numpy()).mean()

y_true = y_test.numpy()

# ── Metrics ───────────────────────────────────────────────────────────────────
st.markdown('<div class="sec"><div class="sec-num">02</div><div class="sec-title">Model Performance</div></div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("ACCURACY",     f"{accuracy:.2%}")
c2.metric("LOSS",         f"{loss_val:.4f}")
c3.metric("TRAIN IMAGES", f"{len(X_train):,}")
c4.metric("TEST IMAGES",  f"{len(X_test):,}")

# ── Training Charts ────────────────────────────────────────────────────────────
st.markdown('<div class="sec"><div class="sec-num">03</div><div class="sec-title">Training History</div></div>', unsafe_allow_html=True)

def dark_ax(ax):
    ax.set_facecolor('#111')
    for s in ax.spines.values(): s.set_color('#333')
    ax.tick_params(colors='#666', labelsize=8)

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4), facecolor='#f5f0e8')
dark_ax(ax1); dark_ax(ax2)

epochs_range = range(len(history['accuracy']))
ax1.plot(epochs_range, history['accuracy'],     color='#ff3d00', linewidth=2.5, label='Train', marker='o', markersize=5)
ax1.plot(epochs_range, history['val_accuracy'], color='#f5f0e8', linewidth=2.5, label='Val',   linestyle='--', marker='s', markersize=5)
ax1.fill_between(epochs_range, history['accuracy'], alpha=0.1, color='#ff3d00')
ax1.set_title('ACCURACY', color='#f5f0e8', fontsize=11, fontweight='bold', fontfamily='monospace')
ax1.legend(facecolor='#111', labelcolor='#999', framealpha=0.6, fontsize=8)
ax1.set_xlabel('EPOCH', color='#666', fontsize=8)

ax2.plot(epochs_range, history['loss'],     color='#ff3d00', linewidth=2.5, label='Train', marker='o', markersize=5)
ax2.plot(epochs_range, history['val_loss'], color='#f5f0e8', linewidth=2.5, label='Val',   linestyle='--', marker='s', markersize=5)
ax2.fill_between(epochs_range, history['loss'], alpha=0.1, color='#ff3d00')
ax2.set_title('LOSS', color='#f5f0e8', fontsize=11, fontweight='bold', fontfamily='monospace')
ax2.legend(facecolor='#111', labelcolor='#999', framealpha=0.6, fontsize=8)
ax2.set_xlabel('EPOCH', color='#666', fontsize=8)

plt.tight_layout()
st.pyplot(fig1)
plt.close()

# ── Sample Grid + Confusion Matrix ────────────────────────────────────────────
st.markdown('<div class="sec"><div class="sec-num">04</div><div class="sec-title">Predictions & Analysis</div></div>', unsafe_allow_html=True)

col_a, col_b = st.columns([1.2, 1])

with col_a:
    st.markdown("**Sample Predictions**")
    indices = np.random.choice(len(X_test), 15, replace=False)
    fig2, axes = plt.subplots(3, 5, figsize=(10, 6), facecolor='#1a1a1a')
    for i, idx in enumerate(indices):
        ax = axes[i//5][i%5]
        ax.imshow(X_test[idx].squeeze().numpy(), cmap='gray')
        pred  = CLASS_NAMES[y_pred[idx]]
        true  = CLASS_NAMES[y_true[idx]]
        color = '#00e676' if pred == true else '#ff3d00'
        ax.set_title(f'{CLASS_ICONS[y_pred[idx]]}\n{pred[:5]}', color=color, fontsize=7, fontweight='bold')
        ax.axis('off')
        for s in ax.spines.values():
            s.set_edgecolor(color)
            s.set_linewidth(2)
    plt.tight_layout(pad=0.4)
    st.pyplot(fig2)
    plt.close()

with col_b:
    st.markdown("**Confusion Matrix**")
    fig3, ax3 = plt.subplots(figsize=(7, 6), facecolor='#f5f0e8')
    ax3.set_facecolor('#1a1a1a')
    cm    = confusion_matrix(y_true, y_pred)
    short = [c[:5].upper() for c in CLASS_NAMES]
    cmap  = matplotlib.colors.LinearSegmentedColormap.from_list("rr", ['#111', '#ff3d00'])
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax3,
                xticklabels=short, yticklabels=short,
                linewidths=1, linecolor='#1a1a1a',
                annot_kws={"size": 7, "color": "white", "weight": "bold"})
    ax3.set_xlabel('PREDICTED', color='#888', fontsize=8)
    ax3.set_ylabel('ACTUAL',    color='#888', fontsize=8)
    ax3.tick_params(colors='#888', labelsize=7)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

# ── Per-Class Accuracy ─────────────────────────────────────────────────────────
st.markdown('<div class="sec"><div class="sec-num">05</div><div class="sec-title">Per-Class Accuracy</div></div>', unsafe_allow_html=True)

class_acc = []
for i in range(10):
    mask = y_true == i
    class_acc.append(np.mean(y_pred[mask] == i) if mask.sum() > 0 else 0)

fig4, ax4 = plt.subplots(figsize=(13, 3.5), facecolor='#f5f0e8')
ax4.set_facecolor('#1a1a1a')
bar_colors = ['#ff3d00' if a >= 0.7 else '#ff3d0066' for a in class_acc]
bars = ax4.bar([f"{CLASS_ICONS[i]}\n{CLASS_NAMES[i][:6]}" for i in range(10)],
               class_acc, color=bar_colors, edgecolor='none', width=0.6)
ax4.axhline(np.mean(class_acc), color='#f5f0e8', linestyle='--', linewidth=1.5,
            label=f'Mean: {np.mean(class_acc):.2%}')
ax4.set_ylabel('ACCURACY', color='#666', fontsize=8)
ax4.set_ylim(0, 1.1)
ax4.legend(facecolor='#1a1a1a', labelcolor='#999', fontsize=9)
for s in ax4.spines.values(): s.set_color('#333')
ax4.tick_params(colors='#888', labelsize=8)
for bar, acc in zip(bars, class_acc):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{acc:.0%}', ha='center', va='bottom', color='#f5f0e8', fontsize=8, fontweight='bold')
plt.tight_layout()
st.pyplot(fig4)
plt.close()

# ── Classification Report ──────────────────────────────────────────────────────
st.markdown('<div class="sec"><div class="sec-num">06</div><div class="sec-title">Classification Report</div></div>', unsafe_allow_html=True)
report    = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(
    report_df.style.background_gradient(cmap='Oranges', subset=['precision','recall','f1-score']),
    use_container_width=True
)

st.markdown('<br><p style="font-family:JetBrains Mono;color:#ccc;text-align:center;letter-spacing:4px;font-size:0.65rem;background:#1a1a1a;padding:12px;">FASHIONNET · CNN · FASHION-MNIST · PYTORCH + STREAMLIT</p>', unsafe_allow_html=True)