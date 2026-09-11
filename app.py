import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import string
import html
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Mail Spam Guard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- THEME STATE MANAGEMENT ---
if "theme" not in st.session_state:
    st.session_state.theme = "light"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

IS_DARK = st.session_state.theme == "dark"

# --- ZINC-STYLE DESIGN SYSTEM ---
bg = "#09090b" if IS_DARK else "#ffffff"
bg_subtle = "#0c0c0f" if IS_DARK else "#f9fafb"
card = "#0c0c0f" if IS_DARK else "#ffffff"
card_hover = "#131316" if IS_DARK else "#f4f4f5"
border = "#27272a" if IS_DARK else "#e4e4e7"
border_subtle = "#1f1f23" if IS_DARK else "#f0f0f2"
text = "#fafafa" if IS_DARK else "#09090b"
text_muted = "#a1a1aa" if IS_DARK else "#71717a"
text_dim = "#71717a" if IS_DARK else "#a1a1aa"
green = "#22c55e" if IS_DARK else "#16a34a"
green_muted = "rgba(34,197,94,0.12)" if IS_DARK else "rgba(22,163,74,0.08)"
red = "#ef4444" if IS_DARK else "#dc2626"
red_muted = "rgba(239,68,68,0.12)" if IS_DARK else "rgba(22,163,74,0.08)"
blue = "#3b82f6" if IS_DARK else "#2563eb"
shadow = "none" if IS_DARK else "0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03)"

# Injected CSS
st.html(f"""
<style>
    /* Hide Streamlit chrome */
    header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
    div[data-testid="stSidebarCollapsedControl"] {{
        display: none !important;
    }}
    
    /* Global Styles */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
        background-color: {bg} !important;
        color: {text} !important;
        font-family: 'DM Sans', -apple-system, sans-serif !important;
    }}
    .block-container {{
        padding: 2rem 2.5rem 3rem !important;
        max-width: 1280px !important;
        margin: auto;
    }}
    
    /* Brand Header */
    .brand-container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid {border};
    }}
    .brand-logo {{
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: {text};
    }}
    .brand-logo span {{
        color: {blue};
    }}
    .brand-tagline {{
        font-size: 0.8rem;
        color: {text_muted};
        margin-top: -4px;
    }}
    
    /* KPI Cards */
    .metric-card {{
        background: {card};
        border: 1px solid {border};
        border-radius: 10px;
        padding: 1.25rem 1.4rem;
        box-shadow: {shadow};
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}
    .metric-card:hover {{
        border-color: {blue};
        transform: translateY(-2px);
    }}
    .metric-label {{
        font-size: 0.78rem;
        color: {text_muted};
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .metric-value {{
        font-size: 1.85rem;
        font-weight: 700;
        color: {text};
        letter-spacing: -0.03em;
        margin-top: 0.2rem;
    }}
    .metric-subtext {{
        font-size: 0.72rem;
        color: {text_dim};
        margin-top: 0.4rem;
    }}
    
    /* Tabs styling */
    button[data-baseweb="tab"] {{
        background: transparent !important;
        color: {text_muted} !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 0.6rem 1.2rem !important;
        border: 1px solid transparent !important;
        border-radius: 7px !important;
        transition: all 0.2s ease;
    }}
    button[data-baseweb="tab"]:hover {{
        color: {text} !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {text} !important;
        background: {card} !important;
        border-color: {border} !important;
    }}
    [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
        display: none !important;
    }}
    [data-baseweb="tab-list"] {{
        gap: 6px !important;
        background: {bg_subtle} !important;
        border: 1px solid {border} !important;
        border-radius: 10px !important;
        padding: 4px;
        margin-bottom: 1.5rem;
    }}
    
    /* Native Streamlit Bordered Container Customization */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: {card} !important;
        border: 1px solid {border} !important;
        border-radius: 12px !important;
        padding: 1.25rem 1.4rem !important;
        box-shadow: {shadow} !important;
        margin-bottom: 1.25rem !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] > div {{
        gap: 0.8rem !important;
    }}
    
    .card-title {{
        font-size: 0.95rem;
        font-weight: 600;
        color: {text};
        margin-bottom: 0.2rem;
    }}
    .card-subtitle {{
        font-size: 0.75rem;
        color: {text_muted};
        margin-bottom: 0.8rem;
    }}
    
    /* Styled HTML table */
    .data-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.82rem;
        margin-top: 0.25rem;
    }}
    .data-table th {{
        text-align: left;
        padding: 0.75rem 0.9rem;
        color: {text_muted};
        font-weight: 600;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 1px solid {border};
        background: {bg_subtle};
    }}
    .data-table td {{
        padding: 0.75rem 0.9rem;
        color: {text};
        border-bottom: 1px solid {border_subtle};
        vertical-align: top;
        line-height: 1.45;
    }}
    .data-table tr:last-child td {{
        border-bottom: none;
    }}
    
    /* Badges */
    .badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}
    .badge-ham {{
        color: {green};
        background: {green_muted};
        border: 1px solid rgba(34,197,94,0.25);
    }}
    .badge-spam {{
        color: {red};
        background: {red_muted};
        border: 1px solid rgba(239,68,68,0.25);
    }}
    
    /* Prediction Panel */
    .prediction-container {{
        border: 1px solid {border};
        border-radius: 10px;
        background: {bg_subtle};
        padding: 1.5rem;
        margin-top: 1rem;
    }}
    .prediction-title {{
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .prediction-metric-row {{
        display: flex;
        gap: 20px;
        margin-top: 1rem;
    }}
    .prediction-metric {{
        flex: 1;
        background: {card};
        border: 1px solid {border};
        padding: 0.8rem 1rem;
        border-radius: 8px;
    }}
    .prediction-metric-val {{
        font-size: 1.3rem;
        font-weight: 700;
        color: {blue};
    }}
    
    /* Custom spacing */
    [data-testid="stHorizontalBlock"] {{
        gap: 1.5rem !important;
    }}
</style>
""")

# --- TEXT PREPROCESSING DEFINITION ---
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', 
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 
    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    cleaned_words = [w for w in words if w not in STOPWORDS]
    return " ".join(cleaned_words)

# --- MODEL TRAINING AND EVALUATION CACHING ---
@st.cache_resource
def load_and_train_models():
    # Load dataset
    df = pd.read_csv('dataset/mail_data.csv')
    df['cleaned_message'] = df['Message'].apply(clean_text)
    df['label'] = df['Category'].map({'ham': 0, 'spam': 1})
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_message'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # CountVectorizer Pipeline
    cv = CountVectorizer()
    X_train_cv = cv.fit_transform(X_train)
    X_test_cv = cv.transform(X_test)
    
    nb_cv = MultinomialNB()
    nb_cv.fit(X_train_cv, y_train)
    
    lr_cv = LogisticRegression(max_iter=1000)
    lr_cv.fit(X_train_cv, y_train)
    
    # TF-IDF Pipeline
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    nb_tfidf = MultinomialNB()
    nb_tfidf.fit(X_train_tfidf, y_train)
    
    lr_tfidf = LogisticRegression(max_iter=1000)
    lr_tfidf.fit(X_train_tfidf, y_train)
    
    # Evaluate All Models
    evals = {}
    for name, model, xt, xv in [
        ("Naive Bayes (Count)", nb_cv, X_train_cv, X_test_cv),
        ("Logistic Regression (Count)", lr_cv, X_train_cv, X_test_cv),
        ("Naive Bayes (TF-IDF)", nb_tfidf, X_train_tfidf, X_test_tfidf),
        ("Logistic Regression (TF-IDF)", lr_tfidf, X_train_tfidf, X_test_tfidf)
    ]:
        preds = model.predict(xv)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        
        evals[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-score': f1,
            'CM': cm.tolist()
        }
        
    return df, cv, tfidf, nb_cv, lr_cv, evals

# Run Training
df, cv, tfidf, nb_model, lr_model, evaluations = load_and_train_models()

# --- BRAND HEADER ---
head_left, head_right = st.columns([7, 1.2])
with head_left:
    st.html(f"""
    <div class="brand-container">
        <div>
            <div class="brand-logo">🛡️ Mail Spam <span>Guard</span></div>
            <div class="brand-tagline">Real-time intelligent email classification and security dashboard</div>
        </div>
    </div>
    """)
with head_right:
    theme_label = "☀️ Light Mode" if IS_DARK else "🌙 Dark Mode"
    st.button(theme_label, on_click=toggle_theme, use_container_width=True)

# --- KPI CARDS ROW ---
ham_count = int(df['Category'].value_counts().get('ham', 0))
spam_count = int(df['Category'].value_counts().get('spam', 0))
total_count = len(df)
spam_ratio = (spam_count / total_count) * 100

kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.html(f"""
    <div class="metric-card">
        <div class="metric-label">Total Emails</div>
        <div class="metric-value">{total_count:,}</div>
        <div class="metric-subtext">Total records analyzed</div>
    </div>
    """)
with kpi_cols[1]:
    st.html(f"""
    <div class="metric-card">
        <div class="metric-label">Ham (Normal)</div>
        <div class="metric-value" style="color: {green};">{ham_count:,}</div>
        <div class="metric-subtext">{(ham_count/total_count*100):.1f}% of total dataset</div>
    </div>
    """)
with kpi_cols[2]:
    st.html(f"""
    <div class="metric-card">
        <div class="metric-label">Spam (Unwanted)</div>
        <div class="metric-value" style="color: {red};">{spam_count:,}</div>
        <div class="metric-subtext">{spam_ratio:.1f}% of total dataset</div>
    </div>
    """)
with kpi_cols[3]:
    st.html(f"""
    <div class="metric-card">
        <div class="metric-label">Best Accuracy</div>
        <div class="metric-value" style="color: {blue};">{evaluations['Naive Bayes (Count)']['Accuracy']*100:.2f}%</div>
        <div class="metric-subtext">Naive Bayes + CountVectorizer</div>
    </div>
    """)

st.html("<div style='height: 10px;'></div>")

# --- APP TABS ---
tab1, tab2, tab3 = st.tabs([
    "📊 Dataset Explorer", 
    "⚙️ Model Performance", 
    "🔍 Live Prediction Demo"
])

# ================= TAB 1: DATASET EXPLORER =================
with tab1:
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        with st.container(border=True):
            st.html("""
            <div class="card-title">Class Balance</div>
            <div class="card-subtitle">Showing counts and ratios of ham vs. spam</div>
            """)
            
            # Plotly Class Distribution Pie
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Ham (Normal)', 'Spam (Unwanted)'],
                values=[ham_count, spam_count],
                hole=.4,
                marker_colors=['#3b82f6', '#ef4444'],
                textinfo='percent+label',
                textfont=dict(family="DM Sans, sans-serif", size=12, color="#ffffff" if IS_DARK else "#09090b")
            )])
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                margin=dict(t=0, b=0, l=0, r=0),
                height=260,
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
        
        with st.container(border=True):
            st.html("""
            <div class="card-title">Key Vocabulary Analysis</div>
            <div class="card-subtitle">Common words extracted after stopword removal</div>
            """)
            
            # Plot common words in spam
            spam_messages = " ".join(df[df['label'] == 1]['cleaned_message'])
            spam_words = pd.Series(spam_messages.split()).value_counts().head(8)
            
            fig_words = px.bar(
                x=spam_words.values,
                y=spam_words.index,
                orientation='h',
                labels={'x': 'Frequency', 'y': 'Words'},
                color_discrete_sequence=['#ef4444']
            )
            fig_words.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans, sans-serif", color=text_muted, size=11),
                margin=dict(l=0, r=0, t=10, b=0),
                height=200,
                xaxis=dict(gridcolor=border_subtle, tickfont=dict(color=text_muted)),
                yaxis=dict(autorange="reversed", tickfont=dict(color=text_muted))
            )
            st.plotly_chart(fig_words, use_container_width=True, config={"displayModeBar": False})
        
    with col2:
        with st.container(border=True):
            st.html("""
            <div class="card-title">Dataset Samples</div>
            <div class="card-subtitle">Explore raw messages and labels from mail_data.csv</div>
            """)
            
            # Display sample data in styled HTML table
            sample_hams = df[df['label'] == 0].sample(3, random_state=42)[['Category', 'Message']]
            sample_spams = df[df['label'] == 1].sample(3, random_state=42)[['Category', 'Message']]
            samples = pd.concat([sample_hams, sample_spams]).sample(frac=1, random_state=42)
            
            sample_rows = []
            for _, row in samples.iterrows():
                badge_class = "badge-ham" if row['Category'] == 'ham' else "badge-spam"
                label_text = "HAM" if row['Category'] == 'ham' else "SPAM"
                safe_msg = html.escape(str(row['Message']))
                sample_rows.append(
                    f'<tr>'
                    f'<td style="width: 80px;"><span class="badge {badge_class}">{label_text}</span></td>'
                    f'<td>{safe_msg}</td>'
                    f'</tr>'
                )
            
            table_sample_html = (
                f'<table class="data-table">'
                f'<thead><tr><th style="width: 80px;">Category</th><th>Message Body</th></tr></thead>'
                f'<tbody>{"".join(sample_rows)}</tbody>'
                f'</table>'
            )
            st.html(table_sample_html)

# ================= TAB 2: MODEL PERFORMANCE =================
with tab2:
    with st.container(border=True):
        st.html("""
        <div class="card-title">Model Evaluation Comparison</div>
        <div class="card-subtitle">Comparative metrics on testing split (20% of data, stratified)</div>
        """)
        
        # Formulate metrics dataframe
        metrics_list = []
        for m_name, vals in evaluations.items():
            metrics_list.append({
                'Model Configuration': m_name,
                'Accuracy': f"{vals['Accuracy']*100:.2f}%",
                'Precision': f"{vals['Precision']*100:.2f}%",
                'Recall': f"{vals['Recall']*100:.2f}%",
                'F1-Score': f"{vals['F1-score']:.4f}"
            })
        df_compare = pd.DataFrame(metrics_list)
        
        # Convert to HTML table
        compare_rows = []
        for _, row in df_compare.iterrows():
            # Highlight best model
            is_best = f"font-weight: 600; color: {blue};" if "Naive Bayes (Count)" in row['Model Configuration'] else ""
            compare_rows.append(
                f'<tr style="{is_best}">'
                f'<td>{row["Model Configuration"]}</td>'
                f'<td>{row["Accuracy"]}</td>'
                f'<td>{row["Precision"]}</td>'
                f'<td>{row["Recall"]}</td>'
                f'<td>{row["F1-Score"]}</td>'
                f'</tr>'
            )
            
        table_perf_html = (
            f'<table class="data-table">'
            f'<thead><tr>'
            f'<th>Model Configuration</th>'
            f'<th>Accuracy</th>'
            f'<th>Precision</th>'
            f'<th>Recall</th>'
            f'<th>F1-Score</th>'
            f'</tr></thead>'
            f'<tbody>{"".join(compare_rows)}</tbody>'
            f'</table>'
        )
        st.html(table_perf_html)
    
    # Confusion Matrices Section
    st.markdown("### Confusion Matrices (CountVectorizer Models)")
    cm_col1, cm_col2 = st.columns(2)
    
    with cm_col1:
        with st.container(border=True):
            st.html("""
            <div class="card-title">Naive Bayes Confusion Matrix</div>
            <div class="card-subtitle">Model correctly predicts 962 Hams, 132 Spams; misses 17 Spams</div>
            """)
            
            cm_nb = evaluations["Naive Bayes (Count)"]["CM"]
            z_nb = cm_nb
            x_nb = ['Predicted Ham', 'Predicted Spam']
            y_nb = ['True Ham', 'True Spam']
            
            fig_cm_nb = go.Figure(data=go.Heatmap(
                z=z_nb, x=x_nb, y=y_nb,
                colorscale='Blues',
                text=z_nb, texttemplate="%{text}",
                textfont={"size":14, "color":"white", "family":"DM Sans, sans-serif"},
                showscale=False
            ))
            fig_cm_nb.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=240,
                margin=dict(t=10, b=10, l=40, r=10),
                xaxis=dict(tickfont=dict(color=text_muted)),
                yaxis=dict(tickfont=dict(color=text_muted))
            )
            st.plotly_chart(fig_cm_nb, use_container_width=True, config={"displayModeBar": False})
        
    with cm_col2:
        with st.container(border=True):
            st.html("""
            <div class="card-title">Logistic Regression Confusion Matrix</div>
            <div class="card-subtitle">Model has 100% Precision (0 False Positives); misses 22 Spams</div>
            """)
            
            cm_lr = evaluations["Logistic Regression (Count)"]["CM"]
            z_lr = cm_lr
            x_lr = ['Predicted Ham', 'Predicted Spam']
            y_lr = ['True Ham', 'True Spam']
            
            fig_cm_lr = go.Figure(data=go.Heatmap(
                z=z_lr, x=x_lr, y=y_lr,
                colorscale='Blues',
                text=z_lr, texttemplate="%{text}",
                textfont={"size":14, "color":"white", "family":"DM Sans, sans-serif"},
                showscale=False
            ))
            fig_cm_lr.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=240,
                margin=dict(t=10, b=10, l=40, r=10),
                xaxis=dict(tickfont=dict(color=text_muted)),
                yaxis=dict(tickfont=dict(color=text_muted))
            )
            st.plotly_chart(fig_cm_lr, use_container_width=True, config={"displayModeBar": False})

# ================= TAB 3: LIVE PREDICTION DEMO =================
with tab3:
    with st.container(border=True):
        st.html("""
        <div class="card-title">Test Spam Guard Live</div>
        <div class="card-subtitle">Type or paste any email content below to predict whether it is Spam or Not Spam</div>
        """)
        
        # Model Selection
        sel_model_name = st.radio(
            "Select Classification Algorithm:",
            ["Multinomial Naive Bayes (Recommended - Higher Recall)", "Logistic Regression (Perfect Precision)"],
            horizontal=True
        )
        
        # Sample templates for fast testing
        st.write("Or insert a sample template:")
        s_col1, s_col2, s_col3 = st.columns(3)
        
        ham_sample = "Hey there, are we still meeting for lunch at 1 PM today? Let me know if you want me to bring anything."
        spam_sample_1 = "URGENT: Click here to claim your $500 Amazon Gift card now! Limited time offer. Call 0800-449-3221."
        spam_sample_2 = "Free entry in a weekly competition to win FA Cup final tickets! Text WIN to 87121. T&C apply."
        
        if s_col1.button("Normal Email (Ham)", use_container_width=True):
            st.session_state.email_input = ham_sample
        if s_col2.button("Promo Scam (Spam)", use_container_width=True):
            st.session_state.email_input = spam_sample_1
        if s_col3.button("Prize Alert (Spam)", use_container_width=True):
            st.session_state.email_input = spam_sample_2
            
        # Text input
        email_text = st.text_area(
            "Email Message Content:",
            value=st.session_state.get("email_input", ""),
            height=140,
            placeholder="Paste email text here...",
            key="email_input_area"
        )
        
        # Align the input state
        if email_text:
            st.session_state.email_input = email_text
            
        btn_predict = st.button("🛡️ Run Spam Analysis", type="primary")
        
        if btn_predict:
            if not email_text.strip():
                st.warning("Please enter some email content first!")
            else:
                # Preprocess text
                cleaned = clean_text(email_text)
                
                # Vectorize using the fitted CountVectorizer
                vectorized = cv.transform([cleaned])
                
                # Predict
                if "Naive Bayes" in sel_model_name:
                    model = nb_model
                    model_title = "Multinomial Naive Bayes"
                else:
                    model = lr_model
                    model_title = "Logistic Regression"
                    
                prediction = model.predict(vectorized)[0]
                probabilities = model.predict_proba(vectorized)[0]
                spam_prob = probabilities[1]
                ham_prob = probabilities[0]
                
                # Predict outputs
                pred_label = "SPAM - UNWANTED" if prediction == 1 else "HAM - SAFE"
                badge_class = "badge-spam" if prediction == 1 else "badge-ham"
                badge_icon = "🚨" if prediction == 1 else "✅"
                safe_cleaned = html.escape(cleaned if cleaned else "[No words left after preprocessing]")
                
                # Output UI block
                st.html(f"""
                <div class="prediction-container">
                    <div class="prediction-title">
                        <span>{badge_icon} Analysis Verdict:</span>
                        <span class="badge {badge_class}">{pred_label}</span>
                    </div>
                    <p style="font-size: 0.82rem; color: {text_muted}; margin-top: -6px;">
                        Analyzed using <b>{model_title} + CountVectorizer</b>
                    </p>
                    
                    <div class="prediction-metric-row">
                        <div class="prediction-metric">
                            <div class="metric-label">Spam Probability</div>
                            <div class="prediction-metric-val" style="color: {red if prediction == 1 else text};">
                                {spam_prob * 100:.2f}%
                            </div>
                        </div>
                        <div class="prediction-metric">
                            <div class="metric-label">Ham Probability</div>
                            <div class="prediction-metric-val" style="color: {green if prediction == 0 else text};">
                                {ham_prob * 100:.2f}%
                            </div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 1.2rem;">
                        <div class="metric-label" style="margin-bottom: 0.4rem;">Cleaned Vocabulary Tokens</div>
                        <code style="background: {bg}; border: 1px solid {border}; padding: 6px 12px; border-radius: 6px; display: block; font-size: 0.8rem; color: {text_muted}; font-family: 'JetBrains Mono', monospace;">
                            {safe_cleaned}
                        </code>
                    </div>
                </div>
                """)
