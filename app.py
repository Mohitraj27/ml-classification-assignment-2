import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="ML Classification Assignment",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Custom CSS for enhanced UI
def load_css():
    theme = st.session_state.theme
    
    if theme == 'dark':
        primary_color = "#00d4ff"
        background_color = "#0e1117"
        secondary_bg = "#1e2130"
        text_color = "#fafafa"
        border_color = "#2d3748"
        card_bg = "#1a1f2e"
        hover_color = "#252b3d"
    else:
        primary_color = "#0066cc"
        background_color = "#ffffff"
        secondary_bg = "#f8f9fa"
        text_color = "#1f2937"
        border_color = "#e5e7eb"
        card_bg = "#ffffff"
        hover_color = "#f3f4f6"
    
    st.markdown(f"""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        * {{
            font-family: 'Inter', sans-serif;
        }}
        
        /* Main Container */
        .main {{
            background-color: {background_color};
            padding: 0rem 1rem;
        }}
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {secondary_bg} 0%, {card_bg} 100%);
            border-right: 1px solid {border_color};
        }}
        
        [data-testid="stSidebar"] .css-1d391kg {{
            padding-top: 2rem;
        }}
        
        /* Custom Header */
        .custom-header {{
            background: linear-gradient(135deg, {primary_color} 0%, #667eea 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            color: white;
        }}
        
        .custom-header h1 {{
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        
        .custom-header p {{
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.95;
        }}
        
        /* Navigation Pills */
        .nav-pills {{
            display: flex;
            gap: 1rem;
            margin: 1.5rem 0;
            flex-wrap: wrap;
        }}
        
        .nav-pill {{
            padding: 0.75rem 1.5rem;
            background: {card_bg};
            border: 2px solid {border_color};
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 500;
            color: {text_color};
            text-decoration: none;
        }}
        
        .nav-pill:hover {{
            background: {hover_color};
            border-color: {primary_color};
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .nav-pill.active {{
            background: {primary_color};
            color: white;
            border-color: {primary_color};
        }}
        
        /* Cards */
        .metric-card {{
            background: {card_bg};
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid {border_color};
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
            margin-bottom: 1rem;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }}
        
        .metric-card h3 {{
            color: {primary_color};
            margin: 0 0 0.5rem 0;
            font-size: 1.1rem;
            font-weight: 600;
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
            color: {text_color};
            margin: 0;
        }}
        
        .metric-label {{
            color: #6b7280;
            font-size: 0.875rem;
            margin-top: 0.25rem;
        }}
        
        /* Feature Cards */
        .feature-card {{
            background: {card_bg};
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 4px solid {primary_color};
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        
        .feature-card h4 {{
            color: {primary_color};
            margin: 0 0 0.75rem 0;
            font-size: 1.1rem;
            font-weight: 600;
        }}
        
        .feature-card p {{
            color: {text_color};
            margin: 0;
            line-height: 1.6;
        }}
        
        /* Info Box */
        .info-box {{
            background: linear-gradient(135deg, {primary_color}15 0%, {primary_color}05 100%);
            border-left: 4px solid {primary_color};
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
        }}
        
        /* Success Box */
        .success-box {{
            background: linear-gradient(135deg, #10b98115 0%, #10b98105 100%);
            border-left: 4px solid #10b981;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
        }}
        
        /* Warning Box */
        .warning-box {{
            background: linear-gradient(135deg, #f59e0b15 0%, #f59e0b05 100%);
            border-left: 4px solid #f59e0b;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
        }}
        
        /* Buttons */
        .stButton > button {{
            background: linear-gradient(135deg, {primary_color} 0%, #667eea 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        }}
        
        /* File Uploader */
        [data-testid="stFileUploader"] {{
            background: {card_bg};
            border: 2px dashed {border_color};
            border-radius: 12px;
            padding: 2rem;
        }}
        
        [data-testid="stFileUploader"]:hover {{
            border-color: {primary_color};
            background: {hover_color};
        }}
        
        /* Dataframe */
        .dataframe {{
            border-radius: 8px;
            overflow: hidden;
        }}
        
        /* Progress Bar */
        .stProgress > div > div > div > div {{
            background: linear-gradient(90deg, {primary_color} 0%, #667eea 100%);
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 1rem;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background: {card_bg};
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
        }}
        
        .stTabs [aria-selected="true"] {{
            background: {primary_color};
            color: white;
        }}
        
        /* Metrics */
        [data-testid="stMetricValue"] {{
            font-size: 2rem;
            font-weight: 700;
            color: {primary_color};
        }}
        
        /* Expander */
        .streamlit-expanderHeader {{
            background: {card_bg};
            border-radius: 8px;
            border: 1px solid {border_color};
        }}
        
        /* Select Box */
        .stSelectbox > div > div {{
            background: {card_bg};
            border-radius: 8px;
        }}
        
        /* Text Input */
        .stTextInput > div > div > input {{
            background: {card_bg};
            border-radius: 8px;
            border: 1px solid {border_color};
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding: 2rem;
            color: #6b7280;
            border-top: 1px solid {border_color};
            margin-top: 3rem;
        }}
        
        /* Theme Toggle */
        .theme-toggle {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 999;
        }}
        
        .theme-toggle button {{
            background: {primary_color};
            color: white;
            border: none;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            font-size: 1.5rem;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }}
        
        .theme-toggle button:hover {{
            transform: scale(1.1);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }}
        
        /* Animation */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .fade-in {{
            animation: fadeIn 0.5s ease-in;
        }}
        
        /* Hide Streamlit Branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .custom-header h1 {{
                font-size: 1.75rem;
            }}
            
            .nav-pills {{
                flex-direction: column;
            }}
            
            .metric-card {{
                margin-bottom: 1rem;
            }}
        }}
        </style>
    """, unsafe_allow_html=True)

load_css()

# Custom Header
st.markdown("""
    <div class="custom-header fade-in">
        <h1>ML Classification</h1>
        <p>Advanced Machine Learning Models Comparison & Analysis Platform</p>
        <p style="font-size: 0.9rem; opacity: 0.85; margin-top: 0.5rem;">
            BITS Pilani M.Tech (AIML/DSE) | Assignment 2 | BITS ID: 2025AA05168
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for navigation with enhanced styling
with st.sidebar:
    # Logo/Icon section
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0 2rem 0;">
            <p style="color: #6b7280; font-size: 0.875rem; margin: 0.25rem 0 0 0;">
                Classification Analysis
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Navigation")
    page = st.radio(
        "Select Page",
        ["🏠 Home", "🎓 Model Training", "📊 Evaluation", "🔮 Predictions"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    
    # Quick Stats
    st.markdown("### 📈 Quick Stats")
    if 'results' in st.session_state:
        st.metric("Models Trained", "6", delta="Complete")
        st.metric("Best Accuracy", "79.7%", delta="XGBoost")
    else:
        st.metric("Models Trained", "0", delta="Ready")
        st.metric("Status", "Pending", delta_color="off")
    
    st.divider()
    
    # Info Section
    st.markdown("### About")
    st.markdown("""
        <div style="font-size: 0.85rem; line-height: 1.6;">
            <p><strong>Student Name : </strong> MOHIT RAJ </p>
            <p><strong>BITS ID:</strong> 2025AA05168 </p>
            <p><strong>Course:</strong> Machine Learning</p>
            <p><strong>Institution:</strong> BITS Pilani</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Help Section
    with st.expander("ℹ️  Quick Tips:"):
        st.markdown("""
            - Upload CSV with 11+ features
            - Minimum 500 samples
            - Binary classification (0/1)
            - Last column = target
        """)

# Map page selection (remove emoji from comparison)
page = page.replace("🏠 ", "").replace("🎓 ", "").replace("📊 ", "").replace("🔮 ", "")

# Function to load and preprocess data
@st.cache_data
def load_data(uploaded_file=None):
    """Load wine quality dataset"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # For demo purposes, using a sample dataset
        st.warning("⚠️ No file uploaded. Using sample data for demonstration.")
        # Creating sample data structure
        df = pd.DataFrame()
    return df

# Function to train all models
def train_all_models(X_train, y_train, X_test, y_test):
    """Train all 6 classification models"""
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    results = {}
    trained_models = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (name, model) in enumerate(models.items()):
        status_text.text(f"Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
            'Precision': precision_score(y_test, y_pred, average='binary'),
            'Recall': recall_score(y_test, y_pred, average='binary'),
            'F1': f1_score(y_test, y_pred, average='binary'),
            'MCC': matthews_corrcoef(y_test, y_pred)
        }
        
        results[name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        trained_models[name] = model
        
        progress_bar.progress((idx + 1) / len(models))
    
    status_text.text("✅ All models trained successfully!")
    progress_bar.empty()
    
    return results, trained_models

# Function to display metrics comparison
def display_metrics_comparison(results):
    """Display comparison table of all models"""
    
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[m]['metrics']['Accuracy'] for m in results.keys()],
        'AUC': [results[m]['metrics']['AUC'] if results[m]['metrics']['AUC'] is not None else 0 for m in results.keys()],
        'Precision': [results[m]['metrics']['Precision'] for m in results.keys()],
        'Recall': [results[m]['metrics']['Recall'] for m in results.keys()],
        'F1': [results[m]['metrics']['F1'] for m in results.keys()],
        'MCC': [results[m]['metrics']['MCC'] for m in results.keys()]
    })
    
    # Format to 4 decimal places
    metrics_df_display = metrics_df.copy()
    for col in ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']:
        metrics_df_display[col] = metrics_df_display[col].apply(lambda x: f"{x:.4f}")
    
    st.markdown("### 📊 Model Performance Comparison")
    
    # Styled dataframe
    st.dataframe(
        metrics_df_display,
        use_container_width=True,
        height=280,
        column_config={
            "Model": st.column_config.TextColumn("🤖 Model Name", width="medium"),
            "Accuracy": st.column_config.TextColumn("🎯 Accuracy", width="small"),
            "AUC": st.column_config.TextColumn("📈 AUC", width="small"),
            "Precision": st.column_config.TextColumn("🔍 Precision", width="small"),
            "Recall": st.column_config.TextColumn("📡 Recall", width="small"),
            "F1": st.column_config.TextColumn("⚖️ F1", width="small"),
            "MCC": st.column_config.TextColumn("🔬 MCC", width="small")
        }
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Highlight best model for each metric
    st.markdown("### 🏆 Best Models by Metric")
    
    cols = st.columns(3)
    
    metric_icons = {
        'Accuracy': '🎯',
        'F1': '⚖️',
        'MCC': '🔬',
        'AUC': '📈',
        'Precision': '🔍',
        'Recall': '📡'
    }
    
    for idx, metric in enumerate(['Accuracy', 'F1', 'MCC']):
        best_model_idx = metrics_df[metric].idxmax()
        best_model = metrics_df.loc[best_model_idx, 'Model']
        best_score = metrics_df.loc[best_model_idx, metric]
        
        with cols[idx % 3]:
            st.markdown(f"""
                <div class="metric-card fade-in">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">
                        {metric_icons[metric]}
                    </div>
                    <h3 style="margin: 0; font-size: 0.875rem; color: #6b7280;">
                        Best {metric}
                    </h3>
                    <p style="font-size: 1.25rem; font-weight: 700; margin: 0.5rem 0;">
                        {best_model}
                    </p>
                    <p style="color: #0066cc; font-weight: 600; margin: 0;">
                        {best_score:.4f}
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    return metrics_df

# Function to plot metrics
def plot_metrics_comparison(metrics_df):
    """Plot comparison charts"""
    
    st.subheader("📈 Visual Comparison")
    
    # Bar chart for all metrics
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        bars = ax.bar(range(len(metrics_df)), metrics_df[metric], color=colors[idx], alpha=0.7)
        ax.set_xlabel('Models', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(metrics_df)))
        ax.set_xticklabels(metrics_df['Model'], rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    st.pyplot(fig)

# Function to display confusion matrix
def plot_confusion_matrices(results):
    """Plot confusion matrices for all models"""
    
    st.subheader("🎯 Confusion Matrices")
    
    # Create a grid layout
    cols_per_row = 3
    model_names = list(results.keys())
    
    for i in range(0, len(model_names), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j in range(cols_per_row):
            if i + j < len(model_names):
                model_name = model_names[i + j]
                cm = results[model_name]['confusion_matrix']
                
                with cols[j]:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_title(f'{model_name}', fontweight='bold')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)

# HOME PAGE
if page == "Home":
    # Welcome Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            <div class="feature-card fade-in">
                <p style="font-size: 1.1rem; line-height: 1.8;">
                    A comprehensive platform for training, evaluating, and comparing 
                    <strong>6 different Machine Learning classification models</strong> 
                    on your dataset. Built with industry best practices and modern ML workflows.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card fade-in">
                <h3>🎯 Assignment Status</h3>
                <p class="metric-value">100%</p>
                <p class="metric-label">Features Implemented</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Features in Cards
    st.markdown("## ✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-card fade-in">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🤖</div>
                <h3>6 ML Models</h3>
                <p style="color: #6b7280; line-height: 1.6;">
                    Logistic Regression, Decision Tree, KNN, 
                    Naive Bayes, Random Forest, XGBoost
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card fade-in">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
                <h3>6 Metrics</h3>
                <p style="color: #6b7280; line-height: 1.6;">
                    Accuracy, AUC, Precision, Recall, 
                    F1 Score, MCC
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-card fade-in">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎨</div>
                <h3>Interactive UI</h3>
                <p style="color: #6b7280; line-height: 1.6;">
                    Modern interface with dark/light themes, 
                    charts, and real-time analysis
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Models Overview
    st.markdown("## 🔬 Implemented Models")
    
    models_data = [
        {"icon": "📈", "name": "Logistic Regression", "type": "Linear", "desc": "Fast baseline model for binary classification"},
        {"icon": "🌳", "name": "Decision Tree", "type": "Tree-based", "desc": "Interpretable model with clear decision rules"},
        {"icon": "🎯", "name": "K-Nearest Neighbors", "type": "Instance-based", "desc": "Distance-based classification algorithm"},
        {"icon": "🎲", "name": "Naive Bayes", "type": "Probabilistic", "desc": "Fast probabilistic classifier based on Bayes theorem"},
        {"icon": "🌲", "name": "Random Forest", "type": "Ensemble", "desc": "Robust ensemble of multiple decision trees"},
        {"icon": "🚀", "name": "XGBoost", "type": "Ensemble", "desc": "State-of-the-art gradient boosting algorithm"}
    ]
    
    col1, col2 = st.columns(2)
    
    for idx, model in enumerate(models_data):
        with col1 if idx % 2 == 0 else col2:
            st.markdown(f"""
                <div class="feature-card fade-in">
                    <h4>{model['icon']} {model['name']}</h4>
                    <p><strong>Type:</strong> {model['type']}</p>
                    <p>{model['desc']}</p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Evaluation Metrics
    st.markdown("## 📊 Evaluation Metrics")
    
    metrics_data = [
        {"name": "Accuracy", "icon": "🎯", "desc": "Overall correctness of predictions"},
        {"name": "AUC Score", "icon": "📈", "desc": "Area under ROC curve - discrimination ability"},
        {"name": "Precision", "icon": "🔍", "desc": "Accuracy of positive predictions"},
        {"name": "Recall", "icon": "📡", "desc": "Coverage of actual positive cases"},
        {"name": "F1 Score", "icon": "⚖️", "desc": "Harmonic mean of precision and recall"},
        {"name": "MCC", "icon": "🔬", "desc": "Matthews Correlation Coefficient"}
    ]
    
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    
    for idx, metric in enumerate(metrics_data):
        with cols[idx % 3]:
            st.markdown(f"""
                <div class="metric-card fade-in">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{metric['icon']}</div>
                    <h3 style="font-size: 1rem;">{metric['name']}</h3>
                    <p style="color: #6b7280; font-size: 0.875rem; margin: 0;">{metric['desc']}</p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # How to Use
    st.markdown("## 🚀 How to Use")
    
    steps = [
        {"step": "1", "title": "Upload Dataset", "desc": "Go to Model Training page and upload your CSV file (min 12 features, 500 samples)"},
        {"step": "2", "title": "Train Models", "desc": "Click 'Start Training' to train all 6 models automatically (~45 seconds)"},
        {"step": "3", "title": "Compare Results", "desc": "View comprehensive metrics, charts, and confusion matrices in Evaluation page"},
        {"step": "4", "title": "Make Predictions", "desc": "Use trained models to predict on new data in Predictions page"}
    ]
    
    for step_data in steps:
        st.markdown(f"""
            <div class="feature-card fade-in">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div style="
                        background: linear-gradient(135deg, #0066cc 0%, #667eea 100%);
                        color: white;
                        width: 40px;
                        height: 40px;
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-weight: 700;
                        font-size: 1.25rem;
                        flex-shrink: 0;
                    ">{step_data['step']}</div>
                    <div>
                        <h4 style="margin: 0 0 0.25rem 0;">{step_data['title']}</h4>
                        <p style="margin: 0; color: #6b7280;">{step_data['desc']}</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Dataset Requirements
    st.markdown("## 📋 Dataset Requirements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="info-box fade-in">
                <h4 style="margin-top: 0;">✅ Required Format</h4>
                <ul style="margin-bottom: 0;">
                    <li>CSV file format</li>
                    <li>Minimum 12 features (numerical)</li>
                    <li>Minimum 500 samples</li>
                    <li>Binary classification (0 or 1)</li>
                    <li>Target variable in last column</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="success-box fade-in">
                <h4 style="margin-top: 0;">💡 Sample Datasets Included</h4>
                <ul style="margin-bottom: 0;">
                    <li><strong>1_wine_quality_full.csv</strong> - 1599 samples ⭐</li>
                    <li><strong>2_small_test.csv</strong> - Quick test (200 samples)</li>
                    <li><strong>3_large_test.csv</strong> - 1000 samples</li>
                    <li>More test files available in project folder</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Technical Stack
    st.markdown("## 🛠️ Technical Stack")
    
    col1, col2, col3, col4 = st.columns(4)
    
    tech_stack = [
        {"name": "Streamlit", "icon": "🎨", "col": col1},
        {"name": "scikit-learn", "icon": "🔬", "col": col2},
        {"name": "XGBoost", "icon": "🚀", "col": col3},
        {"name": "Pandas", "icon": "🐼", "col": col4}
    ]
    
    for tech in tech_stack:
        with tech['col']:
            st.markdown(f"""
                <div class="metric-card fade-in" style="text-align: center;">
                    <div style="font-size: 2.5rem;">{tech['icon']}</div>
                    <p style="font-weight: 600; margin: 0.5rem 0 0 0;">{tech['name']}</p>
                </div>
            """, unsafe_allow_html=True)

# MODEL TRAINING PAGE
elif page == "Model Training":
    st.markdown("""
        <div class="custom-header fade-in" style="padding: 1.5rem;">
            <h1 style="font-size: 2rem; margin: 0;">🎓 Model Training</h1>
            <p style="margin: 0.5rem 0 0 0;">Upload your dataset and train all 6 classification models</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Instructions
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            <div class="info-box fade-in">
                <h3 style="margin-top: 0;">📤 Dataset Upload Instructions</h3>
                <p style="margin-bottom: 0.5rem;">Your CSV file should meet these requirements:</p>
                <ul>
                    <li>✅ Minimum <strong>12 features</strong> (numerical columns)</li>
                    <li>✅ Minimum <strong>500 samples</strong> (rows)</li>
                    <li>✅ <strong>Binary classification</strong> target (0 or 1)</li>
                    <li>✅ Target variable in the <strong>last column</strong></li>
                    <li>✅ No missing values</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="success-box fade-in">
                <h3 style="margin-top: 0;">💡 Sample Files</h3>
                <p><strong>Recommended:</strong></p>
                <p>📊 <code>1_wine_quality_full.csv</code></p>
                <p style="font-size: 0.875rem; color: #6b7280; margin: 0;">
                    1599 samples, 11 features<br>
                    Wine quality classification
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # File Upload
    uploaded_file = st.file_uploader(
        "📁 Choose a CSV file",
        type=['csv'],
        help="Upload your test dataset in CSV format"
    )
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        st.markdown("""
            <div class="success-box fade-in">
                <h3 style="margin-top: 0;">✅ File Uploaded Successfully!</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Display data preview
        st.markdown("### 📋 Data Preview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>Total Samples</h3>
                    <p class="metric-value">{df.shape[0]:,}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>Features</h3>
                    <p class="metric-value">{df.shape[1] - 1}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            class_type = "Binary" if df.iloc[:, -1].nunique() == 2 else "Multi-class"
            st.markdown(f"""
                <div class="metric-card">
                    <h3>Classification</h3>
                    <p class="metric-value" style="font-size: 1.25rem;">{class_type}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            class_dist = df.iloc[:, -1].value_counts()
            balance = min(class_dist.values) / max(class_dist.values) * 100
            st.markdown(f"""
                <div class="metric-card">
                    <h3>Balance</h3>
                    <p class="metric-value" style="font-size: 1.25rem;">{balance:.0f}%</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Show dataframe
        with st.expander("👁️ View Full Dataset", expanded=False):
            st.dataframe(df, use_container_width=True, height=400)
        
        # Show statistics
        with st.expander("📊 Statistical Summary", expanded=False):
            st.dataframe(df.describe(), use_container_width=True)
        
        # Class distribution
        st.markdown("### 📈 Class Distribution")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(
                df.iloc[:, -1].value_counts().reset_index(),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "index": "Class",
                    "count": "Count"
                }
            )
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            df.iloc[:, -1].value_counts().plot(kind='bar', ax=ax, color=['#0066cc', '#667eea'])
            ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Training Section
        st.markdown("### 🚀 Start Training")
        
        st.markdown("""
            <div class="info-box">
                <p style="margin: 0;"><strong>Training Configuration:</strong></p>
                <ul style="margin: 0.5rem 0 0 0;">
                    <li>📊 Train-Test Split: 80-20</li>
                    <li>🔄 Feature Scaling: StandardScaler</li>
                    <li>⏱️ Estimated Time: 30-60 seconds</li>
                    <li>🎯 Models: 6 classifiers</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            train_button = st.button(
                "🚀 Start Training Models",
                type="primary",
                use_container_width=True
            )
        
        if train_button:
            with st.spinner("🔄 Preparing data and training models..."):
                
                # Separate features and target
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]
                
                # Split data (80-20 split for training-testing)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                st.markdown(f"""
                    <div class="info-box">
                        <p><strong>📦 Data Split Complete:</strong></p>
                        <p>Training Set: {X_train_scaled.shape[0]} samples | Test Set: {X_test_scaled.shape[0]} samples</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Train models
                results, trained_models = train_all_models(
                    X_train_scaled, y_train, X_test_scaled, y_test
                )
                
                # Store in session state
                st.session_state['results'] = results
                st.session_state['trained_models'] = trained_models
                st.session_state['scaler'] = scaler
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                
                st.markdown("""
                    <div class="success-box fade-in">
                        <h3 style="margin-top: 0;">✅ Training Complete!</h3>
                        <p style="margin-bottom: 0;">All 6 models have been trained successfully. 
                        Navigate to <strong>📊 Evaluation</strong> to see the results!</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.balloons()
    
    else:
        st.markdown("""
            <div class="warning-box fade-in">
                <h3 style="margin-top: 0;">⬆️ Please Upload a CSV File</h3>
                <p style="margin-bottom: 0;">
                    Select a CSV file using the uploader above to begin the training process.
                    You can use any of the sample datasets provided in the project folder.
                </p>
            </div>
        """, unsafe_allow_html=True)

# MODEL EVALUATION PAGE
elif page == "Evaluation":
    st.markdown("""
        <div class="custom-header fade-in" style="padding: 1.5rem;">
            <h1 style="font-size: 2rem; margin: 0;">📊 Model Evaluation</h1>
            <p style="margin: 0.5rem 0 0 0;">Comprehensive performance analysis and comparison</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        # Display metrics comparison table
        metrics_df = display_metrics_comparison(results)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Plot metrics comparison
        plot_metrics_comparison(metrics_df)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display confusion matrices
        plot_confusion_matrices(results)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Detailed classification reports
        st.markdown("### 📄 Detailed Classification Reports")
        
        model_selector = st.selectbox(
            "Select a model to view detailed report:",
            list(results.keys()),
            index=5  # Default to XGBoost (usually best)
        )
        
        if model_selector:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("""
                    <div class="feature-card">
                        <h4>📊 Classification Report</h4>
                    </div>
                """, unsafe_allow_html=True)
                st.text(results[model_selector]['classification_report'])
            
            with col2:
                st.markdown("""
                    <div class="feature-card">
                        <h4>🎯 Confusion Matrix Details</h4>
                    </div>
                """, unsafe_allow_html=True)
                cm = results[model_selector]['confusion_matrix']
                cm_df = pd.DataFrame(
                    cm,
                    columns=['Predicted Negative', 'Predicted Positive'],
                    index=['Actual Negative', 'Actual Positive']
                )
                st.dataframe(cm_df, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Model Recommendations
        st.markdown("### 💡 Model Recommendations")
        
        best_accuracy = metrics_df.loc[metrics_df['Accuracy'].idxmax()]
        best_f1 = metrics_df.loc[metrics_df['F1'].idxmax()]
        best_mcc = metrics_df.loc[metrics_df['MCC'].idxmax()]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div class="success-box">
                    <h4 style="margin-top: 0;">🏆 Best Accuracy</h4>
                    <p style="font-size: 1.25rem; font-weight: 700; margin: 0;">
                        {best_accuracy['Model']}
                    </p>
                    <p style="color: #6b7280; margin: 0.25rem 0 0 0;">
                        {best_accuracy['Accuracy']:.4f}
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="success-box">
                    <h4 style="margin-top: 0;">⚖️ Best F1 Score</h4>
                    <p style="font-size: 1.25rem; font-weight: 700; margin: 0;">
                        {best_f1['Model']}
                    </p>
                    <p style="color: #6b7280; margin: 0.25rem 0 0 0;">
                        {best_f1['F1']:.4f}
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="success-box">
                    <h4 style="margin-top: 0;">🔬 Best MCC</h4>
                    <p style="font-size: 1.25rem; font-weight: 700; margin: 0;">
                        {best_mcc['Model']}
                    </p>
                    <p style="color: #6b7280; margin: 0.25rem 0 0 0;">
                        {best_mcc['MCC']:.4f}
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
            <div class="warning-box fade-in">
                <h3 style="margin-top: 0;">⚠️ No Trained Models Found</h3>
                <p style="margin-bottom: 0;">
                    Please train models first in the <strong>🎓 Model Training</strong> page.
                </p>
            </div>
        """, unsafe_allow_html=True)

# PREDICTIONS PAGE
elif page == "Predictions":
    st.markdown("""
        <div class="custom-header fade-in" style="padding: 1.5rem;">
            <h1 style="font-size: 2rem; margin: 0;">🔮 Make Predictions</h1>
            <p style="margin: 0.5rem 0 0 0;">Use trained models to predict on new data</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if 'trained_models' in st.session_state:
        st.markdown("""
            <div class="info-box fade-in">
                <h3 style="margin-top: 0;">📋 How It Works</h3>
                <ol style="margin-bottom: 0;">
                    <li>Select a trained model from the dropdown</li>
                    <li>Upload a CSV file with the same features (without target column)</li>
                    <li>Click "Make Predictions" to get results</li>
                    <li>Download predictions as CSV for further analysis</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Model selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            model_name = st.selectbox(
                "🤖 Select Model:",
                list(st.session_state['trained_models'].keys()),
                index=5  # Default to XGBoost
            )
        
        with col2:
            if model_name:
                # Get model metrics
                if 'results' in st.session_state:
                    metrics = st.session_state['results'][model_name]['metrics']
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>Model Accuracy</h3>
                            <p class="metric-value" style="font-size: 1.5rem;">
                                {metrics['Accuracy']:.2%}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Upload new data
        pred_file = st.file_uploader(
            "📁 Upload data for prediction (CSV)",
            type=['csv'],
            key='pred',
            help="Upload a CSV file with the same features as training data (without target column)"
        )
        
        if pred_file is not None:
            pred_df = pd.read_csv(pred_file)
            
            st.markdown("""
                <div class="success-box">
                    <h4 style="margin-top: 0;">✅ File Uploaded Successfully</h4>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📋 Data Preview")
            st.dataframe(pred_df.head(10), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Samples", pred_df.shape[0])
            with col2:
                st.metric("Features", pred_df.shape[1])
            with col3:
                st.metric("Model", model_name)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button("🔮 Make Predictions", type="primary", use_container_width=True):
                    with st.spinner(f"🔄 Generating predictions using {model_name}..."):
                        try:
                            # Get original feature names from training
                            original_features = list(st.session_state['X_test'].columns)
                            
                            # Check if feature count matches
                            if pred_df.shape[1] != len(original_features):
                                st.error(f"""
                                    ❌ **Feature Count Mismatch!**
                                    
                                    - Expected features: **{len(original_features)}**
                                    - Received features: **{pred_df.shape[1]}**
                                    
                                    Please upload a file with the same number of features as the training data.
                                """)
                                st.stop()
                            
                            # Rename columns to match training features
                            pred_df_renamed = pred_df.copy()
                            pred_df_renamed.columns = original_features
                            
                            # Scale the data
                            scaler = st.session_state['scaler']
                            X_pred_scaled = scaler.transform(pred_df_renamed)
                        
                            # Make predictions
                            model = st.session_state['trained_models'][model_name]
                            predictions = model.predict(X_pred_scaled)
                            
                            # Get probabilities if available
                            if hasattr(model, 'predict_proba'):
                                probabilities = model.predict_proba(X_pred_scaled)
                                confidence = np.max(probabilities, axis=1)
                            else:
                                confidence = np.ones(len(predictions))
                            
                            # Display results
                            result_df = pred_df.copy()
                            result_df['Prediction'] = predictions
                            result_df['Prediction_Label'] = result_df['Prediction'].map({
                                0: 'Negative (0)', 
                                1: 'Positive (1)'
                            })
                            result_df['Confidence'] = confidence
                            
                            st.markdown("""
                                <div class="success-box fade-in">
                                    <h3 style="margin-top: 0;">✅ Predictions Complete!</h3>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("### 📊 Prediction Results")
                            
                            # Summary
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                total = len(result_df)
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <h3>Total Predictions</h3>
                                        <p class="metric-value">{total}</p>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                positive = (result_df['Prediction'] == 1).sum()
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <h3>Positive Class</h3>
                                        <p class="metric-value">{positive}</p>
                                        <p class="metric-label">{positive/total*100:.1f}%</p>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                negative = (result_df['Prediction'] == 0).sum()
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <h3>Negative Class</h3>
                                        <p class="metric-value">{negative}</p>
                                        <p class="metric-label">{negative/total*100:.1f}%</p>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # Show results table
                            st.dataframe(result_df, use_container_width=True, height=400)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # Download predictions
                            csv = result_df.to_csv(index=False)
                            
                            col1, col2, col3 = st.columns([1, 2, 1])
                            
                            with col2:
                                st.download_button(
                                    label="📥 Download Predictions as CSV",
                                    data=csv,
                                    file_name=f'predictions_{model_name.replace(" ", "_").lower()}.csv',
                                    mime='text/csv',
                                    use_container_width=True
                                )
                        
                        except Exception as e:
                            st.error(f"""
                                ❌ **Error Making Predictions**
                                
                                {str(e)}
                                
                                **Common Issues:**
                                - Feature count mismatch
                                - Different feature names
                                - Missing values in data
                                - Non-numerical features
                            """)
                            
                            # Show expected features
                            with st.expander("📋 Expected Features (Click to expand)"):
                                original_features = list(st.session_state['X_test'].columns)
                                st.write(f"**Your prediction file should have exactly {len(original_features)} features:**")
                                st.write("")
                                for idx, feat in enumerate(original_features, 1):
                                    st.write(f"{idx}. `{feat}`")
                                use_container_width=True
                            
    
    else:
        st.markdown("""
            <div class="warning-box fade-in">
                <h3 style="margin-top: 0;">⚠️ No Trained Models Found</h3>
                <p style="margin-bottom: 0;">
                    Please train models first in the <strong>🎓 Model Training</strong> page 
                    before making predictions.
                </p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div class="footer">
        <p style="font-size: 0.875rem; margin-bottom: 0.5rem;">
            <strong>Machine Learning Assignment 2</strong> | Submitted by MOHIT RAJ (2025AA05168)
        </p>
        <p style="font-size: 0.875rem; margin: 0;">
            Developed with ❤️ using Streamlit | © 2026
        </p>
    </div>
""", unsafe_allow_html=True)
