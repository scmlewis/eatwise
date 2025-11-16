import streamlit as st
import numpy as np
import pandas as pd
from openai import AzureOpenAI
import time
import base64
import random
import re
import os
from io import BytesIO

# ==================== CONFIGURATION (Backend) ====================
# Load Azure OpenAI credentials from Streamlit secrets or environment variables
# For local development, create .streamlit/secrets.toml with your credentials
# For Streamlit Cloud, add secrets in the app settings panel
try:
    AZURE_API_KEY = st.secrets.get("AZURE_API_KEY") or os.getenv("AZURE_API_KEY")
    AZURE_API_VERSION = st.secrets.get("AZURE_API_VERSION") or os.getenv("AZURE_API_VERSION", "2023-05-15")
    AZURE_ENDPOINT = st.secrets.get("AZURE_ENDPOINT") or os.getenv("AZURE_ENDPOINT", "https://hkust.azure-api.net")
except (FileNotFoundError, AttributeError):
    # Fallback if secrets file doesn't exist
    AZURE_API_KEY = os.getenv("AZURE_API_KEY")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2023-05-15")
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "https://hkust.azure-api.net")

if not AZURE_API_KEY:
    st.error("❌ Error: AZURE_API_KEY is not configured. Please set it in .streamlit/secrets.toml or as an environment variable.")
    st.stop()
# =================================================================

# Page configuration
st.set_page_config(
    page_title="Eatwise",
    page_icon="🥗",
    layout="wide"
)

# Fix sidebar selectboxes only
st.markdown("""
<style>
    /* Sidebar select boxes - white bg, dark text */
    .stSidebar [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border-radius: 10px !important;
    }
    
    .stSidebar [data-baseweb="select"] span,
    .stSidebar [data-baseweb="select"] div {
        color: #0D2818 !important;
    }
    
    /* Dropdown menus (global) */
    [role="listbox"] {
        background-color: #FFFFFF !important;
    }
    
    [role="listbox"] li {
        background-color: #FFFFFF !important;
        color: #0D2818 !important;
    }
    
    [role="listbox"] li:hover {
        background-color: #E8F5E9 !important;
    }
</style>
""", unsafe_allow_html=True)

# Custom CSS for better styling (green-themed nutrition advisor)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Poppins:wght@600;700;800&display=swap');

    :root{
        --primary-dark: #1B5E20;    /* deep forest green */
        --primary: #2E7D32;         /* vibrant green */
        --primary-light: #4CAF50;   /* lighter green */
        --accent: #66BB6A;          /* soft green accent */
        --text-light: #E8F5E9;      /* light text for dark bg */
        --text-muted: #C8E6C9;      /* muted light text */
        --bg-dark: #0D3B0D;         /* dark background */
        --bg-dark-alt: #1B5E20;     /* alternate dark bg */
        --card-dark: #1B3A1B;       /* dark card background */
        --success: #43A047;         /* success green */
    }

    /* FORCE DARK MODE */
    html, body, [class*='stApp'] {
        font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
        background: linear-gradient(135deg, #0D3B0D 0%, #1B5E20 50%, #0D3B0D 100%) !important;
        color: #E8F5E9 !important;
    }

    /* Sidebar base styling */
    .stSidebar {
        background: linear-gradient(180deg, #1B5E20 0%, #2E7D32 50%, #1B5E20 100%) !important;
    }
    .stSidebar [data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #1B5E20 0%, #2E7D32 50%, #1B5E20 100%) !important;
    }

    /* Sidebar text colors - WHITE for high contrast */
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar label, .stSidebar .stMarkdown {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        margin: 0.18rem 0 0.22rem 0 !important;
        font-size: 1rem !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
    }

    .stSidebar .stMarkdown p, .stSidebar .stCaption {
        color: #E8F5E9 !important;
        font-size: 0.78rem !important;
        margin: 0.18rem 0 0.18rem 0 !important;
    }

    /* FIXED: Sidebar dropdowns - no white edges, proper border-radius */
    .stSidebar [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 2px solid #4CAF50 !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }

    .stSidebar [data-baseweb="select"] > div > div {
        background-color: #FFFFFF !important;
        color: #0D2818 !important;
        border-radius: 10px !important;
    }

    /* Select dropdown text */
    .stSidebar [data-baseweb="select"] span {
        color: #0D2818 !important;
    }

    /* Multiselect boxes */
    .stSidebar [data-baseweb="tag"] {
        background-color: #4CAF50 !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
    }

    /* Text inputs in sidebar */
    .stSidebar [data-baseweb="input"], 
    .stSidebar [data-baseweb="textarea"] {
        background-color: #FFFFFF !important;
        border: 2px solid #4CAF50 !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }

    .stSidebar [data-baseweb="base-input"] input,
    .stSidebar textarea {
        background-color: #FFFFFF !important;
        color: #0D2818 !important;
        border-radius: 10px !important;
    }

    /* Hero header - dark mode style */
    .header-container{
        max-width: 1100px;
        margin: 1.2rem auto 0.8rem auto;
        padding: 1.5rem 1.6rem;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(76,175,80,0.2), rgba(102,187,106,0.12));
        border: 2px solid rgba(76,175,80,0.25);
        display: flex;
        align-items: center;
        gap: 1.2rem;
        box-shadow: 0 8px 24px rgba(27,94,32,0.25);
    }

    .hero-icon{
        font-size: 2.8rem;
        background: linear-gradient(135deg, rgba(76,175,80,0.25), rgba(102,187,106,0.15));
        padding: 0.8rem;
        border-radius: 14px;
        border: 2px solid rgba(76,175,80,0.2);
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }

    .main-header{
        font-family: 'Poppins', 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.1rem;
        color: #E8F5E9 !important;
        margin: 0;
        line-height: 1.05;
    }

    .sub-header{
        margin: 0.2rem 0 0 0;
        font-size: 0.95rem;
        color: #C8E6C9 !important;
        font-weight: 500;
    }

    /* Card styles for responses - Dark mode */
    .recommendation-box, .nutrition-analysis-box{
        background: #1B3A1B !important;
        color: #E8F5E9 !important;
        padding: 1.5rem;
        border-radius: 14px;
        border: 2px solid rgba(76,175,80,0.25);
        border-left: 6px solid var(--primary);
        box-shadow: 0 10px 28px rgba(27,94,32,0.25);
        margin: 1.2rem 0;
        transition: all 0.24s cubic-bezier(0.4, 0, 0.2, 1);
        background: linear-gradient(to right, rgba(76,175,80,0.1), #1B3A1B) !important;
    }
    .recommendation-box:hover, .nutrition-analysis-box:hover{
        transform: translateY(-6px);
        box-shadow: 0 16px 40px rgba(27,94,32,0.35);
        border-color: rgba(76,175,80,0.35);
    }

    .recommendation-box { border-left-color: var(--primary-light); }
    .nutrition-analysis-box { border-left-color: var(--success); }

    /* Result presentation - Dark mode */
    .result-header {
        font-family: 'Poppins', 'Inter', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #C8E6C9 !important;
        margin-bottom: 0.8rem;
        border-bottom: 3px solid var(--primary-light);
        padding-bottom: 0.6rem;
    }

    .result-item {
        background: linear-gradient(90deg, rgba(76,175,80,0.12), rgba(102,187,106,0.08));
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        border-radius: 10px;
        border-left: 4px solid var(--primary-light);
        transition: all 0.2s ease;
    }
    .result-item:hover {
        background: linear-gradient(90deg, rgba(76,175,80,0.18), rgba(102,187,106,0.12));
        transform: translateX(4px);
    }

    .result-label {
        color: #A5D6A7 !important;
        font-weight: 700;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }

    .result-value {
        color: #E8F5E9 !important;
        font-size: 1rem;
        line-height: 1.6;
        margin-top: 0.4rem;
    }

    /* Ensure text inside result-value is light */
    .result-value p, .result-value li, .result-value strong, .result-value em {
        color: #E8F5E9 !important;
    }

    /* Button styling */
    .stButton>button{
        background: linear-gradient(180deg, var(--primary-light), var(--primary));
        color: white !important;
        border: none;
        padding: 10px 16px;
        border-radius: 10px;
        font-weight: 600;
        box-shadow: 0 8px 20px rgba(27,94,32,0.3);
        transition: all 0.24s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 12px 28px rgba(27,94,32,0.4);
        transform: translateY(-2px);
        background: linear-gradient(180deg, #66BB6A, var(--primary-light));
    }
    .stButton>button[disabled]{
        opacity: 0.5;
        box-shadow: none;
        cursor: not-allowed;
    }

    /* Main content area inputs - Dark mode */
    textarea, input[type='text'], [data-baseweb="base-input"] input {
        border-radius: 10px !important;
        border: 2px solid rgba(76,175,80,0.4) !important;
        background-color: #1B3A1B !important;
        color: #E8F5E9 !important;
        padding: 11px 12px !important;
        transition: all 0.2s ease !important;
    }
    textarea:focus, input[type='text']:focus, [data-baseweb="base-input"] input:focus {
        border-color: var(--primary-light) !important;
        box-shadow: 0 0 0 3px rgba(76,175,80,0.2) !important;
    }

    /* Placeholder text - Dark mode */
    textarea::placeholder, input::placeholder {
        color: #A5D6A7 !important;
        opacity: 0.7 !important;
    }

    /* Slider styling */
    .stSlider [data-baseweb="slider"] {
        background: linear-gradient(90deg, var(--primary-light), var(--success)) !important;
    }

    /* Tabs - Dark mode */
    .stTabs [data-baseweb="tab-list"] { gap: 1.4rem; }
    .stTabs [data-baseweb="tab"] {
        color: #A5D6A7 !important;
        border-bottom: 3px solid transparent !important;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #C8E6C9 !important;
        border-bottom-color: var(--primary-light) !important;
    }

    /* Info/Alert boxes - Dark mode */
    .stInfo {
        background: linear-gradient(135deg, rgba(76,175,80,0.2), rgba(46,125,50,0.12)) !important;
        border-left: 5px solid var(--success) !important;
        border-radius: 10px !important;
        padding: 1rem 1.2rem !important;
        color: #E8F5E9 !important;
    }

    .stAlert {
        border-radius: 10px !important;
    }

    .stSuccess {
        background: linear-gradient(135deg, rgba(67,160,71,0.2), rgba(46,125,50,0.12)) !important;
        border-left: 5px solid var(--success) !important;
        color: #E8F5E9 !important;
    }

    .stWarning {
        background: linear-gradient(135deg, rgba(251,192,45,0.2), rgba(251,140,0,0.15)) !important;
        border-left: 5px solid #FB8C00 !important;
        color: #FFF3E0 !important;
    }

    /* Expander - Dark mode */
    .stExpander {
        border: 2px solid rgba(76,175,80,0.25) !important;
        border-radius: 10px !important;
        background: rgba(27,58,27,0.5) !important;
    }
    
    .stExpander summary {
        color: #E8F5E9 !important;
        font-weight: 600 !important;
    }

    /* Divider */
    .stDivider {
        background: linear-gradient(90deg, transparent, var(--primary-light), transparent) !important;
        height: 3px !important;
    }

    /* Metric Cards - Dark mode */
    .metric-card {
        background: linear-gradient(135deg, rgba(76,175,80,0.2), rgba(46,125,50,0.12));
        border: 2px solid rgba(76,175,80,0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.6rem 0;
        text-align: center;
        transition: all 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(46,125,50,0.25);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #C8E6C9;
        margin: 0;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #A5D6A7;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.3rem;
    }

    /* Quick Action Pills */
    .quick-pill {
        display: inline-block;
        background: linear-gradient(135deg, var(--primary-light), var(--accent));
        color: white !important;
        padding: 0.6rem 1rem;
        border-radius: 20px;
        margin: 0.4rem 0.4rem 0.4rem 0;
        font-size: 0.85rem;
        font-weight: 600;
        border: none;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(46,125,50,0.3);
        transition: all 0.2s ease;
    }
    .quick-pill:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(46,125,50,0.4);
    }
    
    /* Pill-style headers */
    .pill-header {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: linear-gradient(90deg, var(--primary), var(--primary-light));
        color: #FFFFFF !important;
        padding: 12px 18px;
        border-radius: 999px;
        font-weight: 800;
        margin-bottom: 0.8rem;
        box-shadow: 0 10px 30px rgba(27,94,32,0.3);
        font-size: 1.18rem;
        border: 1px solid rgba(255,255,255,0.1);
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }

    /* History section headers - match pill header size */
    h2, .stMarkdown h2 {
        font-size: 1.18rem !important;
        font-weight: 700 !important;
        color: #E8F5E9 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
            
    /* Tips Widget */
    .tips-widget {
        background: linear-gradient(135deg, rgba(102,187,106,0.2), rgba(76,175,80,0.15));
        border-left: 4px solid var(--primary-light);
        border-radius: 8px;
        padding: 0.6rem;
        margin: 0.6rem 0;
        font-size: 0.82rem;
        line-height: 1.5;
        color: #FFFFFF !important;
    }
    .tips-widget .tip-title {
        font-weight: 700;
        color: #FFFFFF !important;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.95rem;
    }
    .tips-widget .daily-label {
        display: inline-block;
        background: linear-gradient(90deg, #FFFFFF, #E8F5E9);
        color: #1B5E20 !important;
        padding: 6px 10px;
        border-radius: 999px;
        font-weight: 700;
        margin-bottom: 0.45rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        font-size: 0.95rem;
    }

    /* Footer */
    .app-footer{ 
        text-align: center; 
        color: #A5D6A7 !important;
        padding: 1.5rem 0;
        border-top: 2px solid rgba(76,175,80,0.25);
    }

    /* Force all Streamlit text elements to light color */
    .stMarkdown, .stText, p, span, div {
        color: #E8F5E9 !important;
    }

    /* Radio buttons and checkboxes labels */
    .stRadio label, .stCheckbox label {
        color: #E8F5E9 !important;
        font-weight: 500 !important;
    }

    /* File uploader */
    .stFileUploader {
        background: #1B3A1B !important;
        border: 2px dashed rgba(76,175,80,0.4) !important;
        border-radius: 10px !important;
    }

    .stFileUploader label {
        color: #E8F5E9 !important;
    }

    /* Download button */
    .stDownloadButton button {
        background: linear-gradient(180deg, var(--primary-light), var(--primary)) !important;
        color: white !important;
    }

    /* Ensure selectbox dropdowns are visible */
    [data-baseweb="popover"] {
        background-color: #FFFFFF !important;
    }
         
</style>
""", unsafe_allow_html=True)

# App header (hero)
st.markdown(
    """
    <div class="header-container">
        <div class="hero-icon">🥗</div>
        <div>
            <h1 class="main-header">Eatwise</h1>
            <div class="sub-header">Personalized, evidence-based food recommendations & nutritional analysis</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Quick guide - Simpler version
st.markdown("""
<div style='max-width: 1100px; 
            margin: 0 auto 1.5rem auto;
            background: rgba(76, 175, 80, 0.1); 
            border-left: 3px solid #4CAF50; 
            padding: 1rem 1.5rem; 
            border-radius: 6px;'>
    <p style='color: #A5D6A7; margin: 0; font-size: 0.9rem;'>
        <strong style='color: #C8E6C9;'>👋 Welcome to Eatwise!</strong><br>
        • Set your preferences in the sidebar<br>
        • Choose a tab below to get started<br>
        • Ask questions or upload meal photos for instant insights
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'recommendation_history' not in st.session_state:
    st.session_state.recommendation_history = []
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Nutritionist tips database
NUTRITIONIST_TIPS = [
    {
        "emoji": "💧",
        "title": "Hydration Matters",
        "tip": "Drink at least 8 glasses of water daily. Proper hydration improves energy, metabolism, and skin health."
    },
    {
        "emoji": "🥗",
        "title": "Eat the Rainbow",
        "tip": "Include colorful vegetables in your diet. Different colors provide different nutrients and antioxidants."
    },
    {
        "emoji": "🥚",
        "title": "Protein Power",
        "tip": "Include protein at every meal. It helps with satiety, muscle building, and maintaining stable energy levels."
    },
    {
        "emoji": "🌾",
        "title": "Whole Grains Win",
        "tip": "Choose whole grains over refined grains. They provide more fiber and nutrients for better digestion."
    },
    {
        "emoji": "🥑",
        "title": "Healthy Fats",
        "tip": "Don't fear fats! Include sources like avocados, nuts, and olive oil for brain and heart health."
    },
    {
        "emoji": "🍎",
        "title": "Snack Smart",
        "tip": "Plan healthy snacks like fruits, nuts, or yogurt. It prevents overeating at main meals."
    },
    {
        "emoji": "⏰",
        "title": "Meal Timing",
        "tip": "Eat breakfast within 1-2 hours of waking. It kickstarts metabolism and improves concentration."
    },
    {
        "emoji": "🧘",
        "title": "Mindful Eating",
        "tip": "Eat slowly and without distractions. Chew thoroughly to aid digestion and increase satisfaction."
    },
]

# Sidebar for preferences
with st.sidebar:
    st.header("🎯 Your Health Goal")
    health_goal = st.selectbox(
        "Select your primary goal:",
        [
            "General Healthy Eating",
            "Weight Loss",
            "Muscle Building",
            "Keep Fit/Maintenance",
            "Heart Health",
            "Energy Boost",
            "Diabetes Management",
            "High Protein Diet",
            "Vegetarian/Vegan",
            "Low Carb Diet"
        ]
    )

    st.divider()

    # Additional preferences
    st.header("🍽️ Preferences")

    meal_type = st.multiselect(
        "Meal Type (optional):",
        ["Breakfast", "Lunch", "Dinner", "Snack", "Pre-workout", "Post-workout"],
        default=[]
    )

    num_recommendations = st.selectbox(
        "Number of recommendations:",
        options=list(range(1, 11)),
        index=4,
        help="How many food suggestions would you like?"
    )

    dietary_restrictions = st.multiselect(
        "Dietary Restrictions (optional):",
        ["Dairy-free", "Gluten-free", "Nut-free", "Vegetarian", "Vegan", "Halal", "Kosher"],
        default=[]
    )

    st.divider()
    st.caption("💡 Tip: Be specific in your questions for better recommendations!")

    # Nutritionist Tips Widget (inline) - show 'Daily Tip' label inside the box
    tip = random.choice(NUTRITIONIST_TIPS)
    st.markdown(
        f"""
        <div class="tips-widget">
            <div class="daily-label">💚 Daily Tip</div>
            <div class="tip-title">{tip['emoji']} {tip['title']}</div>
            <div>{tip['tip']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Function to create OpenAI client
def create_openai_client():
    try:
        client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT
        )
        return client
    except Exception as e:
        st.error(f"Error creating OpenAI client: {str(e)}")
        return None

# Function to encode image to base64
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')


# Small helper to convert basic markdown (bold, italics, bullet lists) to HTML
def md_to_html(md_text: str) -> str:
    if not md_text:
        return ""
    lines = md_text.splitlines()
    out = []
    in_ul = False
    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        # bullet list
        if stripped.startswith('- '):
            if not in_ul:
                in_ul = True
                out.append('<ul>')
            item = stripped[2:].strip()
            # inline formatting
            item = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', item)
            item = re.sub(r'\*(.+?)\*', r'<em>\1</em>', item)
            out.append(f'<li>{item}</li>')
        else:
            if in_ul:
                out.append('</ul>')
                in_ul = False
            if stripped == '':
                out.append('')
            else:
                line_html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
                line_html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', line_html)
                out.append(f'<p>{line_html}</p>')
    if in_ul:
        out.append('</ul>')
    return '\n'.join([o for o in out if o is not None])

# Function to generate nutrition recommendations
def get_nutrition_recommendations(client, query, health_goal, num_recommendations, meal_type, dietary_restrictions):
    prompt = f"""You are a professional nutrition advisor. Based on the following information, provide {num_recommendations} specific food recommendations.

User's Question: {query}

Health Goal: {health_goal}
Meal Type: {', '.join(meal_type) if meal_type else 'Any meal'}
Dietary Restrictions: {', '.join(dietary_restrictions)}

Please provide exactly {num_recommendations} food recommendations. For each recommendation, include:
1. Food/Meal name
2. Brief description (1-2 sentences)
3. Key nutritional benefits
4. Approximate calories (if relevant)
5. Why it fits the user's goal

Format your response in a clear, organized manner with numbered items."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a knowledgeable nutrition advisor who provides evidence-based, practical food recommendations tailored to individual health goals and dietary needs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return None

# Function to analyze food from image
def analyze_food_from_image(client, image_bytes, additional_query=""):
    base64_image = encode_image(image_bytes)

    prompt = f"""Analyze this food image and provide a detailed nutritional breakdown. Include:

1. **Food Identification**: What food items do you see?
2. **Estimated Portion Size**: Approximate serving size
3. **Nutritional Information**:
   - Calories (approximate)
   - Macronutrients (protein, carbs, fats in grams)
   - Key vitamins and minerals
4. **Health Assessment**: Is this meal healthy? Any concerns?
5. **Recommendations**: How could this meal be improved nutritionally?

{f'Additional Information: {additional_query}' if additional_query else ''}

Provide your analysis in a clear, structured format."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return None

# Function to analyze food from text description
def analyze_food_from_text(client, food_description):
    prompt = f"""Analyze the following food/meal description and provide a detailed nutritional breakdown:

Food Description: {food_description}

Please provide:
1. **Food/Meal Summary**: Brief overview of what was described
2. **Estimated Nutritional Information**:
   - Calories (approximate)
   - Macronutrients (protein, carbs, fats in grams)
   - Key vitamins and minerals
3. **Health Assessment**: Is this meal healthy? Any nutritional concerns?
4. **Recommendations**: How could this meal be improved nutritionally?
5. **Suitable For**: What health goals does this meal support?

Provide your analysis in a clear, structured format."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a nutrition expert who can analyze food descriptions and provide detailed nutritional information and health recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error analyzing food: {str(e)}")
        return None

# Main tabs
tab1, tab2 = st.tabs(["🍴 Get Food Recommendations", "🔍 Analyze Nutritional Content"])

# ===================== TAB 1: Food Recommendations =====================
with tab1:
    col1 = st.columns([1])[0]

    with col1:
        st.markdown("<div class='pill-header'>💬 Ask for Food Recommendations</div>", unsafe_allow_html=True)

# Quick action pills - Context-aware based on health goal
        st.markdown("#### 💡 Quick Suggestions")

        # Generate suggestions based on selected health goal
        if health_goal == "Weight Loss":
            quick_suggestions = [
                ("🥗 Low-Cal Meals", "What are filling but low-calorie meals for weight loss?"),
                ("🍎 Smart Snacks", "What snacks won't sabotage my weight loss goals?"),
                ("🔥 Metabolism Boost", "What foods help boost metabolism for weight loss?"),
            ]
        elif health_goal == "Muscle Building":
            quick_suggestions = [
                ("💪 Protein Power", "What are the best high-protein meals for muscle gain?"),
                ("🏋️ Post-Workout", "What's the ideal post-workout meal for recovery?"),
                ("🥚 Breakfast Gains", "What's a protein-rich breakfast for muscle building?"),
            ]
        elif health_goal == "Keep Fit/Maintenance":
            quick_suggestions = [
                ("⚖️ Balanced Meals", "What are well-balanced meals for maintenance?"),
                ("🏃 Active Lifestyle", "What should I eat to support an active lifestyle?"),
                ("🍱 Meal Prep", "Suggest easy meal prep ideas for the week"),
            ]
        elif health_goal == "Heart Health":
            quick_suggestions = [
                ("❤️ Heart-Healthy Fats", "What are the best heart-healthy fats to include?"),
                ("🧂 Low Sodium", "Suggest flavorful low-sodium meal options"),
                ("🐟 Omega-3 Sources", "What are good omega-3 rich foods besides fish?"),
            ]
        elif health_goal == "Energy Boost":
            quick_suggestions = [
                ("⚡ Morning Energy", "What breakfast gives sustained energy all morning?"),
                ("😴 Beat Afternoon Slump", "What should I eat to avoid the 3pm energy crash?"),
                ("🔋 Pre-Workout Fuel", "What should I eat before a workout for energy?"),
            ]
        elif health_goal == "Diabetes Management":
            quick_suggestions = [
                ("📉 Blood Sugar Control", "What meals help stabilize blood sugar levels?"),
                ("🍞 Low-GI Options", "What are good low-glycemic index food choices?"),
                ("🥗 Balanced Carbs", "How should I balance carbs in my meals?"),
            ]
        elif health_goal == "High Protein Diet":
            quick_suggestions = [
                ("🥩 Protein Variety", "What are diverse protein sources beyond meat?"),
                ("🌱 Plant Protein", "What are the best plant-based protein options?"),
                ("🍳 High-Protein Breakfast", "What's a high-protein breakfast under 400 calories?"),
            ]
        elif health_goal == "Vegetarian/Vegan":
            quick_suggestions = [
                ("🌱 Protein Sources", "What are the best plant-based protein options?"),
                ("💊 Nutrient Coverage", "How do I ensure I get B12, iron, and omega-3?"),
                ("🥗 Complete Meals", "What are balanced vegan meal ideas?"),
            ]
        elif health_goal == "Low Carb Diet":
            quick_suggestions = [
                ("🥑 Keto-Friendly", "What are satisfying low-carb, high-fat meals?"),
                ("🍞 Carb Substitutes", "What are good alternatives to bread, rice, and pasta?"),
                ("🥗 Low-Carb Veggies", "What vegetables are lowest in carbs?"),
            ]
        else:  # General Healthy Eating (default)
            quick_suggestions = [
                ("🥞 Breakfast Ideas", "What are some nutritious breakfast options?"),
                ("🍱 Quick Lunches", "Suggest quick and healthy lunch ideas"),
                ("🍴 Dinner Recipes", "What are some balanced dinner recipes?"),
            ]

        cols_pills = st.columns(len(quick_suggestions))
        for idx, (pill_label, suggestion_text) in enumerate(quick_suggestions):
            with cols_pills[idx]:
                if st.button(pill_label, use_container_width=True, key=f"pill_{idx}"):
                    st.session_state.recommendation_query = suggestion_text

        user_query = st.text_area(
            "What kind of food recommendations are you looking for?",
            placeholder="E.g., 'What should I eat for breakfast to boost my energy?' or 'Suggest protein-rich snacks for muscle building'",
            height=100,
            key="recommendation_query"
        )

        # selection badges removed per user request

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            submit_button = st.button("🔍 Get Recommendations", type="primary", use_container_width=True)
        with col_btn2:
            clear_rec_button = st.button("🗑️ Clear History", use_container_width=True, key="clear_rec")

        if clear_rec_button:
            st.session_state.recommendation_history = []
            st.rerun()

    # 'Current Settings' panel removed (redundant)

    # Handle recommendation submission
    if submit_button:
        if not user_query:
            st.warning("⚠️ Please enter your question or food preference.")
        else:
            client = create_openai_client()
            if client:
                with st.spinner("🤔 Generating personalized recommendations..."):
                    recommendations = get_nutrition_recommendations(
                        client,
                        user_query,
                        health_goal,
                        num_recommendations,
                        meal_type,
                        dietary_restrictions
                    )

                    if recommendations:
                        st.session_state.recommendation_history.append({
                            'query': user_query,
                            'goal': health_goal,
                            'response': recommendations,
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                        st.success("✅ Recommendations generated successfully!")

# Display recommendation history (each AI suggestion rendered as a separate card)
    if st.session_state.recommendation_history:
        st.header("📜 Recommendation History")
        for idx, chat in enumerate(reversed(st.session_state.recommendation_history)):
            with st.expander(f"🕒 {chat['timestamp']} - {chat['goal']}", expanded=(idx==0)):
                st.markdown(f"**Your Question:** {chat['query']}")
                st.divider()
                st.markdown(f'<div class="result-header">✨ AI Recommendations</div>', unsafe_allow_html=True)

                resp_text = chat.get("response", "") or ""
                
                # Split by numbered items (1. 2. 3. etc.)
                parts = re.split(r'\n(?=\d+\.\s+\*\*)', resp_text)
                parts = [p.strip() for p in parts if p.strip()]

                # Filter out only the INTRO sentence (not "Why it fits")
                intro_keywords = ['certainly', 'here are', 'sure thing', 'of course', 
                                'glad to help', "i'd be happy", 'let me suggest']
                
                filtered_parts = []
                for part in parts:
                    # Skip only if it's a short intro sentence at the start
                    is_intro = (
                        len(part.split()) < 30 and
                        any(kw in part.lower() for kw in intro_keywords) and
                        not part.strip().startswith('**') and  # Not a titled section
                        not re.match(r'^\d+\.', part.strip())  # Not a numbered item
                    )
                    if not is_intro:
                        filtered_parts.append(part)

                # Fallback if everything was filtered
                if not filtered_parts:
                    filtered_parts = parts

                # Display recommendations
                if not filtered_parts:
                    # Absolute fallback: show whole response
                    html_body = md_to_html(resp_text)
                    st.markdown(f'<div class="recommendation-box"><div class="result-value">{html_body}</div></div>', 
                              unsafe_allow_html=True)
                else:
                    for part in filtered_parts:
                        lines = part.splitlines()
                        
                        # Extract title (first line, remove numbering)
                        title = lines[0].strip() if lines else "Recommendation"
                        title_clean = re.sub(r'^\d+\.\s*', '', title)  # Remove "1. "
                        title_clean = title_clean.strip('*').strip()   # Remove markdown stars
                        
                        # Extract body (everything after first line)
                        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
                        
                        # Convert to HTML (preserves "Why it fits" since it's in the body)
                        body_html = md_to_html(body)
                        
                        st.markdown(
                            f'''<div class="recommendation-box">
                                <div class="result-label">{title_clean}</div>
                                <div class="result-value">{body_html}</div>
                            </div>''',
                            unsafe_allow_html=True
                        )

# ===================== TAB 2: Nutritional Analysis =====================
with tab2:
    st.markdown("<div class='pill-header'>🔍 Analyze Nutritional Content</div>", unsafe_allow_html=True)

    analysis_method = st.radio(
        "Choose analysis method:",
        ["📸 Upload Food Photo", "📝 Describe Food in Text"],
        horizontal=True
    )

    if analysis_method == "📸 Upload Food Photo":
        st.subheader("Upload a photo of your food")

        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photo of your food for nutritional analysis"
        )

        additional_context = st.text_input(
            "Additional Information (optional):",
            placeholder="E.g., 'grilled chicken breast, 200g' or 'homemade pasta with tomato sauce'",
            key="image_context"
        )

        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(uploaded_file, caption="Uploaded Food Image", use_container_width=True)

            with col2:
                if st.button("🔬 Analyze Food", type="primary", use_container_width=True, key="analyze_image"):
                    client = create_openai_client()
                    if client:
                        with st.spinner("🧠 Analyzing nutritional content..."):
                            image_bytes = uploaded_file.getvalue()
                            analysis = analyze_food_from_image(client, image_bytes, additional_context)

                            if analysis:
                                st.session_state.analysis_history.append({
                                    'method': 'image',
                                    'context': additional_context if additional_context else 'No additional context',
                                    'analysis': analysis,
                                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                                })
                                st.success("✅ Analysis complete!")

    else:  # Text description
        st.subheader("Describe the food you want to analyze")

        food_description = st.text_area(
            "Describe your food/meal:",
            placeholder="E.g., 'One bowl of brown rice with grilled salmon, steamed broccoli, and avocado' or 'Two slices of whole wheat toast with peanut butter and banana'",
            height=100,
            key="food_description"
        )

        if st.button("🔬 Analyze Food", type="primary", key="analyze_text"):
            if not food_description:
                st.warning("⚠️ Please describe the food you want to analyze.")
            else:
                client = create_openai_client()
                if client:
                    with st.spinner("🧠 Analyzing nutritional content..."):
                        analysis = analyze_food_from_text(client, food_description)

                        if analysis:
                            st.session_state.analysis_history.append({
                                'method': 'text',
                                'description': food_description,
                                'analysis': analysis,
                                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                            })
                            st.success("✅ Analysis complete!")

    # Clear analysis history button
    if st.session_state.analysis_history:
        if st.button("🗑️ Clear Analysis History", key="clear_analysis"):
            st.session_state.analysis_history = []
            st.rerun()

    # Display analysis history
    if st.session_state.analysis_history:
        st.header("📊 Analysis History")
        for idx, analysis_item in enumerate(reversed(st.session_state.analysis_history)):
            with st.expander(f"🕒 {analysis_item['timestamp']} - {analysis_item['method'].upper()} Analysis", expanded=(idx==0)):
                if analysis_item['method'] == 'image':
                    st.markdown(f"**Additional Information:** {analysis_item['context']}")
                else:
                    st.markdown(f"**Food Description:** {analysis_item['description']}")
                st.divider()
                st.markdown(f'<div class="result-header">🔬 Nutritional Analysis</div>', unsafe_allow_html=True)

                analysis_text = analysis_item.get('analysis', '') or ''
                # Split analysis into sections by numbered headings or markdown headings (e.g., '1.' or '###')
                sections = re.split(r"\n(?=\s*(?:\d+\.|#{1,6}\s))", "\n" + analysis_text)
                sections = [s.strip() for s in sections if s and s.strip()]

                if not sections:
                    # fallback: show whole analysis
                    html_body = md_to_html(analysis_text)
                    st.markdown(f'<div class="nutrition-analysis-box"><div class="result-value">{html_body}</div></div>', unsafe_allow_html=True)
                else:
                    for s in sections:
                        # determine title and body
                        lines = s.splitlines()
                        first = lines[0].strip() if lines else ''
                        # if first line looks like a heading (starts with digits or #), extract label
                        m = re.match(r'^(?:#{1,6}\s*)?(\d+\.?\s*)(.*)', first)
                        if m:
                            title = m.group(2).strip() if m.group(2) else f"Section {m.group(1).strip()}"
                        else:
                            # remove leading markdown hashes if present
                            title = re.sub(r'^#{1,6}\s*', '', first).strip()
                        title_clean = title.strip().strip('*').strip()
                        body = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ''
                        body_html = md_to_html(body)
                        st.markdown(
                            f'<div class="nutrition-analysis-box"><div class="result-label">{title_clean}</div><div class="result-value">{body_html}</div></div>',
                            unsafe_allow_html=True
                        )

# Footer: App disclaimer (moved back to footer as requested)
st.divider()
st.markdown("""
<div class="app-footer">
    <p><strong>⚠️ Disclaimer:</strong> This app provides AI-generated nutritional suggestions for informational purposes only.
    Always consult with a qualified healthcare professional or registered dietitian before making significant dietary changes.</p>
    <p style='font-size: 0.9rem; margin-top: 0.5rem;'>Powered by Azure OpenAI GPT-4o | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)