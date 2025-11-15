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

# Custom CSS for better styling (green-themed nutrition advisor)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Poppins:wght@600;700;800&display=swap');

    :root{
        --primary-dark: #1B5E20;    /* deep forest green */
        --primary: #2E7D32;         /* vibrant green */
        --primary-light: #4CAF50;   /* lighter green */
        --accent: #66BB6A;          /* soft green accent */
        --text-dark: #0D2818;       /* DARKER green text for better contrast */
        --text-muted: #2C4A2C;      /* DARKER muted green-gray */
        --bg-light: #F1F8F1;        /* very light green background */
        --bg-primary: #E8F5E9;      /* light green background */
        --card: #FFFFFF;
        --success: #43A047;         /* success green */
    }

    /* Global font + background */
    html, body, [class*='stApp'] {
        font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
        background: linear-gradient(135deg, var(--bg-light) 0%, var(--bg-primary) 50%, #F1F8F1 100%);
        color: var(--text-dark);
    }

    /* Sidebar base styling */
    .stSidebar {
        background: linear-gradient(180deg, #1B5E20 0%, #2E7D32 50%, #1B5E20 100%) !important;
    }
    .stSidebar [data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #1B5E20 0%, #2E7D32 50%, #1B5E20 100%) !important;
    }

    /* Sidebar text colors (compact) - ENHANCED CONTRAST */
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar label, .stSidebar .stMarkdown {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        margin: 0.18rem 0 0.22rem 0 !important;
        font-size: 1rem !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
    }

    .stSidebar .stMarkdown p, .stSidebar .stCaption {
        color: #E8F5E9 !important;  /* LIGHTER for better contrast */
        font-size: 0.78rem !important;
        margin: 0.18rem 0 0.18rem 0 !important;
    }

    /* Sidebar inputs */
    .stSidebar [data-baseweb="input"], 
    .stSidebar [data-baseweb="textarea"], 
    .stSidebar [data-baseweb="select"] {
        background-color: #FFFFFF !important;  /* WHITE background for inputs */
        border-color: #4CAF50 !important;
        color: #0D2818 !important;  /* DARK text */
    }

    /* Sidebar select/multiselect */
    .stSidebar [data-baseweb="base-input"] input {
        background-color: #FFFFFF !important;  /* WHITE background */
        color: #0D2818 !important;  /* DARK text */
        border: 2px solid #4CAF50 !important;
        border-radius: 8px !important;
    }

    /* Hero header */
    .header-container{
        max-width: 1100px;
        margin: 1.2rem auto 0.8rem auto;
        padding: 1.5rem 1.6rem;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(46,125,50,0.12), rgba(76,175,80,0.08));
        border: 2px solid rgba(46,125,50,0.15);
        display: flex;
        align-items: center;
        gap: 1.2rem;
        box-shadow: 0 8px 24px rgba(27,94,32,0.08);
    }

    .hero-icon{
        font-size: 2.8rem;
        background: linear-gradient(135deg, rgba(46,125,50,0.14), rgba(76,175,80,0.1));
        padding: 0.8rem;
        border-radius: 14px;
        border: 2px solid rgba(46,125,50,0.1);
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }

    .main-header{
        font-family: 'Poppins', 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.1rem;
        color: #0D2818;  /* DARKER for contrast */
        margin: 0;
        line-height: 1.05;
    }

    .sub-header{
        margin: 0.2rem 0 0 0;
        font-size: 0.95rem;
        color: #2C4A2C;  /* DARKER muted */
        font-weight: 500;
    }

    /* Card styles for responses - Modernized */
    .recommendation-box, .nutrition-analysis-box{
        background: var(--card);
        padding: 1.5rem;
        border-radius: 14px;
        border: 2px solid rgba(46,125,50,0.12);
        border-left: 6px solid var(--primary);
        box-shadow: 0 10px 28px rgba(27,94,32,0.08);
        margin: 1.2rem 0;
        transition: all 0.24s cubic-bezier(0.4, 0, 0.2, 1);
        background: linear-gradient(to right, rgba(46,125,50,0.02), #FFFFFF);
    }
    .recommendation-box:hover, .nutrition-analysis-box:hover{
        transform: translateY(-6px);
        box-shadow: 0 16px 40px rgba(27,94,32,0.12);
        border-color: rgba(46,125,50,0.25);
    }

    .recommendation-box { border-left-color: var(--primary); }
    .nutrition-analysis-box { border-left-color: var(--success); }

    /* Modernized result presentation */
    .result-header {
        font-family: 'Poppins', 'Inter', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #0D2818;  /* DARKER */
        margin-bottom: 0.8rem;
        border-bottom: 3px solid var(--primary);
        padding-bottom: 0.6rem;
    }

    .result-item {
        background: linear-gradient(90deg, rgba(46,125,50,0.04), rgba(76,175,80,0.02));
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        border-radius: 10px;
        border-left: 4px solid var(--primary);
        transition: all 0.2s ease;
    }
    .result-item:hover {
        background: linear-gradient(90deg, rgba(46,125,50,0.08), rgba(76,175,80,0.05));
        transform: translateX(4px);
    }

    .result-label {
        color: #0D2818;  /* DARKER */
        font-weight: 700;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }

    .result-value {
        color: #0D2818;  /* DARKER for readability */
        font-size: 1rem;
        line-height: 1.6;
        margin-top: 0.4rem;
    }

    /* Ensure text inside result-value is also dark */
    .result-value p, .result-value li, .result-value strong, .result-value em {
        color: #0D2818 !important;
    }

    /* Button styling */
    .stButton>button{
        background: linear-gradient(180deg, var(--primary), var(--primary-dark));
        color: white !important;
        border: none;
        padding: 10px 16px;
        border-radius: 10px;
        font-weight: 600;
        box-shadow: 0 8px 20px rgba(27,94,32,0.18);
        transition: all 0.24s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 12px 28px rgba(27,94,32,0.25);
        transform: translateY(-2px);
    }
    .stButton>button[disabled]{
        opacity: 0.5;
        box-shadow: none;
        cursor: not-allowed;
    }

    /* Inputs and textareas */
    textarea, input[type='text'], [data-baseweb="base-input"] input {
        border-radius: 10px !important;
        border: 2px solid rgba(46,125,50,0.25) !important;  /* MORE visible border */
        background-color: #FFFFFF !important;  /* Pure white */
        color: #0D2818 !important;  /* DARK text */
        padding: 11px 12px !important;
        transition: all 0.2s ease !important;
    }
    textarea:focus, input[type='text']:focus, [data-baseweb="base-input"] input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(46,125,50,0.1) !important;
    }

    /* Placeholder text */
    textarea::placeholder, input::placeholder {
        color: #6B8F6B !important;  /* Medium contrast for placeholders */
        opacity: 0.8 !important;
    }

    /* Slider styling */
    .stSlider [data-baseweb="slider"] {
        background: linear-gradient(90deg, var(--primary-light), var(--success)) !important;
    }

    /* Tabs spacing */
    .stTabs [data-baseweb="tab-list"] { gap: 1.4rem; }
    .stTabs [data-baseweb="tab"] {
        color: #2C4A2C !important;  /* DARKER */
        border-bottom: 3px solid transparent !important;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #1B5E20 !important;  /* DARKEST green */
        border-bottom-color: var(--primary) !important;
    }

    /* Info box styling */
    .stInfo {
        background: linear-gradient(135deg, rgba(76,175,80,0.12), rgba(46,125,50,0.08)) !important;
        border-left: 5px solid var(--success) !important;
        border-radius: 10px !important;
        padding: 1rem 1.2rem !important;
        color: #0D2818 !important;  /* DARK text */
    }

    .stAlert {
        border-radius: 10px !important;
    }

    .stSuccess {
        background: linear-gradient(135deg, rgba(67,160,71,0.12), rgba(46,125,50,0.08)) !important;
        border-left: 5px solid var(--success) !important;
        color: #0D2818 !important;
    }

    .stWarning {
        background: linear-gradient(135deg, rgba(251,192,45,0.12), rgba(251,192,45,0.08)) !important;
        border-left: 5px solid #F57C00 !important;  /* Darker warning color */
        color: #3E2723 !important;  /* Dark brown text */
    }

    /* Expander styling */
    .stExpander {
        border: 2px solid rgba(46,125,50,0.15) !important;
        border-radius: 10px !important;
    }
    
    .stExpander summary {
        color: #0D2818 !important;  /* DARK text */
        font-weight: 600 !important;
    }

    /* Divider */
    .stDivider {
        background: linear-gradient(90deg, transparent, var(--primary-light), transparent) !important;
        height: 3px !important;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(76,175,80,0.12), rgba(46,125,50,0.06));
        border: 2px solid rgba(76,175,80,0.2);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.6rem 0;
        text-align: center;
        transition: all 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(46,125,50,0.12);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1B5E20;  /* DARKER */
        margin: 0;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #2C4A2C;  /* DARKER */
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
        box-shadow: 0 4px 12px rgba(46,125,50,0.18);
        transition: all 0.2s ease;
    }
    .quick-pill:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(46,125,50,0.25);
    }
    
    /* Pill-style headers for main sections */
    .pill-header {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: linear-gradient(90deg, var(--primary-dark), var(--primary));
        color: #FFFFFF !important;
        padding: 12px 18px;
        border-radius: 999px;
        font-weight: 800;
        margin-bottom: 0.8rem;
        box-shadow: 0 10px 30px rgba(27,94,32,0.18);
        font-size: 1.18rem;
        border: 1px solid rgba(255,255,255,0.06);
        text-shadow: 0 1px 2px rgba(0,0,0,0.15);
    }

    /* Tips Widget (compact) */
    .tips-widget {
        background: linear-gradient(135deg, rgba(102,187,106,0.15), rgba(76,175,80,0.10));
        border-left: 4px solid var(--primary);
        border-radius: 8px;
        padding: 0.6rem;
        margin: 0.6rem 0;
        font-size: 0.82rem;
        line-height: 1.5;
        color: #FFFFFF !important;  /* WHITE text for sidebar */
    }
    .tips-widget .tip-title {
        font-weight: 700;
        color: #FFFFFF !important;  /* WHITE */
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.95rem;
    }
    .tips-widget .daily-label {
        display: inline-block;
        background: linear-gradient(90deg, #FFFFFF, #E8F5E9);
        color: #1B5E20 !important;  /* DARK green text */
        padding: 6px 10px;
        border-radius: 999px;
        font-weight: 700;
        margin-bottom: 0.45rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        font-size: 0.95rem;
    }

    /* Footer */
    .app-footer{ 
        text-align: center; 
        color: #2C4A2C;  /* DARKER */
        padding: 1.5rem 0;
        border-top: 2px solid rgba(46,125,50,0.15);
    }

    /* Streamlit native elements - force dark text */
    .stMarkdown, .stText, p, span, div {
        color: #0D2818 !important;
    }

    /* Radio buttons and checkboxes labels */
    .stRadio label, .stCheckbox label {
        color: #0D2818 !important;
        font-weight: 500 !important;
    }

    /* Dark mode adjustments */
    @media (prefers-color-scheme: dark) {
        html, body, [class*='stApp'] {
            background: linear-gradient(135deg, #0D3B0D 0%, #1B5E20 50%, #0D3B0D 100%);
            color: #E8F5E9;
        }
        .header-container { 
            background: linear-gradient(135deg, rgba(76,175,80,0.2), rgba(102,187,106,0.12));
            box-shadow: 0 8px 24px rgba(27,94,32,0.25);
        }
        .main-header { color: #E8F5E9; }
        .sub-header { color: #C8E6C9; }
        .recommendation-box, .nutrition-analysis-box { 
            background: #1B3A1B; 
            color: #E8F5E9;
            border-color: rgba(76,175,80,0.25);
            background: linear-gradient(to right, rgba(76,175,80,0.1), #1B3A1B);
        }
        .result-header { color: #C8E6C9; }
        .result-item {
            background: linear-gradient(90deg, rgba(76,175,80,0.12), rgba(102,187,106,0.08));
        }
        .result-label { color: #A5D6A7; }
        .result-value, .result-value p, .result-value li, .result-value strong, .result-value em { 
            color: #E8F5E9 !important; 
        }
        .stInfo { 
            background: linear-gradient(135deg, rgba(76,175,80,0.2), rgba(46,125,50,0.12)) !important;
            color: #E8F5E9 !important;
        }
        textarea, input[type='text'], [data-baseweb="base-input"] input {
            background-color: #1B3A1B !important;
            border-color: rgba(76,175,80,0.4) !important;
            color: #E8F5E9 !important;
        }
        .stMarkdown, .stText, p, span, div {
            color: #E8F5E9 !important;
        }
        .stRadio label, .stCheckbox label {
            color: #E8F5E9 !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #A5D6A7 !important;
        }
        .stTabs [aria-selected="true"] {
            color: #C8E6C9 !important;
        }
        .app-footer { color: #A5D6A7; }
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

        # Quick action pills
        st.markdown("#### 💡 Quick Suggestions")
        quick_suggestions = [
            ("🥞 Breakfast", "What are some healthy breakfast options for my health goal?"),
            ("💪 Post-Workout", "What should I eat after a workout?"),
            ("🍱 Quick Lunch", "Suggest quick, healthy lunch options"),
            ("🍴 Dinner", "What are some balanced dinner recipes?"),
            ("🍌 Snacks", "Recommend healthy snacks"),
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
                # Try to split numbered items (common LLM numbered list format)
                parts = re.split(r"\n\s*\d+\.\s+", "\n" + resp_text)
                parts = [p.strip() for p in parts if p and p.strip()]

                if not parts:
                    # fallback: show whole response in one card
                    st.markdown(f'<div class="recommendation-box"><div class="result-value">{resp_text}</div></div>', unsafe_allow_html=True)
                else:
                    for i, part in enumerate(parts, start=1):
                        lines = part.splitlines()
                        title = lines[0].strip() if lines else f"Recommendation {i}"
                        # clean simple markdown markers from title
                        title_clean = title.strip().strip('*').strip()
                        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
                        body_html = md_to_html(body)
                        st.markdown(
                            f'<div class="recommendation-box"><div class="result-label">{title_clean}</div><div class="result-value">{body_html}</div></div>',
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