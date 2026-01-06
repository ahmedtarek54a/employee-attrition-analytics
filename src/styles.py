
def get_glassmorphism_css():
    return """
    <style>
    /* Importing neat fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap');

    /* Main Variables */
    :root {
        --primary-bg: #050510;
        --card-bg: rgba(20, 20, 35, 0.7);
        --neon-cyan: #00f3ff;
        --neon-pink: #bc13fe;
        --text-color: #e0e0e0;
        --accent-glow: 0 0 10px rgba(0, 243, 255, 0.5), 0 0 20px rgba(0, 243, 255, 0.3);
    }

    /* Global Reset & Body */
    .stApp {
        background-color: var(--primary-bg);
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(188, 19, 254, 0.1) 0%, transparent 20%),
            radial-gradient(circle at 90% 80%, rgba(0, 243, 255, 0.1) 0%, transparent 20%);
        color: var(--text-color);
        font-family: 'Roboto', sans-serif;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', sans-serif !important;
        color: white !important;
        text-shadow: 0 0 5px rgba(255, 255, 255, 0.5);
    }
    
    h1 {
        background: -webkit-linear-gradient(45deg, var(--neon-cyan), var(--neon-pink));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(188, 19, 254, 0.3);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 10, 20, 0.85);
        border-right: 1px solid rgba(0, 243, 255, 0.2);
        box-shadow: 5px 0 15px rgba(0, 0, 0, 0.5);
    }

    /* Cards / Containers (Glassmorphism + Neon) */
    .css-1r6slb0, .stDataFrame, .stPlotlyChart, div[data-testid="stMetric"] {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.6);
        transition: all 0.3s ease-in-out;
    }
    
    .stDataFrame:hover, .stPlotlyChart:hover, div[data-testid="stMetric"]:hover {
        border-color: rgba(0, 243, 255, 0.5);
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.2), inset 0 0 10px rgba(0, 243, 255, 0.05);
        transform: translateY(-2px);
    }

    /* Buttons */
    .stButton>button {
        background: transparent;
        border: 1px solid var(--neon-cyan);
        color: var(--neon-cyan);
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        transition: all 0.3s ease;
        box-shadow: 0 0 5px rgba(0, 243, 255, 0.2);
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    
    .stButton>button:hover {
        background: var(--neon-cyan);
        color: #000;
        box-shadow: 0 0 20px var(--neon-cyan);
        border-color: var(--neon-cyan);
        transform: scale(1.02);
    }
    
    .stButton>button:active {
        transform: scale(0.98);
        box-shadow: 0 0 10px var(--neon-cyan);
    }

    /* Metric Values */
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        color: #fff;
        text-shadow: 0 0 10px var(--neon-pink);
    }

    /* Inputs */
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>input {
        background-color: rgba(0, 0, 0, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white !important;
        border-radius: 4px;
    }
    
    .stTextInput>div>div>input:focus, .stSelectbox>div>div>div:focus-within {
        border-color: var(--neon-pink);
        box-shadow: 0 0 8px rgba(188, 19, 254, 0.4);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 4px;
        color: white;
        padding: 5px 15px;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(188, 19, 254, 0.2) !important;
        border-color: var(--neon-pink) !important;
        color: var(--neon-cyan) !important;
        box-shadow: 0 0 10px rgba(188, 19, 254, 0.3);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(255,255,255,0.05);
        color: white;
        border-radius: 4px;
    }

    /* Keyframes for simple breathing animation */
    @keyframes breathe {
        0% { box-shadow: 0 0 5px rgba(0, 243, 255, 0.2); }
        50% { box-shadow: 0 0 15px rgba(0, 243, 255, 0.5); }
        100% { box-shadow: 0 0 5px rgba(0, 243, 255, 0.2); }
    }
    
    .neon-breathing {
        animation: breathe 3s infinite ease-in-out;
    }
    </style>
    """
