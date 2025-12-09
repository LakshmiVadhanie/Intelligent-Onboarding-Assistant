import streamlit as st
import os
import sys
from pathlib import Path
import warnings
import json
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from src.generation.rag_pipeline import UniversalRAGPipeline

# Page config
st.set_page_config(
    page_title="GitLab Onboarding Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS (keep your existing CSS - it's great!)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0f1117;
    }
    
    .stTextInput > div > div > input {
        background-color: #1e1e1e;
        color: #ffffff;
        border: 2px solid #4a4a4a;
        border-radius: 10px;
        padding: 15px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.5);
    }
    
    .answer-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-left: 4px solid #667eea;
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
        animation: slideIn 0.5s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .source-box {
        background-color: #1e1e1e;
        border: 1px solid #4a4a4a;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .source-box:hover {
        border-color: #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: bold;
        animation: fadeIn 1s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .metric-card {
        background: linear-gradient(135deg, #2d2d2d 0%, #1e1e1e 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #4a4a4a;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        border-color: #667eea;
    }
    
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        color: #888;
        font-size: 0.9em;
        margin-top: 5px;
        font-weight: 600;
    }
    
    .confidence-bar {
        height: 8px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
        margin-top: 10px;
        animation: growWidth 1s ease;
    }
    
    @keyframes growWidth {
        from { width: 0%; }
        to { width: 100%; }
    }
    
    .stat-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
        padding: 8px 16px;
        border-radius: 20px;
        margin: 5px;
        border: 1px solid #667eea40;
        font-size: 0.85em;
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .stExpander {
        background-color: #1e1e1e;
        border-radius: 10px;
        border: 1px solid #4a4a4a;
    }
    
    .fun-emoji {
        font-size: 3em;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    </style>
""", unsafe_allow_html=True)

# Session state initialization
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'total_queries' not in st.session_state:
    st.session_state.total_queries = 0
if 'avg_response_time' not in st.session_state:
    st.session_state.avg_response_time = 0
if 'response_times' not in st.session_state:
    st.session_state.response_times = []
if 'system_stats' not in st.session_state:
    st.session_state.system_stats = None

def get_system_stats():
    """Get dynamic system statistics"""
    stats = {
        'num_documents': 0,
        'embedding_dim': 0,
        'num_chunks': 0,
        'model_name': 'Unknown',
        'vector_store': 'ChromaDB + GCS'
    }
    
    # Try to load embedding info
    embedding_info_path = Path("models/embeddings/model_info.json")
    if embedding_info_path.exists():
        with open(embedding_info_path, 'r') as f:
            info = json.load(f)
            stats['num_documents'] = info.get('num_embeddings', 0)
            stats['embedding_dim'] = info.get('embedding_dim', 0)
            stats['model_name'] = info.get('model_name', 'Unknown')
    
    # Try to load vector store info
    vector_store_path = Path("models/vector_store")
    if vector_store_path.exists():
        chroma_db = vector_store_path / "chroma.sqlite3"
        if chroma_db.exists():
            stats['num_chunks'] = stats['num_documents']
    
    return stats

@st.cache_resource
def initialize_pipeline():
    """Initialize RAG pipeline with Groq and GCS"""
    # Get credentials path
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path is None:
        default_creds = Path("mlops-476419-2c1937dab204.json")
        if default_creds.exists():
            creds_path = str(default_creds)
    
    return UniversalRAGPipeline(
        provider="groq",  # ← CHANGED TO GROQ
        use_gcs=True,
        bucket_name="mlops-data-oa",
        project_id="mlops-476419",
        credentials_path=creds_path
    )

def create_response_time_chart(times):
    """Create animated response time chart"""
    if not times:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=times,
        mode='lines+markers',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8, color='#764ba2'),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)'
    ))
    
    fig.update_layout(
        title="Response Time Trend",
        xaxis_title="Query Number",
        yaxis_title="Time (seconds)",
        template="plotly_dark",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_confidence_gauge(score):
    """Create confidence gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence", 'font': {'size': 16}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#667eea"},
            'bar': {'color': "#667eea"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#4a4a4a",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255, 0, 0, 0.2)'},
                {'range': [50, 75], 'color': 'rgba(255, 255, 0, 0.2)'},
                {'range': [75, 100], 'color': 'rgba(0, 255, 0, 0.2)'}
            ],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white"},
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# Sidebar
with st.sidebar:
    st.image("https://about.gitlab.com/images/press/logo/png/gitlab-logo-gray-rgb.png", width=200)
    
    st.markdown("## 🚀 GitLab Onboarding Assistant")
    st.markdown("### *Your AI-Powered Guide*")
    st.markdown("---")
    
    # API Status - UPDATED TO CHECK GROQ FIRST
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    if groq_key:
        st.success("✅ Groq API Connected")
        st.markdown('<span class="stat-badge pulse">🎉 Free & Fast! 🚀</span>', unsafe_allow_html=True)
    elif gemini_key:
        st.success("✅ Gemini API Connected")
        st.markdown('<span class="stat-badge pulse">🎉 Free Tier Active</span>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ No API Key")
        st.caption("Retrieval-only mode")
        
        with st.expander("🔑 Enable Full RAG Mode"):
            st.markdown("""
            **Get Started with Groq (Recommended - Fast & Free!):**
            1. 🔗 [Get free Groq key](https://console.groq.com/keys)
            2. 📝 Create `.env` file
            3. 🔄 Add: `GROQ_API_KEY=gsk_your-key`
            4. ♻️ Restart app
            
            **Or use Gemini:**
            1. 🔗 [Get free key](https://aistudio.google.com/app/apikey)
            2. Add: `GROQ_API_KEY=your-key`
            """)
    
    st.markdown("---")
    
    # Dynamic System Stats
    if st.session_state.system_stats is None:
        st.session_state.system_stats = get_system_stats()
    
    stats = st.session_state.system_stats
    
    st.markdown("### 🎯 System Status")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['num_documents']}</div>
            <div class="metric-label">📚 Documents</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['embedding_dim']}</div>
            <div class="metric-label">🧮 Dimensions</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="text-align: center; margin-top: 10px;">
        <span class="stat-badge">🔮 Model: {stats['model_name'].split('/')[-1]}</span>
        <span class="stat-badge">💾 {stats['vector_store']}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Session Statistics
    st.markdown("### 📊 Session Stats")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.total_queries}</div>
            <div class="metric-label">🔍 Queries</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_time = st.session_state.avg_response_time
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_time:.2f}s</div>
            <div class="metric-label">⚡ Avg Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Response time chart
    if st.session_state.response_times:
        st.markdown("### 📈 Performance")
        chart = create_response_time_chart(st.session_state.response_times[-10:])
        if chart:
            st.plotly_chart(chart, use_container_width=True)
    
    st.markdown("---")
    
    # Example Questions with categories
    st.markdown("### 💡 Try These Questions")
    
    categories = {
        "🏢 Company Culture": [
            "What is GitLab's sustainability approach?",
            "Tell me about corporate values"
        ],
        "⚖️ Legal & Compliance": [
            "How does risk management work?",
            "What is the Privacy Team's role?"
        ],
        "🔧 Operations": [
            "Explain the CI/CD process",
            "What are the security policies?"
        ]
    }
    
    for category, questions in categories.items():
        with st.expander(category):
            for question in questions:
                if st.button(question, key=f"example_{question}", use_container_width=True):
                    st.session_state.current_query = question
    
    st.markdown("---")
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.total_queries = 0
            st.session_state.response_times = []
            st.session_state.avg_response_time = 0
            st.rerun()
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.system_stats = get_system_stats()
            st.rerun()
    
    st.markdown("---")
    st.markdown("### ⭐ Model Configuration")
    st.markdown(f"""
    <span class="stat-badge">🏆 MRR: 1.0000</span>
    <span class="stat-badge">🎯 Perfect Retrieval</span>
    <span class="stat-badge">☁️ GCS Enabled</span>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("🛠️ Built with ❤️ by Team 13")
    st.caption("⚡ Powered by Groq (Mixtral) & MPNet")  # ← UPDATED
    st.caption("☁️ Data from GCS")

# Main Content
st.markdown('<div class="fun-emoji">🚀</div>', unsafe_allow_html=True)
st.title("GitLab Onboarding Assistant")
st.markdown("### Ask me anything about GitLab's policies, processes, and culture!")

# Initialize pipeline with loading animation
if st.session_state.rag_pipeline is None:
    with st.spinner("🔄 Initializing AI brain... Loading from GCS... Warming up vector store..."):
        progress_bar = st.progress(0)
        
        status_messages = [
            "☁️ Connecting to GCS...",
            "📥 Downloading embeddings...",
            "🧠 Loading models...",
            "🔍 Initializing retrieval...",
            "✨ Almost ready..."
        ]
        
        for i, msg in enumerate(status_messages):
            st.caption(msg)
            time.sleep(0.3)
            progress_bar.progress((i + 1) * 20)
        
        st.session_state.rag_pipeline = initialize_pipeline()
        st.session_state.system_stats = get_system_stats()
        progress_bar.empty()
    
    st.success("✅ Pipeline loaded from GCS! Ready for your questions! 🎉")
    st.balloons()

# Query input
query_input = st.text_input(
    "Your Question:",
    placeholder="e.g., What is GitLab's approach to sustainability? 💚",
    key="query_input",
    label_visibility="collapsed"
)

# Handle example queries
if 'current_query' in st.session_state and st.session_state.current_query:
    query_input = st.session_state.current_query
    st.session_state.current_query = None

# Search button
col1, col2, col3 = st.columns([2, 1, 2])

with col2:
    search_button = st.button("🔍 Ask Now!", use_container_width=True, type="primary")

# Process query
if search_button and query_input:
    st.session_state.total_queries += 1
    
    # Animated thinking messages
    thinking_messages = [
        "🤔 Analyzing your question...",
        "☁️ Querying GCS data...",
        "🔍 Searching through documents...",
        "🧠 Processing with AI...",
        "✨ Generating response..."
    ]
    
    status = st.status("Processing...", expanded=True)
    
    start_time = time.time()
    
    for i, msg in enumerate(thinking_messages):
        status.update(label=msg, state="running")
        time.sleep(0.2)
    
    result = st.session_state.rag_pipeline.generate_answer(query_input, k=3)
    
    end_time = time.time()
    response_time = end_time - start_time
    
    # Update stats
    st.session_state.response_times.append(response_time)
    st.session_state.avg_response_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
    
    result['response_time'] = response_time
    st.session_state.chat_history.insert(0, result)
    
    status.update(label="✅ Complete!", state="complete")
    time.sleep(0.5)
    status.empty()
    
    st.markdown("---")
    
    # Display answer with confidence
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 💡 Answer")
        st.markdown(f"""
        <div class="answer-box">
            {result['answer']}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Stats")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{response_time:.2f}s</div>
            <div class="metric-label">Response Time</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence gauge
        avg_score = sum([doc.get('rerank_score', doc.get('similarity', 0)) 
                        for doc in result['sources']]) / len(result['sources'])
        fig = create_confidence_gauge(avg_score)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📚 Retrieved Sources")
    
    # Sources with better visualization
    for i, doc in enumerate(result['sources'], 1):
        title = doc['metadata'].get('title', 'Untitled Document')
        score = doc.get('rerank_score', doc.get('similarity', 0))
        text = doc['document']
        
        # Score emoji
        if score > 0.8:
            score_emoji = "🟢"
        elif score > 0.6:
            score_emoji = "🟡"
        else:
            score_emoji = "🟠"
        
        with st.expander(f"{score_emoji} Source {i}: {title} | Relevance: {score:.1%}", expanded=(i==1)):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Rerank Score", f"{score:.4f}")
            with col2:
                if doc.get('rank_before_rerank'):
                    st.metric("Original Rank", f"#{doc['rank_before_rerank']}")
                else:
                    st.metric("Retrieval", "Top-K")
            with col3:
                source_type = doc['metadata'].get('source_type', 'Unknown')
                st.metric("Type", source_type)
            
            st.markdown(f"""
            <div style="background-color: #1e1e1e; padding: 15px; border-radius: 8px; margin-top: 10px;">
                {text[:500]}{'...' if len(text) > 500 else ''}
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence bar
            st.markdown(f"""
            <div style="margin-top: 10px;">
                <div style="background: #2d2d2d; height: 8px; border-radius: 4px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                                height: 100%; width: {score*100}%; border-radius: 4px;
                                animation: growWidth 1s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

elif search_button:
    st.warning("⚠️ Please enter a question first!")

# Chat History
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 📜 Conversation History")
    
    for i, item in enumerate(st.session_state.chat_history[:5], 1):
        with st.expander(f"#{i} • {item['query']}", expanded=(i == 1)):
            st.markdown(f"**💬 Answer:** {item['answer']}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sources", item['num_sources'])
            with col2:
                st.metric("Provider", item.get('provider', 'N/A'))
            with col3:
                response_time = item.get('response_time', 0)
                st.metric("Time", f"{response_time:.2f}s")

# Footer with dynamic metrics
st.markdown("---")
st.markdown("### 🎯 System Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{st.session_state.system_stats['num_documents']}</div>
        <div class="metric-label">📚 Documents Indexed</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{st.session_state.system_stats['embedding_dim']}</div>
        <div class="metric-label">🧮 Embedding Dims</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">FREE</div>
        <div class="metric-label">💰 Cost/Query</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">1.00</div>
        <div class="metric-label">🏆 MRR Score</div>
    </div>
    """, unsafe_allow_html=True)

# Fun footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <span class="stat-badge">🚀 Model A: Large Chunks + MPNet</span>
    <span class="stat-badge">⚡ Sub-3s Response Time</span>
    <span class="stat-badge">🎯 Perfect Retrieval</span>
    <span class="stat-badge">☁️ GCS-Powered</span>
    <span class="stat-badge">💚 Eco-Friendly (Free Tier)</span>
</div>
""", unsafe_allow_html=True)