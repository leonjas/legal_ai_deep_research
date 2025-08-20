import os
import streamlit as st
import sys

# Fix OpenMP conflict issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.components.contract_upload import ContractUploadComponent

# Page configuration
st.set_page_config(
    page_title="Contract Analysis - Unfair Clause Detection",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown(
    """
    <style>
    /* Main content area */
    .main > div {
        max-width: 1200px !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* Ensure consistent sidebar width */
    .css-1d391kg {
        width: 14rem !important;
    }

    /* Main container styles */
    .stApp {
        max-width: 100%;
        width: 100%;
    }

    /* Form container styles */
    .stForm {
        width: 100%;
        max-width: 1200px;
        margin: 0 auto;
    }

    /* Text area consistent sizing */
    .stTextArea textarea {
        min-height: 100px;
        width: 100% !important;
        box-sizing: border-box;
    }

    /* Headers */
    h1, h2, h3 {
        color: #1E4175;
        margin: 1rem 0;
        width: 100%;
    }

    /* Button styles */
    .stButton button {
        background-color: #1E4175;
        color: white;
        width: 100%;
        margin: 0.5rem 0;
    }

    /* Download button exception */
    .stDownloadButton button {
        width: auto;
        margin: 1rem 0;
        padding: 0.5rem 1rem;
    }

    /* Progress bar */
    .stProgress .st-bo {
        background-color: #1E4175;
    }

    /* Prevent horizontal scroll */
    .element-container {
        width: 100% !important;
        overflow-x: hidden;
    }

    /* Ensure markdown containers don't cause shifts */
    .stMarkdown {
        width: 100% !important;
    }

    /* Tab container styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        width: 100%;
    }

    .stTabs [data-baseweb="tab-panel"] {
        width: 100%;
    }

    /* Add padding at the bottom to prevent content from being hidden behind the footer */
    .main .block-container {
        padding-bottom: 5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.title("📄 Contract Analysis - Unfair Clause Detection")
st.markdown(
    """
    **Advanced contract analysis system using LexGLUE methodology for detecting potentially unfair clauses.**
    
    Upload contract documents (PDF, DOCX, TXT) or paste text directly to identify and analyze potentially unfair terms using state-of-the-art NLP models.
    """
)

# Simple sidebar with information
with st.sidebar:
    st.markdown("## 📄 Contract Analysis")
    st.markdown("""
    **Features:**
    - Pattern-based detection
    - ML-based classification  
    - Risk assessment
    - Detailed recommendations
    - Export capabilities
    
    **Supported formats:**
    - PDF, DOCX, DOC, TXT
    - Direct text input
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    
    # Contract analysis settings
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.1,
        help="Minimum confidence for clause detection"
    )
    
    analysis_method = st.selectbox(
        "Analysis Method",
        ["Both (Pattern + ML)", "Pattern Only", "ML Only"],
        index=0,
        help="Choose detection method"
    )
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info("""
    This system uses:
    - **LexGLUE UNFAIR-ToS** models
    - **Pattern-based** detection
    - **Risk assessment** algorithms
    - **Comprehensive** reporting
    """)

# Main contract analysis interface
st.markdown("## Upload and Analyze Contracts")

# Import the contract analysis UI component
from app.components.contract_analysis_ui import ContractAnalysisUI

# Render the contract analysis interface
ContractAnalysisUI.render_pipeline_upload()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888; margin-top: 2rem;">
        <p><strong>⚠️ Legal Disclaimer:</strong> This system is for informational purposes only and does not constitute legal advice.</p>
        <p>Built with Streamlit, Transformers, and spaCy | For personal, non-commercial use only</p>
    </div>
    """,
    unsafe_allow_html=True
)

if __name__ == "__main__":
    pass
