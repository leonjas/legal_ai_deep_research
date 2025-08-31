import os
import streamlit as st
import sys

# Fix OpenMP conflict issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import only what we need for the main UI
from app.components.contract_analysis_ui import ContractAnalysisUI

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

# Main contract analysis interface with tabs for all three features
st.markdown("## Contract Analysis Suite")

# Create tabs for the three main features
tab1, tab2, tab3 = st.tabs([
    "📄 Contract Summarization", 
    "⚖️ Unfair Clause Detection", 
    "🎯 Appeal Recommendations"
])

# Import all required modules
from app.components.contract_analysis_ui import ContractAnalysisUI
from app.models.contract_summarizer import ContractSummarizer
from app.models.appeal_recommender import AppealRecommendationPipeline

with tab1:
    st.markdown("### 📄 Contract Summarization")
    st.markdown("Extract key information, terms, and obligations from your contracts.")
    
    # File upload for summarization
    uploaded_file_summary = st.file_uploader(
        "Upload contract for summarization",
        type=['pdf', 'docx', 'txt'],
        key="summary_upload",
        help="Upload a contract to get an intelligent summary"
    )
    
    # Text input for summarization
    contract_text_summary = st.text_area(
        "Or paste contract text here:",
        height=200,
        key="summary_text",
        placeholder="Paste your contract text here for summarization..."
    )
    
    # Summarization button
    if st.button("📝 Generate Summary", type="primary", key="summarize_btn"):
        if uploaded_file_summary is not None or contract_text_summary.strip():
            with st.spinner("Generating contract summary..."):
                try:
                    # Initialize summarizer
                    summarizer = ContractSummarizer()
                    
                    if uploaded_file_summary:
                        # Process uploaded file
                        import tempfile
                        import os
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file_summary.name.split(".")[-1]}') as tmp_file:
                            tmp_file.write(uploaded_file_summary.read())
                            tmp_path = tmp_file.name
                        
                        # Extract text and summarize
                        from app.utils.pdf_parser import PDFParser
                        parser = PDFParser()
                        text = parser.extract_text(tmp_path)
                        summary = summarizer.summarize_contract(text)
                        
                        # Clean up
                        os.unlink(tmp_path)
                    else:
                        # Process text directly
                        summary = summarizer.summarize_contract(contract_text_summary)
                    
                    # Display summary results
                    st.success("✅ Summary generated successfully!")
                    
                    # Executive Summary
                    st.markdown("## 📋 Executive Summary")
                    st.markdown(summary.executive_summary)
                    
                    # Key Information in columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 👥 Parties Involved")
                        for party in summary.parties_involved:
                            st.markdown(f"• {party}")
                        
                        st.markdown("### 📅 Important Dates")
                        for date_info in summary.important_dates:
                            st.markdown(f"• **{date_info.get('type', 'Date')}**: {date_info.get('date', 'N/A')}")
                    
                    with col2:
                        st.markdown("### 📋 Key Terms")
                        for term, description in summary.key_terms.items():
                            st.markdown(f"• **{term}**: {description}")
                        
                        st.markdown("### 💰 Financial Terms")
                        for term, value in summary.financial_terms.items():
                            st.markdown(f"• **{term}**: {value}")
                    
                    # Key Obligations
                    st.markdown("### 📝 Key Obligations")
                    for party, obligations in summary.key_obligations.items():
                        st.markdown(f"**{party}:**")
                        for obligation in obligations:
                            st.markdown(f"  • {obligation}")
                    
                    # Contract Details
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Contract Type", summary.contract_type)
                    with col2:
                        st.metric("Word Count", summary.word_count)
                    with col3:
                        st.metric("Est. Read Time", f"{summary.estimated_read_time} min")
                    
                except Exception as e:
                    st.error(f"❌ Summarization failed: {str(e)}")
                    st.error("Please check the file format and try again.")
        else:
            st.warning("⚠️ Please upload a file or enter contract text to summarize.")

with tab2:
    st.markdown("### ⚖️ Unfair Clause Detection")
    st.markdown("Identify potentially unfair or problematic clauses in your contracts.")
    
    # Use the existing unfair clause detection UI
    ContractAnalysisUI.render_pipeline_upload()

with tab3:
    st.markdown("### 🎯 Appeal Recommendations")
    st.markdown("Get legal appeal recommendations and strategies based on your contract analysis.")
    
    # File upload for appeal analysis
    uploaded_file_appeal = st.file_uploader(
        "Upload contract for appeal analysis",
        type=['pdf', 'docx', 'txt'],
        key="appeal_upload",
        help="Upload a contract to get appeal recommendations"
    )
    
    # Text input for appeal analysis
    contract_text_appeal = st.text_area(
        "Or paste contract text here:",
        height=200,
        key="appeal_text",
        placeholder="Paste your contract text here for appeal analysis..."
    )
    
    # Appeal analysis button
    if st.button("🎯 Generate Appeal Recommendations", type="primary", key="appeal_btn"):
        if uploaded_file_appeal is not None or contract_text_appeal.strip():
            with st.spinner("Analyzing contract for appeal opportunities..."):
                try:
                    # Initialize appeal recommender
                    appeal_pipeline = AppealRecommendationPipeline()
                    
                    if uploaded_file_appeal:
                        # Process uploaded file
                        import tempfile
                        import os
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file_appeal.name.split(".")[-1]}') as tmp_file:
                            tmp_file.write(uploaded_file_appeal.read())
                            tmp_path = tmp_file.name
                        
                        # Extract text and analyze
                        from app.utils.pdf_parser import PDFParser
                        parser = PDFParser()
                        text = parser.extract_text(tmp_path)
                        appeal_result = appeal_pipeline.generate_appeal_recommendations(text)
                        
                        # Clean up
                        os.unlink(tmp_path)
                    else:
                        # Process text directly
                        appeal_result = appeal_pipeline.generate_appeal_recommendations(contract_text_appeal)
                    
                    # Display appeal recommendations
                    st.success("✅ Appeal analysis completed!")
                    
                    # Risk Assessment
                    risk_score = appeal_result.get('risk_score', 0.5)
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Risk Score", f"{risk_score:.1%}")
                    with col2:
                        st.metric("Issues Found", len(appeal_result.get('recommendations', [])))
                    with col3:
                        precedents = len(appeal_result.get('precedents', []))
                        st.metric("Precedents", precedents)
                    
                    # Appeal Recommendations
                    st.markdown("## 🎯 Recommended Appeal Strategies")
                    
                    recommendations = appeal_result.get('recommendations', [])
                    for i, rec in enumerate(recommendations, 1):
                        with st.expander(f"📋 Recommendation #{i}: {rec.get('title', 'Strategy')}"):
                            st.markdown(f"**Strategy**: {rec.get('strategy', 'N/A')}")
                            st.markdown(f"**Reasoning**: {rec.get('reasoning', 'N/A')}")
                            st.markdown(f"**Strength**: {rec.get('strength', 'Medium')}")
                            if rec.get('precedents'):
                                st.markdown("**Supporting Precedents:**")
                                for precedent in rec.get('precedents', []):
                                    st.markdown(f"• {precedent}")
                    
                    # Legal Precedents
                    if appeal_result.get('precedents'):
                        st.markdown("## 📚 Relevant Legal Precedents")
                        for i, precedent in enumerate(appeal_result.get('precedents', []), 1):
                            with st.expander(f"Case #{i}: {precedent.get('case_name', 'Unknown Case')}"):
                                st.markdown(f"**Relevance**: {precedent.get('relevance_score', 0):.1%}")
                                st.markdown(f"**Summary**: {precedent.get('summary', 'N/A')}")
                                st.markdown(f"**Key Points**: {precedent.get('key_points', 'N/A')}")
                    
                except Exception as e:
                    st.error(f"❌ Appeal analysis failed: {str(e)}")
                    st.error("Please check the file format and try again.")
        else:
            st.warning("⚠️ Please upload a file or enter contract text for appeal analysis.")

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
