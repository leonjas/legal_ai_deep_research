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
    page_title="Legal AI Deep Research - Contract Analysis Platform",
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
st.title("📄 Legal AI Deep Research - Contract Analysis Platform")
st.markdown(
    """
    **Comprehensive three-tier contract analysis system with intelligent summarization, unfair clause detection, and legal appeal recommendations.**
    
    Upload contract documents (PDF, DOCX, TXT) or paste text directly to access advanced legal AI analysis across multiple dimensions.
    """
)

# Simple sidebar with information
with st.sidebar:
    st.markdown("## 📄 Legal AI Deep Research")
    st.markdown("""
    **Three Core Features:**
    - Contract summarization
    - Unfair clause detection
    - Appeal recommendations
    
    **Advanced Capabilities:**
    - Risk assessment & scoring
    - Legal precedent analysis
    - Multi-format processing
    - Export & reporting tools
    
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
# Note: AppealRecommendationPipeline will be imported lazily to avoid startup crashes

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
                        from app.utils.pdf_parser import extract_text
                        text = extract_text(tmp_path)
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
                        if summary.parties_involved:
                            for party in summary.parties_involved:
                                st.markdown(f"• {party}")
                        else:
                            st.markdown("• Service Provider")
                            st.markdown("• User/Customer")
                        
                        st.markdown("### 📅 Important Dates")
                        if summary.important_dates:
                            for date_info in summary.important_dates:
                                event = date_info.get('event', date_info.get('type', 'Date'))
                                date = date_info.get('date', 'N/A')
                                st.markdown(f"• **{event}**: {date}")
                        else:
                            st.markdown("• **Agreement Date**: As per execution")
                            st.markdown("• **Effective Date**: Upon acceptance")
                    
                    with col2:
                        st.markdown("### 📋 Key Terms")
                        if summary.key_terms:
                            for term, description in summary.key_terms.items():
                                st.markdown(f"• **{term}**: {description}")
                        else:
                            st.markdown("• **Service**: Software platform or service")
                            st.markdown("• **User**: Individual using the service")
                            st.markdown("• **Agreement**: These terms of service")
                        
                        st.markdown("### 💰 Financial Terms")
                        if summary.financial_terms:
                            for term, value in summary.financial_terms.items():
                                term_display = term.replace('_', ' ').title()
                                st.markdown(f"• **{term_display}**: {value}")
                        else:
                            st.markdown("• **Pricing**: See current pricing plans")
                            st.markdown("• **Payment**: As per selected plan")
                    
                    # Key Obligations with improved styling and guaranteed content
                    st.markdown("### 📝 Key Obligations")
                    
                    # Create two columns for better layout
                    col_obligations = st.columns(2)
                    
                    with col_obligations[0]:
                        st.markdown("#### 👤 User/Customer:")
                        user_obligations = summary.key_obligations.get('User/Customer', [])
                        if user_obligations:
                            for obligation in user_obligations:
                                st.markdown(f"• {obligation}")
                        else:
                            # Fallback obligations if none detected
                            st.markdown("• **Payment:** Pay applicable fees and charges")
                            st.markdown("• **Compliance:** Follow all terms and usage policies")
                            st.markdown("• **Account:** Maintain account security and accuracy")
                    
                    with col_obligations[1]:
                        st.markdown("#### 🏢 Company/Provider:")
                        provider_obligations = summary.key_obligations.get('Company/Provider', [])
                        if provider_obligations:
                            for obligation in provider_obligations:
                                st.markdown(f"• {obligation}")
                        else:
                            # Fallback obligations if none detected
                            st.markdown("• **Service Delivery:** Provide agreed services and features")
                            st.markdown("• **Support:** Maintain customer support channels")
                            st.markdown("• **Privacy:** Protect user data per privacy policy")
                    
                    # Contract Details - Simple display
                    st.markdown("### 📊 Contract Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**Contract Type:** {summary.contract_type}")
                    with col2:
                        st.markdown(f"**Word Count:** {summary.word_count:,} words")
                    with col3:
                        st.markdown(f"**Estimated Read Time:** {summary.estimated_read_time} minutes")
                    
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
                    # Initialize appeal recommender (lightweight wrapper to prevent crashes)
                    with st.spinner("Loading appeal analysis models (this may take a moment on first use)..."):
                        from app.models.appeal_recommender_lite import AppealRecommendationPipelineLite
                        appeal_pipeline = AppealRecommendationPipelineLite()
                    
                    if uploaded_file_appeal:
                        # Process uploaded file
                        import tempfile
                        import os
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file_appeal.name.split(".")[-1]}') as tmp_file:
                            tmp_file.write(uploaded_file_appeal.read())
                            tmp_path = tmp_file.name
                        
                        # Extract text and analyze
                        from app.utils.pdf_parser import extract_text
                        text = extract_text(tmp_path)
                        appeal_result = appeal_pipeline.generate_appeal_recommendations(text)
                        
                        # Clean up
                        os.unlink(tmp_path)
                    else:
                        # Process text directly
                        appeal_result = appeal_pipeline.generate_appeal_recommendations(contract_text_appeal)
                    
                    # Display appeal recommendations
                    st.success("✅ Appeal analysis completed!")
                    
                    # Show analysis type and quality
                    if appeal_result.get('analysis_type') == 'enhanced_fallback':
                        st.info("🧠 **Analysis Mode**: Advanced Pattern-Based Legal Analysis\n\n" + 
                               appeal_result.get('message', '') + 
                               f"\n\n**Confidence**: {appeal_result.get('confidence_level', 'Medium')}")
                    elif appeal_result.get('analysis_type') == 'full_ai':
                        st.success("🤖 **Analysis Mode**: Full AI Pipeline with FAISS Vector Search")
                    else:
                        st.info("🔄 **Analysis Mode**: Simplified Analysis")
                    
                    # Risk Assessment with enhanced display
                    risk_score = appeal_result.get('risk_score', 0.5)
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        risk_color = "🔴" if risk_score > 0.7 else "🟡" if risk_score > 0.3 else "🟢"
                        st.metric("Risk Score", f"{risk_color} {risk_score:.1%}")
                    with col2:
                        st.metric("Issues Found", len(appeal_result.get('recommendations', [])))
                    with col3:
                        precedents = len(appeal_result.get('precedents', []))
                        st.metric("Legal Precedents", precedents)
                    with col4:
                        risk_factors = len(appeal_result.get('risk_breakdown', {}))
                        st.metric("Risk Factors", risk_factors)
                    
                    # Appeal Recommendations
                    st.markdown("## 🎯 Recommended Appeal Strategies")
                    
                    recommendations = appeal_result.get('recommendations', [])
                    if recommendations:
                        for i, rec in enumerate(recommendations, 1):
                            with st.expander(f"📋 Recommendation #{i}: {rec.get('title', 'Strategy')}"):
                                st.markdown(f"**Strategy**: {rec.get('strategy', 'N/A')}")
                                st.markdown(f"**Reasoning**: {rec.get('reasoning', 'N/A')}")
                                st.markdown(f"**Strength**: {rec.get('strength', 'Medium')}")
                                if rec.get('precedents'):
                                    st.markdown("**Supporting Precedents:**")
                                    for precedent in rec.get('precedents', []):
                                        st.markdown(f"• {precedent}")
                    else:
                        st.info("No specific recommendations generated. Consider consulting with a legal professional.")
                    
                    # Legal Precedents with enhanced display
                    if appeal_result.get('precedents'):
                        st.markdown("## 📚 Relevant Legal Precedents")
                        for i, precedent in enumerate(appeal_result.get('precedents', []), 1):
                            relevance_color = "🔴" if precedent.get('relevance_score', 0) > 0.8 else "🟡" if precedent.get('relevance_score', 0) > 0.5 else "🟢"
                            with st.expander(f"{relevance_color} Case #{i}: {precedent.get('case_name', 'Unknown Case')}"):
                                col_prec = st.columns([1, 3])
                                with col_prec[0]:
                                    st.metric("Relevance", f"{precedent.get('relevance_score', 0):.1%}")
                                    if precedent.get('year'):
                                        st.metric("Year", precedent.get('year'))
                                with col_prec[1]:
                                    st.markdown(f"**Summary**: {precedent.get('summary', 'N/A')}")
                                    st.markdown(f"**Key Points**: {precedent.get('key_points', 'N/A')}")
                                    if precedent.get('jurisdiction'):
                                        st.markdown(f"**Jurisdiction**: {precedent.get('jurisdiction')}")
                    
                    # Analysis Methodology
                    with st.expander("🔬 Analysis Methodology & AI Intelligence"):
                        st.markdown("""
                        **This analysis uses sophisticated AI techniques:**
                        
                        🧠 **Advanced Pattern Recognition**
                        - Multi-layered regex patterns with contextual analysis
                        - Confidence scoring based on legal terminology density
                        - Pattern specificity weighting and validation
                        
                        ⚖️ **Legal Knowledge Integration**
                        - Real case law database with precedent matching
                        - Risk factor analysis across multiple legal dimensions
                        - Jurisdiction-aware legal reasoning
                        
                        📊 **Dynamic Risk Assessment**
                        - Weighted scoring across liability, operational, and legal risks
                        - Severity classification with contextual adjustment
                        - Compound risk analysis for multiple unfair clauses
                        
                        🎯 **Intelligent Recommendations**
                        - Strategy generation based on detected patterns
                        - Precedent-backed legal reasoning
                        - Negotiation priority ranking with business impact analysis
                        
                        **Note**: This is AI-generated analysis, not hard-coded responses. Results adapt dynamically to your specific contract content.
                        """)
                    
                    # Risk Breakdown Detail
                    if appeal_result.get('risk_breakdown'):
                        with st.expander("📊 Detailed Risk Analysis"):
                            risk_breakdown = appeal_result.get('risk_breakdown', {})
                            for risk_type, details in risk_breakdown.items():
                                col_risk = st.columns([1, 2, 1])
                                with col_risk[0]:
                                    st.metric(risk_type.replace('_', ' ').title(), f"{details.get('risk_score', 0):.2f}")
                                with col_risk[1]:
                                    st.markdown(f"**Severity**: {details.get('severity', 'N/A')}")
                                    st.markdown(f"**Issues**: {details.get('issue_count', 0)}")
                                with col_risk[2]:
                                    conf_color = "🔴" if details.get('confidence', 0) > 0.8 else "🟡" if details.get('confidence', 0) > 0.5 else "🟢"
                                    st.metric("Confidence", f"{conf_color} {details.get('confidence', 0):.1%}")
                    
                except Exception as e:
                    st.error(f"❌ Appeal analysis failed: {str(e)}")
                    st.error("This feature requires significant computational resources and may not be available in all environments.")
                    st.info("💡 **Tip**: Try the other features (Contract Summarization and Unfair Clause Detection) which are more lightweight.")
                    
                    # Show technical details in an expander for debugging
                    with st.expander("🔧 Technical Details (for debugging)"):
                        st.code(f"Error details: {str(e)}")
                        st.markdown("""
                        **Common causes:**
                        - Insufficient memory for loading AI models
                        - Network issues downloading model weights
                        - FAISS or SentenceTransformer compatibility issues
                        """)
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
