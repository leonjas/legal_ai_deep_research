import streamlit as st
from typing import Dict, Any

from app.utils.config import (
    DEFAULT_CONFIDENCE_THRESHOLD, 
    CLAUSE_SEVERITY, 
    RISK_THRESHOLDS,
    APP_NAME,
    APP_VERSION
)

def create_sidebar() -> Dict[str, Any]:
    """
    Create sidebar for contract analysis application.
    
    Returns:
        Dictionary with user configuration settings
    """
    with st.sidebar:
        st.markdown(f"## 📄 {APP_NAME}")
        st.markdown(f"*Version {APP_VERSION}*")
        
        st.markdown("---")
        st.markdown("### 🔧 Analysis Settings")
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=DEFAULT_CONFIDENCE_THRESHOLD,
            step=0.1,
            help="Minimum confidence required to flag a clause as unfair"
        )
        
        # Analysis methods
        st.markdown("**Detection Methods:**")
        use_ml_detection = st.checkbox(
            "Machine Learning Detection",
            value=True,
            help="Use trained models to detect unfair clauses"
        )
        
        use_pattern_detection = st.checkbox(
            "Pattern-based Detection", 
            value=True,
            help="Use rule-based patterns to identify problematic clauses"
        )
        
        # Risk assessment
        show_risk_assessment = st.checkbox(
            "Show Risk Assessment",
            value=True,
            help="Include overall contract risk scoring"
        )
        
        # Show recommendations
        show_recommendations = st.checkbox(
            "Generate Recommendations",
            value=True,
            help="Provide suggested improvements for unfair clauses"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Display Options")
        
        # Results filtering
        show_low_confidence = st.checkbox(
            "Show Low Confidence Results",
            value=False,
            help="Include results below the confidence threshold"
        )
        
        # Clause severity filter
        severity_filter = st.multiselect(
            "Filter by Severity",
            options=["high", "medium", "low"],
            default=["high", "medium", "low"],
            help="Select which severity levels to display"
        )
        
        # Export options
        st.markdown("---")
        st.markdown("### 📄 Export Options")
        
        export_format = st.selectbox(
            "Export Format",
            options=["JSON", "CSV", "Text Report", "PDF Report"],
            index=0,
            help="Choose format for exporting results"
        )
        
        # Information section
        st.markdown("---")
        st.markdown("### ℹ️ About")
        
        with st.expander("📋 Features"):
            st.markdown("""
            **Core Features:**
            - Unfair clause detection
            - Contract summarization
            - Appeal recommendations
            - Risk assessment
            - Export capabilities
            
            **Supported Formats:**
            - PDF, DOCX, DOC, TXT
            - Direct text input
            """)
        
        with st.expander("🤖 AI Models"):
            st.markdown("""
            **Primary Model:**
            - LexGLUE UNFAIR-ToS
            
            **Fallback Models:**
            - CodeHima/TOSRobertaV2
            - Legal-BERT
            - Standard BERT
            
            **Analysis Methods:**
            - Machine learning classification
            - Rule-based pattern matching
            - Semantic similarity search
            """)
        
        with st.expander("⚖️ Legal Disclaimer"):
            st.warning("""
            **Important Notice:**
            
            This tool is for informational purposes only and does not constitute legal advice. 
            
            Always consult with qualified legal professionals for contract review and advice.
            
            The analysis results should be used as a starting point for further legal review, not as definitive legal guidance.
            """)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 12px; color: #666;">
            Built with Streamlit, Transformers, and spaCy
        </div>
        """, unsafe_allow_html=True)
    
    return {
        "confidence_threshold": confidence_threshold,
        "use_ml_detection": use_ml_detection,
        "use_pattern_detection": use_pattern_detection,
        "show_risk_assessment": show_risk_assessment,
        "show_recommendations": show_recommendations,
        "show_low_confidence": show_low_confidence,
        "severity_filter": severity_filter,
        "export_format": export_format.lower()
    } 