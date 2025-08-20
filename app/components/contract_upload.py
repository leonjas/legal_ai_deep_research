"""
Contract Upload Component for Streamlit UI
"""

import streamlit as st
from typing import Optional, Tuple
import tempfile
import os
from pathlib import Path

from app.utils.contract_processor import ContractProcessor
from app.models.contract_analyzer import ContractAnalyzer
from app.components.contract_analysis_ui import ContractAnalysisUI

class ContractUploadComponent:
    """Component for handling contract file uploads and analysis"""
    
    def __init__(self):
        self.processor = ContractProcessor()
        self.analyzer = ContractAnalyzer()
    
    def render(self):
        """Render the contract upload and analysis interface"""
        st.markdown("## 📄 Contract Analysis")
        st.markdown("""
        Upload a contract file to analyze for potentially unfair clauses. 
        Supported formats: PDF, DOCX, DOC, TXT
        """)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a contract file",
            type=['pdf', 'docx', 'doc', 'txt'],
            help="Upload a contract document for analysis"
        )
        
        if uploaded_file is not None:
            self._process_uploaded_file(uploaded_file)
    
    def _process_uploaded_file(self, uploaded_file):
        """Process the uploaded contract file"""
        try:
            # Display file info
            st.markdown("### 📋 File Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Filename", uploaded_file.name)
            with col2:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            with col3:
                st.metric("File Type", uploaded_file.type)
            
            # Extract text
            with st.spinner("Extracting text from document..."):
                try:
                    text, metadata = self.processor.extract_text_from_file(
                        file_content=uploaded_file.read(),
                        filename=uploaded_file.name
                    )
                    
                    # Reset file pointer for potential re-reading
                    uploaded_file.seek(0)
                    
                except Exception as e:
                    st.error(f"Error extracting text: {str(e)}")
                    return
            
            # Validate contract content
            validation = self.processor.validate_contract_content(text)
            self._display_validation_results(validation)
            
            if not validation['is_likely_contract']:
                st.warning("⚠️ This document may not be a legal contract. Analysis may not be accurate.")
                if not st.checkbox("Proceed anyway"):
                    return
            
            # Display extracted text preview
            self._display_text_preview(text, metadata)
            
            # Analysis options
            st.markdown("### ⚙️ Analysis Options")
            
            col1, col2 = st.columns(2)
            with col1:
                use_ml_detection = st.checkbox(
                    "Use ML Detection", 
                    value=True,
                    help="Use machine learning models for clause detection"
                )
            
            with col2:
                use_pattern_detection = st.checkbox(
                    "Use Pattern Detection",
                    value=True, 
                    help="Use rule-based pattern matching for clause detection"
                )
            
            # Advanced options
            with st.expander("🔧 Advanced Options"):
                confidence_threshold = st.slider(
                    "Confidence Threshold",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.3,
                    step=0.1,
                    help="Minimum confidence required for flagging clauses"
                )
                
                show_low_confidence = st.checkbox(
                    "Show Low Confidence Results",
                    value=False,
                    help="Include results below the confidence threshold"
                )
            
            # Analyze button
            if st.button("🔍 Analyze Contract", type="primary"):
                if not use_ml_detection and not use_pattern_detection:
                    st.error("Please select at least one detection method.")
                    return
                
                self._perform_analysis(
                    text, 
                    uploaded_file.name,
                    use_ml_detection,
                    use_pattern_detection,
                    confidence_threshold,
                    show_low_confidence
                )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    def _display_validation_results(self, validation):
        """Display contract validation results"""
        st.markdown("### ✅ Document Validation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            confidence_color = "green" if validation['confidence'] > 0.7 else "orange" if validation['confidence'] > 0.3 else "red"
            st.markdown(f"**Contract Confidence:** <span style='color: {confidence_color}'>{validation['confidence']:.1%}</span>", 
                       unsafe_allow_html=True)
        
        with col2:
            status = "✅ Likely Contract" if validation['is_likely_contract'] else "❌ May Not Be Contract"
            st.markdown(f"**Status:** {status}")
        
        with col3:
            st.markdown(f"**Indicators Found:** {len(validation['indicators_found'])}")
        
        # Show warnings if any
        if validation['warnings']:
            st.warning("⚠️ " + " | ".join(validation['warnings']))
        
        # Show found indicators
        if validation['indicators_found']:
            with st.expander("📋 Contract Indicators Found"):
                indicators_text = ", ".join(validation['indicators_found'][:10])
                if len(validation['indicators_found']) > 10:
                    indicators_text += f" ... and {len(validation['indicators_found']) - 10} more"
                st.markdown(indicators_text)
    
    def _display_text_preview(self, text: str, metadata: dict):
        """Display extracted text preview and metadata"""
        st.markdown("### 📄 Document Preview")
        
        # Metadata
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Characters", f"{metadata['character_count']:,}")
        with col2:
            st.metric("Words", f"{metadata['word_count']:,}")
        with col3:
            st.metric("File Type", metadata['file_type'])
        with col4:
            st.metric("File Size", f"{metadata['file_size']:,} bytes")
        
        # Text preview
        with st.expander("📖 View Extracted Text (First 2000 characters)"):
            preview_text = text[:2000]
            if len(text) > 2000:
                preview_text += "..."
            st.text_area("Extracted Text", preview_text, height=300, disabled=True)
        
        # Contract sections if available
        sections = self.processor.segment_contract_sections(text)
        if len(sections) > 1:
            with st.expander("📑 Identified Contract Sections"):
                for section_name, section_content in sections.items():
                    if section_content.strip():
                        st.markdown(f"**{section_name.replace('_', ' ').title()}:** {len(section_content.split())} words")
        
        # Key information extraction
        key_info = self.processor.extract_key_information(text)
        if any(key_info.values()):
            with st.expander("🔑 Key Information Extracted"):
                if key_info['parties']:
                    st.markdown(f"**Parties:** {', '.join(key_info['parties'])}")
                if key_info['effective_date']:
                    st.markdown(f"**Effective Date:** {key_info['effective_date']}")
                if key_info['governing_law']:
                    st.markdown(f"**Governing Law:** {key_info['governing_law']}")
                if key_info['contract_type']:
                    st.markdown(f"**Contract Type:** {key_info['contract_type'].title()}")
    
    def _perform_analysis(self, text: str, filename: str, use_ml: bool, use_patterns: bool, 
                         confidence_threshold: float, show_low_confidence: bool):
        """Perform the contract analysis"""
        
        with st.spinner("🔍 Analyzing contract for unfair clauses..."):
            try:
                # Modify analyzer behavior based on options
                original_method = self.analyzer.analyze_contract
                
                def custom_analyze(contract_text):
                    # Get base analysis
                    result = original_method(contract_text)
                    
                    # Filter based on detection method preferences
                    filtered_clauses = []
                    for clause in result.unfair_clauses:
                        include_clause = False
                        
                        # Check if clause should be included based on detection method
                        if use_patterns and clause.clause_type != "ml_detected":
                            include_clause = True
                        elif use_ml and clause.clause_type == "ml_detected":
                            include_clause = True
                        
                        # Apply confidence filtering
                        if include_clause:
                            if clause.confidence >= confidence_threshold:
                                filtered_clauses.append(clause)
                            elif show_low_confidence:
                                filtered_clauses.append(clause)
                    
                    # Update result with filtered clauses
                    result.unfair_clauses = filtered_clauses
                    
                    # Recalculate risk score
                    if filtered_clauses:
                        result.overall_risk_score = self.analyzer._calculate_risk_score(
                            filtered_clauses, result.total_sentences
                        )
                    else:
                        result.overall_risk_score = 0.0
                    
                    # Update summary and recommendations
                    result.summary = self.analyzer._generate_summary(filtered_clauses, result.overall_risk_score)
                    result.recommendations = self.analyzer._generate_recommendations(filtered_clauses)
                    
                    return result
                
                # Perform analysis
                result = custom_analyze(text)
                
                # Display results
                st.markdown("---")
                st.markdown("# 📊 Analysis Results")
                
                ContractAnalysisUI.render_analysis_results(result, text)
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.exception(e)  # For debugging
    
    def render_text_input_option(self):
        """Render option to input contract text directly"""
        st.markdown("### ✏️ Or Enter Contract Text Directly")
        
        contract_text = st.text_area(
            "Paste contract text here:",
            height=200,
            placeholder="Paste your contract text here for analysis..."
        )
        
        if contract_text.strip():
            if st.button("🔍 Analyze Text", type="secondary"):
                # Validate the text
                validation = self.processor.validate_contract_content(contract_text)
                
                if validation['is_likely_contract'] or st.checkbox("Proceed with analysis anyway"):
                    with st.spinner("Analyzing contract text..."):
                        result = self.analyzer.analyze_contract(contract_text)
                        
                        st.markdown("---")
                        st.markdown("# 📊 Analysis Results")
                        ContractAnalysisUI.render_analysis_results(result, contract_text)
    
    def render_sample_contracts(self):
        """Render sample contracts for testing"""
        st.markdown("### 📚 Try Sample Contracts")
        
        sample_contracts = {
            "Employment Agreement (Problematic)": self._get_sample_employment_contract(),
            "Service Agreement (Moderate Issues)": self._get_sample_service_contract(),
            "Software License (Aggressive Terms)": self._get_sample_license_contract()
        }
        
        selected_sample = st.selectbox(
            "Select a sample contract:",
            options=[""] + list(sample_contracts.keys())
        )
        
        if selected_sample and selected_sample in sample_contracts:
            if st.button(f"🔍 Analyze {selected_sample}", type="secondary"):
                contract_text = sample_contracts[selected_sample]
                
                with st.spinner(f"Analyzing {selected_sample}..."):
                    result = self.analyzer.analyze_contract(contract_text)
                    
                    st.markdown("---")
                    st.markdown(f"# 📊 Analysis Results - {selected_sample}")
                    ContractAnalysisUI.render_analysis_results(result, contract_text)
    
    def _get_sample_employment_contract(self):
        """Return a sample employment contract with several unfair clauses"""
        return """
        EMPLOYMENT AGREEMENT
        
        This Employment Agreement is entered into between ABC Corporation ("Company") and Employee.
        
        1. TERMINATION: Company may terminate this agreement at any time, for any reason or no reason, 
        with or without cause, and without notice to Employee. Employee may only terminate this agreement 
        by providing 30 days written notice to Company.
        
        2. AUTOMATIC RENEWAL: This agreement shall automatically renew for successive one-year terms unless 
        Company provides notice of non-renewal at least 90 days before expiration. Employee must provide 
        180 days notice to prevent automatic renewal.
        
        3. INDEMNIFICATION: Employee agrees to indemnify and hold harmless Company from any and all claims, 
        damages, losses, and expenses regardless of cause, including Company's own negligence.
        
        4. LIQUIDATED DAMAGES: In the event of breach by Employee, Employee shall pay Company liquidated 
        damages of $50,000, which the parties agree is reasonable compensation for any damages.
        
        5. ARBITRATION: Any disputes must be resolved through binding arbitration. Employee waives the 
        right to a jury trial and class action participation.
        
        This agreement is governed by the laws of State of Delaware.
        """
    
    def _get_sample_service_contract(self):
        """Return a sample service contract with moderate issues"""
        return """
        SERVICE AGREEMENT
        
        This Service Agreement is between ServiceCorp ("Provider") and Client for consulting services.
        
        1. TERM: This agreement commences on the effective date and continues until terminated by either party.
        Provider may terminate immediately for any reason. Client must provide 60 days notice for termination.
        
        2. LIABILITY: Provider's maximum liability shall not exceed the fees paid in the prior 12 months.
        Provider disclaims all warranties and shall not be liable for any indirect or consequential damages.
        
        3. RENEWAL: This agreement automatically extends for additional one-year periods unless either party 
        provides 30 days written notice of non-renewal.
        
        4. DISPUTE RESOLUTION: All disputes shall be resolved through mandatory arbitration in Provider's 
        home jurisdiction.
        
        This agreement shall be governed by the laws of New York.
        """
    
    def _get_sample_license_contract(self):
        """Return a sample software license with aggressive terms"""
        return """
        SOFTWARE LICENSE AGREEMENT
        
        This License Agreement governs the use of Software provided by TechCorp ("Licensor").
        
        1. TERMINATION: Licensor may terminate this license immediately upon notice for any reason or no reason.
        Upon termination, User must immediately cease all use and destroy all copies.
        
        2. WARRANTIES: SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTIES. LICENSOR DISCLAIMS ALL WARRANTIES, 
        EXPRESS OR IMPLIED, INCLUDING MERCHANTABILITY AND FITNESS FOR PURPOSE.
        
        3. LIABILITY: IN NO EVENT SHALL LICENSOR BE LIABLE FOR ANY DAMAGES, regardless of the cause. 
        User assumes all risks associated with software use.
        
        4. INDEMNIFICATION: User shall defend, indemnify and hold harmless Licensor from all claims 
        arising from User's use of the software, including third-party claims.
        
        5. AUTOMATIC RENEWAL: This license automatically renews annually at Licensor's then-current rates 
        unless User cancels at least 90 days before renewal.
        
        6. BINDING ARBITRATION: All disputes must be resolved through individual binding arbitration. 
        User waives all rights to class action or jury trial.
        
        This agreement is governed by California law.
        """
