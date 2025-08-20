"""
Contract Analysis Components for Streamlit UI
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
from app.models.contract_analyzer import UnfairClause
from app.models.unfair_pipeline import UnfairDetectionResult
from typing import List, Dict, Any, Optional
import io
import base64
import tempfile
import os
from datetime import datetime

from app.models.contract_analyzer import ContractAnalysisResult, UnfairClause
from app.models.unfair_pipeline import UnfairDetectionResult, run_unfair_pipeline, run_unfair_pipeline_text

class ContractAnalysisUI:
    """UI components for contract analysis results"""
    
    @staticmethod
    def render_pipeline_upload():
        """Render file upload interface for the new modular pipeline"""
        st.markdown("## 📄 Upload Contract for Analysis")
        
        # Configuration options
        with st.expander("⚙️ Advanced Settings", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                # Model selection
                model_options = {
                    "marmolpen3/lexglue-unfair-tos": "UNFAIR-ToS (Recommended)",
                    "CodeHima/TOSRobertaV2": "TOSRobertaV2 (Alternative)",
                    "nlpaueb/legal-bert-base-uncased": "Legal BERT (General)"
                }
                selected_model = st.selectbox(
                    "Select Detection Model:",
                    options=list(model_options.keys()),
                    format_func=lambda x: model_options[x]
                )
            
            with col2:
                # Confidence threshold
                confidence_threshold = st.slider(
                    "Confidence Threshold:",
                    min_value=0.1,
                    max_value=0.95,
                    value=0.60,
                    step=0.05,
                    help="Higher values = more strict detection"
                )
                
                # Minimum words per clause
                min_words = st.number_input(
                    "Min Words per Clause:",
                    min_value=2,
                    max_value=20,
                    value=4,
                    help="Shorter clauses will be ignored"
                )
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a PDF contract file",
            type=['pdf'],
            help="Upload a PDF contract to analyze for unfair clauses"
        )
        
        # Text input option
        st.markdown("**Or paste contract text directly:**")
        contract_text = st.text_area(
            "Contract Text",
            height=200,
            placeholder="Paste your contract text here..."
        )
        
        # Analysis button
        if st.button("🔍 Analyze Contract", type="primary"):
            if uploaded_file is not None or contract_text.strip():
                with st.spinner("Analyzing contract for unfair clauses..."):
                    try:
                        if uploaded_file:
                            # Save uploaded file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                tmp_file.write(uploaded_file.read())
                                tmp_path = tmp_file.name
                            
                            # Run pipeline on PDF
                            result = run_unfair_pipeline(
                                file_path=tmp_path,
                                model_name=selected_model,
                                min_conf=confidence_threshold,
                                min_words=min_words
                            )
                            
                            # Clean up temp file
                            os.unlink(tmp_path)
                            
                            # Display results
                            ContractAnalysisUI.render_pipeline_results(result)
                            
                        else:
                            # Run pipeline on text
                            result = run_unfair_pipeline_text(
                                text=contract_text,
                                model_name=selected_model,
                                min_conf=confidence_threshold,
                                min_words=min_words
                            )
                            
                            # Display results
                            ContractAnalysisUI.render_text_results(result)
                            
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        st.error("Please check the file format and try again.")
            else:
                st.warning("⚠️ Please upload a PDF file or enter contract text to analyze.")
    
    @staticmethod
    def render_pipeline_results(result: UnfairDetectionResult):
        """Render results from the modular pipeline"""
        
        # Header with summary
        summary = result.get_summary()
        
        st.success(f"✅ Analysis completed in {summary['processing_time']:.2f} seconds")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("File", summary['file_name'])
        
        with col2:
            st.metric("Total Clauses", summary['total_clauses'])
        
        with col3:
            st.metric("Unfair Clauses", summary['unfair_count'])
        
        with col4:
            unfair_pct = summary['unfair_percentage']
            st.metric("Unfair %", f"{unfair_pct:.1f}%")
        
        # Risk assessment
        if unfair_pct > 20:
            st.error(f"🚨 High Risk: {unfair_pct:.1f}% of clauses may be unfair")
        elif unfair_pct > 10:
            st.warning(f"⚠️ Medium Risk: {unfair_pct:.1f}% of clauses may be unfair")
        elif unfair_pct > 0:
            st.info(f"ℹ️ Low Risk: {unfair_pct:.1f}% of clauses may be unfair")
        else:
            st.success("✅ No unfair clauses detected!")
        
        # Model info
        st.info(f"📊 Analysis performed using: {summary['model_used']}")
        
        # Detailed results
        if result.unfair_clauses:
            st.markdown("## 🚨 Detected Unfair Clauses")
            
            # Sort by confidence
            sorted_clauses = sorted(result.unfair_clauses, 
                                  key=lambda x: x.confidence, reverse=True)
            
            for i, clause in enumerate(sorted_clauses, 1):
                with st.expander(f"📋 Clause #{i} - {clause.clause_type} "
                               f"(Confidence: {clause.confidence:.1%})"):
                    
                    # Display clause text
                    st.markdown("**Clause Text:**")
                    st.markdown(f'<div style="background-color: #ffebee; padding: 15px; '
                              f'border-left: 4px solid #f44336; border-radius: 4px; '
                              f'font-style: italic; margin: 10px 0;">'
                              f'{clause.text}</div>', unsafe_allow_html=True)
                    
                    # Confidence and prediction details
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Type:** {clause.clause_type}")
                        st.markdown(f"**Confidence:** {clause.confidence:.2%}")
                    
                    with col2:
                        st.markdown(f"**Position:** Index {clause.sentence_index}")
                        st.markdown(f"**Length:** {len(clause.text.split())} words")
            
            # Export options
            ContractAnalysisUI._render_pipeline_export(result)
        
        else:
            st.success("✅ No potentially unfair clauses were detected in this contract.")
    
    @staticmethod
    def render_text_results(result: Dict):
        """Render results from text-based analysis"""
        
        st.success(f"✅ Analysis completed in {result['processing_time']:.2f} seconds")
        st.info(result['summary'])
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Clauses", result['total_clauses'])
        
        with col2:
            st.metric("Unfair Clauses", len(result['unfair_clauses']))
        
        with col3:
            if result['total_clauses'] > 0:
                unfair_pct = (len(result['unfair_clauses']) / result['total_clauses']) * 100
                st.metric("Unfair %", f"{unfair_pct:.1f}%")
        
        # Model info
        model_info = result.get('model_info', {})
        if model_info:
            st.info(f"📊 Analysis performed using: {model_info.get('model_name', 'unknown')}")
        
        # Results display
        if result['unfair_clauses']:
            st.markdown("## 🚨 Detected Unfair Clauses")
            
            for i, clause in enumerate(result['unfair_clauses'], 1):
                with st.expander(f"📋 Clause #{i} - {clause.get('clause_type', 'Unknown')} "
                               f"(Confidence: {clause.get('confidence', 0):.1%})"):
                    
                    st.markdown("**Clause Text:**")
                    st.markdown(f'<div style="background-color: #ffebee; padding: 15px; '
                              f'border-left: 4px solid #f44336; border-radius: 4px; '
                              f'font-style: italic; margin: 10px 0;">'
                              f'{clause.get("text", "")}</div>', unsafe_allow_html=True)
                    
                    st.markdown(f"**Type:** {clause.get('clause_type', 'Unknown')}")
                    st.markdown(f"**Confidence:** {clause.get('confidence', 0):.2%}")
        
        else:
            st.success("✅ No potentially unfair clauses were detected in the provided text.")
    
    @staticmethod
    def _render_pipeline_export(result: UnfairDetectionResult):
        """Render export options for pipeline results"""
        st.markdown("## 📥 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Generate detailed report
            report = ContractAnalysisUI._generate_pipeline_report(result)
            st.download_button(
                label="📄 Download Report",
                data=report,
                file_name=f"unfair_clause_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        with col2:
            # Export unfair clauses to CSV
            if result.unfair_clauses:
                csv_data = ContractAnalysisUI._generate_pipeline_csv(result.unfair_clauses)
                st.download_button(
                    label="📊 Export to CSV",
                    data=csv_data,
                    file_name=f"unfair_clauses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            # Generate JSON export
            import json
            json_data = {
                "summary": result.get_summary(),
                "unfair_clauses": result.unfair_clauses,
                "model_info": result.model_info,
                "analysis_timestamp": datetime.now().isoformat()
            }
            st.download_button(
                label="🔗 Export to JSON",
                data=json.dumps(json_data, indent=2),
                file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    @staticmethod
    def _generate_pipeline_report(result: UnfairDetectionResult) -> str:
        """Generate a detailed report for pipeline results"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary = result.get_summary()
        
        report = f"""
UNFAIR CLAUSE DETECTION REPORT
==============================
Generated: {timestamp}
File: {summary['file_name']}
Model: {summary['model_used']}
Processing Time: {summary['processing_time']:.2f}s

EXECUTIVE SUMMARY
=================
Total Clauses Analyzed: {summary['total_clauses']}
Potentially Unfair Clauses: {summary['unfair_count']}
Risk Percentage: {summary['unfair_percentage']:.1f}%

DETAILED FINDINGS
=================
"""
        
        if result.unfair_clauses:
            for i, clause in enumerate(result.unfair_clauses, 1):
                report += f"""
Issue #{i}:
-----------
Type: {clause.clause_type}
Confidence: {clause.confidence:.2%}
Position: Index {clause.sentence_index}
Severity: {clause.severity}

Text:
"{clause.text}"

Explanation:
{clause.explanation}

{'='*50}
"""
        else:
            report += "No unfair clauses detected.\n"
        
        report += f"""

METHODOLOGY
===========
This analysis used the {summary['model_used']} model from Hugging Face,
specifically trained for unfair clause detection in Terms of Service documents.
The model analyzes each clause independently and provides confidence scores.

DISCLAIMER
==========
This automated analysis is for informational purposes only and does not
constitute legal advice. Please consult with qualified legal professionals
for authoritative contract interpretation.

END OF REPORT
"""
        
        return report
    
    @staticmethod
    def _generate_pipeline_csv(clauses: List[UnfairClause]) -> str:
        """Generate CSV export for unfair clauses"""
        data = []
        for i, clause in enumerate(clauses, 1):
            data.append({
                "Clause_Number": i,
                "Type": clause.clause_type,
                "Confidence": clause.confidence,
                "Position_Index": clause.sentence_index,
                "Severity": clause.severity,
                "Word_Count": len(clause.text.split()),
                "Text": clause.text.replace('"', '""'),  # Escape quotes
                "Explanation": clause.explanation.replace('"', '""')
            })
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False)
    
    @staticmethod
    def render_analysis_results(result: ContractAnalysisResult, contract_text: str):
        """Render the complete analysis results"""
        
        # Header with risk score
        ContractAnalysisUI._render_risk_header(result)
        
        # Summary
        st.markdown("## 📋 Analysis Summary")
        st.markdown(result.summary)
        
        # Unfair clauses section
        if result.unfair_clauses:
            ContractAnalysisUI._render_unfair_clauses(result.unfair_clauses, contract_text)
        else:
            st.success("✅ No potentially unfair clauses detected in this contract.")
        
        # Recommendations
        if result.recommendations:
            ContractAnalysisUI._render_recommendations(result.recommendations)
        
        # Visualizations
        ContractAnalysisUI._render_visualizations(result)
        
        # Detailed analysis
        ContractAnalysisUI._render_detailed_analysis(result)
        
        # Export options
        ContractAnalysisUI._render_export_options(result, contract_text)
    
    @staticmethod
    def _render_risk_header(result: ContractAnalysisResult):
        """Render the risk score header"""
        risk_score = result.overall_risk_score
        
        # Determine risk level and color
        if risk_score >= 0.7:
            risk_level = "High Risk"
            color = "🔴"
            alert_type = "error"
        elif risk_score >= 0.4:
            risk_level = "Medium Risk"
            color = "🟠"
            alert_type = "warning"
        else:
            risk_level = "Low Risk"
            color = "🟢"
            alert_type = "success"
        
        # Create columns for layout
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; border-radius: 10px; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; margin: 20px 0;">
                <h2 style="margin: 0; color: white;">{color} Contract Risk Assessment</h2>
                <h1 style="margin: 10px 0; color: white;">{risk_level}</h1>
                <p style="margin: 0; font-size: 18px; color: white;">
                    Risk Score: {risk_score:.1%}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show alert based on risk level
        if alert_type == "error":
            st.error(f"⚠️ This contract contains multiple potentially unfair clauses that require immediate attention.")
        elif alert_type == "warning":
            st.warning(f"⚡ This contract contains some clauses that should be reviewed carefully.")
        else:
            st.success(f"✅ This contract appears to have relatively standard terms.")
    
    @staticmethod
    def _render_unfair_clauses(clauses: List[UnfairClause], contract_text: str):
        """Render detected unfair clauses"""
        st.markdown("## 🚨 Potentially Unfair Clauses")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Clauses", len(clauses))
        
        with col2:
            high_severity = sum(1 for c in clauses if c.severity == "high")
            st.metric("High Severity", high_severity)
        
        with col3:
            avg_confidence = sum(c.confidence for c in clauses) / len(clauses)
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        with col4:
            unique_types = len(set(c.clause_type for c in clauses))
            st.metric("Issue Types", unique_types)
        
        # Sort clauses by severity and confidence
        sorted_clauses = sorted(clauses, 
                              key=lambda x: (
                                  {"high": 3, "medium": 2, "low": 1}[x.severity],
                                  x.confidence
                              ), reverse=True)
        
        # Display each clause
        for i, clause in enumerate(sorted_clauses, 1):
            with st.expander(f"📋 Issue #{i}: {clause.clause_type.replace('_', ' ').title()} "
                           f"({clause.severity.title()} Severity - {clause.confidence:.1%} Confidence)"):
                
                # Clause details
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Problematic Text:**")
                    st.markdown(f'<div style="background-color: #ffebee; padding: 10px; '
                              f'border-left: 4px solid #f44336; border-radius: 4px; '
                              f'font-style: italic;">{clause.text}</div>', 
                              unsafe_allow_html=True)
                
                with col2:
                    # Severity indicator
                    severity_colors = {"high": "🔴", "medium": "🟠", "low": "🟡"}
                    st.markdown(f"**Severity:** {severity_colors[clause.severity]} {clause.severity.title()}")
                    st.markdown(f"**Confidence:** {clause.confidence:.1%}")
                    st.markdown(f"**Position:** Sentence {clause.sentence_index + 1}")
                
                # Explanation
                st.markdown("**Why this might be unfair:**")
                st.markdown(clause.explanation)
                
                # Suggested action based on clause type
                suggestions = {
                    "unilateral_termination": "💡 **Suggestion:** Negotiate for mutual termination rights with reasonable notice periods.",
                    "automatic_renewal": "💡 **Suggestion:** Request explicit consent requirements for renewals or opt-out periods.",
                    "broad_indemnification": "💡 **Suggestion:** Limit indemnification scope and add mutual indemnification clauses.",
                    "limitation_of_liability": "💡 **Suggestion:** Ensure liability limitations are reasonable and mutual.",
                    "unreasonable_penalties": "💡 **Suggestion:** Review penalties for reasonableness and seek caps on damages.",
                    "mandatory_arbitration": "💡 **Suggestion:** Consider the implications of waiving jury trial rights."
                }
                
                if clause.clause_type in suggestions:
                    st.markdown(suggestions[clause.clause_type])
    
    @staticmethod
    def _render_recommendations(recommendations: List[str]):
        """Render recommendations"""
        st.markdown("## 🎯 Recommendations")
        
        for i, recommendation in enumerate(recommendations, 1):
            st.markdown(f"**{i}.** {recommendation}")
    
    @staticmethod
    def _render_visualizations(result: ContractAnalysisResult):
        """Render analysis visualizations"""
        if not result.unfair_clauses:
            return
            
        st.markdown("## 📊 Analysis Visualizations")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Clause Types", "Severity Distribution", "Risk Metrics"])
        
        with tab1:
            ContractAnalysisUI._render_clause_types_chart(result.unfair_clauses)
        
        with tab2:
            ContractAnalysisUI._render_severity_chart(result.unfair_clauses)
        
        with tab3:
            ContractAnalysisUI._render_risk_metrics(result)
    
    @staticmethod
    def _render_clause_types_chart(clauses: List[UnfairClause]):
        """Render clause types distribution chart"""
        # Count clause types
        clause_counts = {}
        for clause in clauses:
            clause_type = clause.clause_type.replace('_', ' ').title()
            clause_counts[clause_type] = clause_counts.get(clause_type, 0) + 1
        
        if clause_counts:
            # Create pie chart
            fig = px.pie(
                values=list(clause_counts.values()),
                names=list(clause_counts.keys()),
                title="Distribution of Potentially Unfair Clause Types"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
            # Also show as bar chart
            fig_bar = px.bar(
                x=list(clause_counts.keys()),
                y=list(clause_counts.values()),
                title="Count of Each Clause Type",
                labels={'x': 'Clause Type', 'y': 'Count'}
            )
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    @staticmethod
    def _render_severity_chart(clauses: List[UnfairClause]):
        """Render severity distribution chart"""
        # Count by severity
        severity_counts = {"High": 0, "Medium": 0, "Low": 0}
        for clause in clauses:
            severity_counts[clause.severity.title()] += 1
        
        # Create donut chart
        colors = ["#ff4444", "#ff8800", "#ffcc00"]
        fig = go.Figure(data=[go.Pie(
            labels=list(severity_counts.keys()),
            values=list(severity_counts.values()),
            hole=.3,
            marker_colors=colors
        )])
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(title="Severity Distribution of Detected Issues")
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence distribution
        confidences = [clause.confidence for clause in clauses]
        fig_hist = px.histogram(
            x=confidences,
            nbins=10,
            title="Confidence Score Distribution",
            labels={'x': 'Confidence Score', 'y': 'Number of Clauses'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    @staticmethod
    def _render_risk_metrics(result: ContractAnalysisResult):
        """Render risk metrics dashboard"""
        clauses = result.unfair_clauses
        
        # Calculate metrics
        high_severity_count = sum(1 for c in clauses if c.severity == "high")
        avg_confidence = sum(c.confidence for c in clauses) / len(clauses) if clauses else 0
        risk_density = len(clauses) / result.total_sentences if result.total_sentences > 0 else 0
        
        # Create gauge chart for overall risk
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=result.overall_risk_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Risk Score (%)"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics grid
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "High Severity Issues",
                high_severity_count,
                delta=f"{high_severity_count/len(clauses)*100:.1f}%" if clauses else "0%"
            )
        
        with col2:
            st.metric(
                "Average Confidence",
                f"{avg_confidence:.1%}",
                delta="Detection Quality"
            )
        
        with col3:
            st.metric(
                "Risk Density",
                f"{risk_density:.2%}",
                delta="Issues per sentence"
            )
    
    @staticmethod
    def _render_detailed_analysis(result: ContractAnalysisResult):
        """Render detailed analysis table"""
        if not result.unfair_clauses:
            return
            
        st.markdown("## 📋 Detailed Analysis Table")
        
        # Create DataFrame
        data = []
        for i, clause in enumerate(result.unfair_clauses, 1):
            data.append({
                "Issue #": i,
                "Type": clause.clause_type.replace('_', ' ').title(),
                "Severity": clause.severity.title(),
                "Confidence": f"{clause.confidence:.1%}",
                "Sentence": clause.sentence_index + 1,
                "Text Preview": clause.text[:100] + "..." if len(clause.text) > 100 else clause.text
            })
        
        df = pd.DataFrame(data)
        
        # Style the dataframe
        def color_severity(val):
            color = {'High': 'background-color: #ffebee', 
                    'Medium': 'background-color: #fff3e0',
                    'Low': 'background-color: #f1f8e9'}.get(val, '')
            return color
        
        styled_df = df.style.applymap(color_severity, subset=['Severity'])
        st.dataframe(styled_df, use_container_width=True)
    
    @staticmethod
    def _render_export_options(result: ContractAnalysisResult, contract_text: str):
        """Render export options"""
        st.markdown("## 📥 Export Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Generate report
            report = ContractAnalysisUI._generate_report(result, contract_text)
            st.download_button(
                label="📄 Download Detailed Report",
                data=report,
                file_name=f"contract_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        with col2:
            # Export to CSV
            if result.unfair_clauses:
                csv_data = ContractAnalysisUI._generate_csv(result.unfair_clauses)
                st.download_button(
                    label="📊 Export to CSV",
                    data=csv_data,
                    file_name=f"unfair_clauses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            # Generate audit report
            audit_report = ContractAnalysisUI._generate_audit_report(result)
            st.download_button(
                label="🔍 Audit Report",
                data=audit_report,
                file_name=f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    @staticmethod
    def _generate_report(result: ContractAnalysisResult, contract_text: str) -> str:
        """Generate detailed analysis report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
CONTRACT ANALYSIS REPORT
Generated on: {timestamp}

=== EXECUTIVE SUMMARY ===
{result.summary}

=== RISK ASSESSMENT ===
Overall Risk Score: {result.overall_risk_score:.1%}
Total Sentences Analyzed: {result.total_sentences}
Potentially Unfair Clauses Found: {len(result.unfair_clauses)}

=== DETAILED FINDINGS ===
"""
        
        for i, clause in enumerate(result.unfair_clauses, 1):
            report += f"""
Issue #{i}: {clause.clause_type.replace('_', ' ').title()}
Severity: {clause.severity.title()}
Confidence: {clause.confidence:.1%}
Location: Sentence {clause.sentence_index + 1}

Problematic Text:
"{clause.text}"

Explanation:
{clause.explanation}

{'='*60}
"""
        
        report += f"""
=== RECOMMENDATIONS ===
"""
        for i, rec in enumerate(result.recommendations, 1):
            report += f"{i}. {rec}\n"
        
        report += f"""
=== METHODOLOGY ===
This analysis was performed using a combination of:
1. Pattern-based detection for known unfair clause types
2. Machine learning classification using legal language models
3. Confidence scoring based on multiple detection methods
4. Risk assessment considering clause severity and frequency

=== DISCLAIMER ===
This analysis is for informational purposes only and does not constitute legal advice.
Please consult with a qualified attorney for legal interpretation of contract terms.

End of Report
"""
        
        return report
    
    @staticmethod
    def _generate_csv(clauses: List[UnfairClause]) -> str:
        """Generate CSV export of unfair clauses"""
        data = []
        for clause in clauses:
            data.append({
                "Clause_Type": clause.clause_type,
                "Severity": clause.severity,
                "Confidence": clause.confidence,
                "Sentence_Index": clause.sentence_index,
                "Text": clause.text.replace('"', '""'),  # Escape quotes
                "Explanation": clause.explanation.replace('"', '""')
            })
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False)
    
    @staticmethod
    def _generate_audit_report(result: ContractAnalysisResult) -> str:
        """Generate standardized audit report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate audit metrics
        high_risk_count = sum(1 for c in result.unfair_clauses if c.severity == "high")
        medium_risk_count = sum(1 for c in result.unfair_clauses if c.severity == "medium")
        low_risk_count = sum(1 for c in result.unfair_clauses if c.severity == "low")
        
        risk_level = "FAIL" if result.overall_risk_score > 0.7 else "REVIEW" if result.overall_risk_score > 0.3 else "PASS"
        
        audit_report = f"""
STANDARDIZED CONTRACT AUDIT REPORT
===================================
Audit Date: {timestamp}
Analysis Engine: LexGLUE Unfair Clause Detection System

AUDIT RESULT: {risk_level}
Overall Risk Score: {result.overall_risk_score:.1%}

EXECUTIVE SUMMARY:
- Total Clauses Analyzed: {result.total_sentences}
- Issues Identified: {len(result.unfair_clauses)}
- High Risk Issues: {high_risk_count}
- Medium Risk Issues: {medium_risk_count}
- Low Risk Issues: {low_risk_count}

RISK BREAKDOWN:
{"█" * int(result.overall_risk_score * 20)} {result.overall_risk_score:.1%}

COMPLIANCE STATUS:
{"❌ FAIL - Contract requires significant revision" if risk_level == "FAIL" 
 else "⚠️ REVIEW - Contract needs careful consideration" if risk_level == "REVIEW" 
 else "✅ PASS - Contract appears reasonable"}

ISSUE CATEGORIES DETECTED:
"""
        
        clause_types = set(c.clause_type for c in result.unfair_clauses)
        for clause_type in clause_types:
            count = sum(1 for c in result.unfair_clauses if c.clause_type == clause_type)
            audit_report += f"- {clause_type.replace('_', ' ').title()}: {count} instance(s)\n"
        
        audit_report += f"""
RECOMMENDED ACTIONS:
"""
        for i, rec in enumerate(result.recommendations, 1):
            audit_report += f"{i}. {rec}\n"
        
        audit_report += f"""
AUDITOR NOTES:
This automated audit identifies potentially unfair contract clauses based on
established legal precedents and industry standards. Manual legal review is
recommended for all contracts, especially those with FAIL or REVIEW ratings.

END OF AUDIT REPORT
"""
        
        return audit_report
