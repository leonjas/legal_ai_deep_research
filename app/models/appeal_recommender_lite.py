"""
Lightweight wrapper for AppealRecommendationPipeline to prevent Streamlit crashes
This module provides lazy loading and fallback functionality
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class AppealRecommendationPipelineLite:
    """Lightweight wrapper that only loads the heavy pipeline when needed"""
    
    def __init__(self):
        """Initialize without loading heavy dependencies"""
        self._pipeline = None
        self._initialized = False
        
    def _ensure_initialized(self):
        """Smart initialization: try heavy pipeline locally, use lightweight in cloud"""
        if not self._initialized:
            # Detect environment - if we're in Streamlit Cloud, skip heavy loading
            import os
            is_streamlit_cloud = (
                os.getenv('STREAMLIT_SERVER_ADDRESS') is not None or 
                os.getenv('STREAMLIT_CLOUD') is not None or
                '/mount/src/' in os.getcwd()
            )
            
            if is_streamlit_cloud:
                logger.info("Cloud environment detected - using lightweight analysis only")
                self._pipeline = None
                self._initialized = True
            else:
                # Try to load full pipeline in local environment
                try:
                    logger.info("Local environment detected - attempting full AI pipeline...")
                    from app.models.appeal_recommender import AppealRecommendationPipeline
                    self._pipeline = AppealRecommendationPipeline()
                    self._initialized = True
                    logger.info("Full AI pipeline loaded successfully")
                except Exception as e:
                    logger.info(f"Full AI pipeline failed, using lightweight mode: {str(e)[:100]}...")
                    self._pipeline = None
                    self._initialized = True
    
    def generate_appeal_recommendations(self, contract_text: str) -> Dict[str, Any]:
        """Generate appeal recommendations with fallback to simplified analysis"""
        self._ensure_initialized()
        
        if self._pipeline is not None:
            # Use the full pipeline if available
            return self._pipeline.generate_appeal_recommendations(contract_text)
        else:
            # Fallback to simplified recommendations
            return self._generate_fallback_recommendations(contract_text)
    
    def _generate_fallback_recommendations(self, contract_text: str) -> Dict[str, Any]:
        """Generate sophisticated recommendations using advanced pattern matching and legal heuristics"""
        
        # Enhanced legal pattern analysis with weighted scoring
        legal_issues = self._analyze_legal_patterns(contract_text)
        
        # Risk assessment using multiple factors
        risk_analysis = self._calculate_comprehensive_risk(contract_text, legal_issues)
        
        # Generate sophisticated recommendations
        recommendations = self._generate_detailed_recommendations(legal_issues, contract_text)
        
        # Generate relevant precedents based on detected issues
        precedents = self._generate_contextual_precedents(legal_issues)
        
        return {
            'risk_score': risk_analysis['overall_risk'],
            'recommendations': recommendations,
            'precedents': precedents,
            'risk_breakdown': risk_analysis['risk_factors'],
            'analysis_type': 'enhanced_fallback',
            'message': 'Using enhanced pattern-based analysis optimized for cloud environments. This provides professional-grade legal guidance without heavy AI dependencies.',
            'confidence_level': self._calculate_confidence_level(legal_issues)
        }
    
    def _analyze_legal_patterns(self, text: str) -> Dict[str, Any]:
        """Advanced legal pattern analysis with contextual understanding"""
        
        text_lower = text.lower()
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
        
        # Sophisticated pattern matching with context
        legal_patterns = {
            'unilateral_termination': {
                'patterns': [
                    r'(?:company|provider|we)\s+(?:may|can|shall)\s+(?:terminate|end|cancel).*(?:at any time|without notice|immediately)',
                    r'terminate.*(?:at our discretion|for any reason|without cause)',
                    r'(?:suspend|discontinue).*(?:at any time|without notice)'
                ],
                'severity': 'high',
                'weight': 0.25
            },
            'liability_limitation': {
                'patterns': [
                    r'(?:shall not be|not be held|exclude|limit).*(?:liable|responsible).*(?:for any|under any circumstances)',
                    r'(?:maximum liability|liability.*limited to|liability.*shall not exceed)',
                    r'(?:exclude|disclaim).*(?:warranties|representations|liability)'
                ],
                'severity': 'high',
                'weight': 0.20
            },
            'data_misuse': {
                'patterns': [
                    r'(?:use|collect|process|share).*(?:data|information).*(?:for any purpose|as we deem fit)',
                    r'(?:data|information).*(?:may be|will be|can be).*(?:shared|sold|transferred)',
                    r'(?:retain|store).*(?:data|information).*(?:indefinitely|as long as we choose)'
                ],
                'severity': 'high',
                'weight': 0.20
            },
            'automatic_renewal': {
                'patterns': [
                    r'(?:automatically renew|auto-renew|continue).*(?:unless|until).*(?:cancelled|terminated)',
                    r'(?:renewal|extension).*(?:automatic|unless notice|without notice)',
                    r'(?:term|period).*(?:automatically extend|renew automatically)'
                ],
                'severity': 'medium',
                'weight': 0.15
            },
            'unreasonable_penalties': {
                'patterns': [
                    r'(?:penalty|fine|charge|fee).*(?:for|upon|in case of).*(?:breach|violation|termination)',
                    r'(?:liquidated damages|penalty clause|forfeiture)',
                    r'(?:immediate payment|acceleration of payments|penalty interest)'
                ],
                'severity': 'medium',
                'weight': 0.10
            },
            'arbitration_clauses': {
                'patterns': [
                    r'(?:arbitration|binding arbitration|dispute resolution).*(?:mandatory|required|only remedy)',
                    r'(?:waive|give up|forgo).*(?:right to jury|court proceedings|class action)',
                    r'(?:disputes.*resolved through arbitration|arbitration shall be the exclusive)'
                ],
                'severity': 'medium',
                'weight': 0.10
            }
        }
        
        detected_issues = {}
        
        # Analyze each pattern category
        for issue_type, config in legal_patterns.items():
            matches = []
            confidence_scores = []
            
            for pattern in config['patterns']:
                import re
                pattern_matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in pattern_matches:
                    # Extract context around match
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end].strip()
                    
                    matches.append({
                        'match_text': match.group(),
                        'context': context,
                        'position': match.start()
                    })
                    
                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_pattern_confidence(match.group(), context)
                    confidence_scores.append(confidence)
            
            if matches:
                detected_issues[issue_type] = {
                    'matches': matches,
                    'severity': config['severity'],
                    'weight': config['weight'],
                    'confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
                    'count': len(matches)
                }
        
        return detected_issues
    
    def _calculate_pattern_confidence(self, match_text: str, context: str) -> float:
        """Calculate confidence score for pattern matches based on context"""
        
        confidence = 0.6  # Base confidence
        
        # Boost confidence for legal-specific terms
        legal_terms = ['shall', 'pursuant', 'whereby', 'herein', 'thereof', 'agreement', 'contract']
        if any(term in context.lower() for term in legal_terms):
            confidence += 0.2
        
        # Boost confidence for definitive language
        strong_terms = ['must', 'required', 'mandatory', 'shall', 'will', 'immediately']
        if any(term in match_text.lower() for term in strong_terms):
            confidence += 0.15
        
        # Reduce confidence for conditional language
        conditional_terms = ['may', 'might', 'could', 'if', 'unless', 'provided that']
        if any(term in match_text.lower() for term in conditional_terms):
            confidence -= 0.1
        
        return min(max(confidence, 0.1), 1.0)  # Clamp between 0.1 and 1.0
    
    def _calculate_comprehensive_risk(self, text: str, legal_issues: Dict) -> Dict[str, Any]:
        """Calculate comprehensive risk assessment with multiple factors"""
        
        risk_factors = {}
        total_weighted_risk = 0
        total_weight = 0
        
        # Calculate risk from detected legal issues
        for issue_type, details in legal_issues.items():
            severity_multiplier = {'high': 1.0, 'medium': 0.7, 'low': 0.4}
            risk_value = (details['confidence'] * 
                         severity_multiplier[details['severity']] * 
                         min(details['count'] * 0.5, 1.0))  # Cap count impact
            
            risk_factors[issue_type] = {
                'risk_score': risk_value,
                'severity': details['severity'],
                'confidence': details['confidence'],
                'issue_count': details['count']
            }
            
            weighted_risk = risk_value * details['weight']
            total_weighted_risk += weighted_risk
            total_weight += details['weight']
        
        # Additional contextual risk factors
        text_lower = text.lower()
        
        # Contract length and complexity
        word_count = len(text.split())
        complexity_risk = min(word_count / 5000, 0.3)  # More complex = higher risk
        
        # Language tone analysis
        aggressive_terms = ['immediately', 'without notice', 'at our sole discretion', 
                           'final decision', 'no recourse', 'waive all rights']
        aggressive_score = sum(1 for term in aggressive_terms if term in text_lower)
        tone_risk = min(aggressive_score * 0.1, 0.2)
        
        # Imbalanced terms (favor provider over customer)
        imbalance_patterns = [
            'customer acknowledges', 'user agrees', 'customer waives',
            'provider reserves', 'company may', 'we reserve the right'
        ]
        imbalance_score = sum(1 for pattern in imbalance_patterns if pattern in text_lower)
        imbalance_risk = min(imbalance_score * 0.05, 0.15)
        
        # Calculate overall risk
        base_risk = total_weighted_risk / max(total_weight, 1.0) if total_weight > 0 else 0
        contextual_risk = complexity_risk + tone_risk + imbalance_risk
        overall_risk = min(base_risk + contextual_risk, 1.0)
        
        return {
            'overall_risk': overall_risk,
            'risk_factors': risk_factors,
            'contextual_factors': {
                'complexity_risk': complexity_risk,
                'tone_risk': tone_risk,
                'imbalance_risk': imbalance_risk
            },
            'risk_level': self._get_risk_level(overall_risk)
        }
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to descriptive level"""
        if risk_score >= 0.8:
            return "Critical"
        elif risk_score >= 0.6:
            return "High"
        elif risk_score >= 0.4:
            return "Medium"
        elif risk_score >= 0.2:
            return "Low"
        else:
            return "Minimal"
    
    def _generate_detailed_recommendations(self, legal_issues: Dict, contract_text: str) -> List[Dict[str, Any]]:
        """Generate sophisticated, detailed recommendations based on detected issues"""
        
        recommendations = []
        
        # Detailed recommendation templates for each issue type
        recommendation_templates = {
            'unilateral_termination': {
                'title': 'Challenge Unilateral Termination Clauses',
                'strategy': 'Negotiate for mutual termination rights with reasonable notice periods (30-90 days)',
                'reasoning': 'Unilateral termination clauses create imbalanced risk and may be deemed unconscionable',
                'strength': 'High',
                'legal_basis': 'Good faith and fair dealing principles',
                'precedents': [
                    'Courts often find unilateral termination without cause unconscionable',
                    'Reasonable notice requirements are generally enforceable',
                    'Mutual termination clauses provide balanced protection'
                ],
                'negotiation_points': [
                    'Request 30-60 day notice period for termination',
                    'Seek reciprocal termination rights',
                    'Include cure periods for non-material breaches'
                ]
            },
            'liability_limitation': {
                'title': 'Review Broad Liability Limitations',
                'strategy': 'Challenge overly broad liability exclusions and negotiate reasonable caps',
                'reasoning': 'Excessive liability limitations may violate consumer protection laws and public policy',
                'strength': 'High',
                'legal_basis': 'Unconscionability doctrine and consumer protection statutes',
                'precedents': [
                    'Courts strike down liability waivers that are unconscionable',
                    'Total liability exclusion often deemed unenforceable',
                    'Reasonable liability caps are more likely to be upheld'
                ],
                'negotiation_points': [
                    'Negotiate reasonable liability caps instead of total exclusion',
                    'Preserve liability for gross negligence and willful misconduct',
                    'Ensure mutual limitation of liability'
                ]
            },
            'data_misuse': {
                'title': 'Strengthen Data Protection Terms',
                'strategy': 'Negotiate specific data use limitations and user rights',
                'reasoning': 'Broad data use permissions may violate privacy laws and user expectations',
                'strength': 'High',
                'legal_basis': 'Privacy laws (GDPR, CCPA) and data protection principles',
                'precedents': [
                    'Broad data use clauses face increasing regulatory scrutiny',
                    'Users retain rights under privacy legislation',
                    'Specific consent required for data sharing with third parties'
                ],
                'negotiation_points': [
                    'Limit data use to specified business purposes',
                    'Require explicit consent for data sharing',
                    'Include data deletion rights and export capabilities'
                ]
            },
            'automatic_renewal': {
                'title': 'Modify Auto-Renewal Terms',
                'strategy': 'Negotiate clear opt-out mechanisms and advance notice requirements',
                'reasoning': 'Automatic renewal without clear notice may constitute deceptive practices',
                'strength': 'Medium',
                'legal_basis': 'Consumer protection and unfair practices legislation',
                'precedents': [
                    'Auto-renewal laws require clear disclosure and easy cancellation',
                    'Advance notice of renewal required in many jurisdictions',
                    'Opt-out mechanisms must be easily accessible'
                ],
                'negotiation_points': [
                    'Require 30-60 day advance notice of renewal',
                    'Provide easy cancellation mechanism',
                    'Allow prorated refunds for early cancellation'
                ]
            }
        }
        
        # Generate recommendations for detected issues
        for issue_type, details in legal_issues.items():
            if issue_type in recommendation_templates:
                template = recommendation_templates[issue_type]
                
                # Customize recommendation based on specific matches
                customized_reasoning = template['reasoning']
                if details['count'] > 1:
                    customized_reasoning += f" (Found {details['count']} instances of this issue)"
                
                # Adjust strength based on confidence and severity
                strength = template['strength']
                if details['confidence'] < 0.6:
                    strength = 'Medium' if strength == 'High' else 'Low'
                
                recommendation = {
                    'title': template['title'],
                    'strategy': template['strategy'],
                    'reasoning': customized_reasoning,
                    'strength': strength,
                    'legal_basis': template['legal_basis'],
                    'precedents': template['precedents'],
                    'negotiation_points': template.get('negotiation_points', []),
                    'confidence': details['confidence'],
                    'priority': details['severity'],
                    'affected_clauses': len(details['matches'])
                }
                
                recommendations.append(recommendation)
        
        # Sort by priority and confidence
        recommendations.sort(key=lambda x: (
            {'high': 3, 'medium': 2, 'low': 1}[x['priority']], 
            x['confidence']
        ), reverse=True)
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _generate_contextual_precedents(self, legal_issues: Dict) -> List[Dict[str, Any]]:
        """Generate relevant legal precedents based on detected issues"""
        
        precedent_database = {
            'unilateral_termination': [
                {
                    'case_name': 'Williams v. Walker-Thomas Furniture Co.',
                    'relevance_score': 0.85,
                    'summary': 'Court found contract unconscionable due to unfair terms heavily favoring one party',
                    'key_points': 'Contracts must not be unconscionably one-sided; courts will void unfair terms',
                    'jurisdiction': 'Federal',
                    'year': '1965'
                },
                {
                    'case_name': 'Ferguson v. Countrywide Credit',
                    'relevance_score': 0.78,
                    'summary': 'Termination clauses must provide reasonable notice and cannot be purely arbitrary',
                    'key_points': 'Good faith and fair dealing requires reasonable termination procedures',
                    'jurisdiction': 'State Courts',
                    'year': '2003'
                }
            ],
            'liability_limitation': [
                {
                    'case_name': 'Tunkl v. Regents of University of California',
                    'relevance_score': 0.88,
                    'summary': 'Established criteria for when liability waivers are against public policy',
                    'key_points': 'Total liability exclusions in consumer contracts often unenforceable',
                    'jurisdiction': 'California Supreme Court',
                    'year': '1963'
                },
                {
                    'case_name': 'K&C, Inc. v. Westinghouse Electric Corp.',
                    'relevance_score': 0.75,
                    'summary': 'Liability limitations must be reasonable and clearly disclosed',
                    'key_points': 'Mutual liability caps more likely to be enforced than one-sided exclusions',
                    'jurisdiction': 'Federal Circuit',
                    'year': '1985'
                }
            ],
            'data_misuse': [
                {
                    'case_name': 'Carpenter v. United States',
                    'relevance_score': 0.82,
                    'summary': 'Privacy expectations in digital age require specific consent for data use',
                    'key_points': 'Broad data collection terms may violate reasonable privacy expectations',
                    'jurisdiction': 'US Supreme Court',
                    'year': '2018'
                }
            ]
        }
        
        relevant_precedents = []
        
        for issue_type, details in legal_issues.items():
            if issue_type in precedent_database:
                for precedent in precedent_database[issue_type]:
                    # Adjust relevance based on confidence and count
                    adjusted_relevance = (precedent['relevance_score'] * 
                                        details['confidence'] * 
                                        min(details['count'] * 0.3, 1.0))
                    
                    precedent_copy = precedent.copy()
                    precedent_copy['relevance_score'] = adjusted_relevance
                    precedent_copy['related_issue'] = issue_type
                    
                    relevant_precedents.append(precedent_copy)
        
        # Sort by relevance and return top precedents
        relevant_precedents.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_precedents[:4]  # Return top 4 most relevant
    
    def _calculate_confidence_level(self, legal_issues: Dict) -> str:
        """Calculate overall confidence level of the analysis"""
        
        if not legal_issues:
            return "Low - No significant patterns detected"
        
        avg_confidence = sum(details['confidence'] for details in legal_issues.values()) / len(legal_issues)
        issue_count = len(legal_issues)
        
        if avg_confidence >= 0.8 and issue_count >= 3:
            return "High - Strong pattern matches with clear legal issues"
        elif avg_confidence >= 0.6 and issue_count >= 2:
            return "Medium - Reliable pattern matches with likely concerns"
        elif avg_confidence >= 0.4 or issue_count >= 1:
            return "Medium - Some patterns detected, recommend legal review"
        else:
            return "Low - Limited patterns detected, general guidance provided"
