"""
Contract Summarization Module
Provides automatic summarization of contract content and key terms extraction
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

# Make spaCy optional
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

# Make nltk optional too
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    nltk = None

logger = logging.getLogger(__name__)

@dataclass
class ContractSummary:
    """Data class for contract summary results"""
    executive_summary: str
    key_terms: Dict[str, str]
    parties_involved: List[str]
    contract_type: str
    key_obligations: Dict[str, List[str]]
    important_dates: List[Dict[str, str]]
    financial_terms: Dict[str, str]
    termination_conditions: List[str]
    governing_law: str
    word_count: int
    estimated_read_time: int

@dataclass 
class EnhancedContractSummary:
    """Data class for enhanced analytical contract summary"""
    executive_summary: str
    sections_analysis: Dict[str, Dict]
    risk_prioritization: Dict[str, Any]
    clause_relationships: Dict[str, List[str]]
    actionable_insights: List[Dict[str, str]]
    negotiation_strategy: Dict[str, Any]
    compliance_concerns: List[Dict[str, str]]
    basic_summary: ContractSummary  # Include basic summary too

class ContractSummarizer:
    """
    Analyzes and summarizes contract content to extract key information
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._load_patterns()
    
    def _load_patterns(self):
        """Load regex patterns for extracting key contract elements"""
        self.patterns = {
            'parties': [
                r'(?i)between\s+([^,\n]+?)\s*\([^)]*\)\s*and\s+([^,\n]+?)\s*\([^)]*\)',
                r'(?i)this\s+agreement.*?between\s+([^,\n]+?)\s+and\s+([^,\n]+)',
                r'(?i)"([^"]+)"\s*\([^)]*(?:company|corporation|inc|llc)[^)]*\)',
                r'(?i)([A-Z][a-z]+\s+(?:[A-Z][a-z]+\s*)*(?:Inc|LLC|Corp|Corporation|Solutions|Systems|Services))'
            ],
            'contract_type': [
                r'(?i)(software|service|employment|lease|rental|licensing|subscription|terms\s+of\s+service|privacy\s+policy|agreement)',
                r'(?i)(contract|agreement)\s+(?:for|of)\s+([^,\n]+)',
            ],
            'financial_terms': [
                r'(?i)(fee|payment|cost|price|amount).*?(\$[\d,]+(?:\.\d{2})?)',
                r'(?i)(\$[\d,]+(?:\.\d{2})?)\s*(?:per|\/)\s*(month|year|hour|day)',
                r'(?i)(monthly|annual|yearly|weekly|hourly)\s+(?:fee|cost|payment).*?(\$[\d,]+(?:\.\d{2})?)',
                r'(?i)penalty.*?(\$[\d,]+(?:\.\d{2})?)',
            ],
            'dates': [
                r'(?i)(effective|start|begin|commence|end|expire|terminate|due).*?(\d{1,2}\/\d{1,2}\/\d{2,4}|\d{1,2}-\d{1,2}-\d{2,4}|[A-Z][a-z]+\s+\d{1,2},?\s+\d{2,4})',
                r'(?i)(\d+)\s+(day|week|month|year)s?\s+(?:notice|prior|advance)',
            ],
            'termination': [
                r'(?i)terminat[ei].*?(?:with|without)\s+(cause|notice|reason)',
                r'(?i)(?:may|can|shall)\s+(?:end|cancel|terminate).*?(?:at\s+any\s+time|immediately|without\s+notice)',
                r'(?i)automatic.*?(?:renewal|extension)',
            ],
            'governing_law': [
                r'(?i)govern[ed]*\s+by.*?laws?\s+of\s+([^,\n\.]+)',
                r'(?i)jurisdiction.*?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'(?i)courts?\s+of\s+([^,\n\.]+)',
            ]
        }
    
    def summarize_contract(self, contract_text: str) -> ContractSummary:
        """
        Generate comprehensive summary of contract content
        
        Args:
            contract_text: Full contract text
            
        Returns:
            ContractSummary with extracted information
        """
        self.logger.info("Starting contract summarization...")
        
        # Clean text
        cleaned_text = self._clean_text(contract_text)
        
        # Extract key components
        parties = self._extract_parties(cleaned_text)
        contract_type = self._identify_contract_type(cleaned_text)
        key_terms = self._extract_key_terms(cleaned_text)
        obligations = self._extract_obligations(cleaned_text)
        dates = self._extract_important_dates(cleaned_text)
        financial = self._extract_financial_terms(cleaned_text)
        termination = self._extract_termination_conditions(cleaned_text)
        governing_law = self._extract_governing_law(cleaned_text)
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            contract_type, parties, key_terms, financial, termination
        )
        
        # Calculate metrics
        word_count = len(cleaned_text.split())
        read_time = max(1, word_count // 200)  # ~200 words per minute
        
        summary = ContractSummary(
            executive_summary=executive_summary,
            key_terms=key_terms,
            parties_involved=parties,
            contract_type=contract_type,
            key_obligations=obligations,
            important_dates=dates,
            financial_terms=financial,
            termination_conditions=termination,
            governing_law=governing_law,
            word_count=word_count,
            estimated_read_time=read_time
        )
        
        self.logger.info(f"Contract summarization completed. {word_count} words analyzed.")
        return summary
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize contract text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers, headers, footers
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        return text.strip()
    
    def _segment_into_sentences(self, text: str) -> List[str]:
        """Segment text into sentences using available tools"""
        # Use simple fallback by default to avoid hanging on model loading
        return self._fallback_sentence_split(text)
    
    def _fallback_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting fallback"""
        # Split on periods, exclamation marks, question marks followed by whitespace or end of string
        sentences = re.split(r'[.!?]+\s+|\n{2,}', text)
        # Clean and filter sentences
        cleaned_sentences = []
        for s in sentences:
            s = s.strip()
            if s and len(s.split()) > 3:  # At least 3 words
                # Add period back if missing
                if not s[-1] in '.!?':
                    s += '.'
                cleaned_sentences.append(s)
        return cleaned_sentences
    
    def _extract_parties(self, text: str) -> List[str]:
        """Extract parties involved in the contract with better pattern matching"""
        parties = set()
        
        # Use more comprehensive, safer patterns
        simple_patterns = [
            r'(?i)between\s+([A-Z][A-Za-z\s]+(?:Inc|LLC|Corp|Corporation|Company))',
            r'(?i)between\s+([A-Z][A-Za-z\s]{3,30})\s+and\s+([A-Z][A-Za-z\s]{3,30})',
            r'(?i)"([A-Z][A-Za-z\s]{3,30})"\s*\(',
            r'(?i)([A-Z][A-Za-z\s]+(?:Inc|LLC|Corp|Corporation|Solutions|Systems|Services|Company))',
            r'(?i)this\s+agreement\s+is\s+between\s+([^,\n]+)\s+and\s+([^,\n]+)',
            r'(?i)company["\s]*([A-Z][A-Za-z\s]+)',
            r'(?i)provider["\s]*([A-Z][A-Za-z\s]+)',
        ]
        
        # Look in the first 1500 characters for better accuracy
        search_text = text[:1500]
        
        for pattern in simple_patterns:
            try:
                matches = re.findall(pattern, search_text)
                for match in matches:
                    if isinstance(match, tuple):
                        for m in match:
                            if m.strip() and len(m.strip()) > 3:
                                parties.add(m.strip().strip('"').strip("'"))
                    else:
                        if match.strip() and len(match.strip()) > 3:
                            parties.add(match.strip().strip('"').strip("'"))
            except:
                continue  # Skip problematic patterns
        
        # Clean and filter parties
        cleaned_parties = []
        excluded_words = ['the', 'user', 'you', 'we', 'us', 'company', 'agreement', 
                         'service', 'terms', 'privacy', 'policy', 'arising from use of our services']
        
        for party in parties:
            party_clean = party.strip().strip('"').strip("'")
            if (len(party_clean) > 3 and 
                party_clean.lower() not in excluded_words and
                not party_clean.lower().startswith(('the ', 'this ', 'these ', 'our ', 'your '))):
                cleaned_parties.append(party_clean)
        
        # If no specific parties found, provide generic ones
        if not cleaned_parties:
            cleaned_parties = ["Service Provider", "User/Customer"]
        
        return list(set(cleaned_parties))[:5]  # Limit to top 5
    
    def _identify_contract_type(self, text: str) -> str:
        """Identify the type of contract"""
        contract_types = {
            'software': ['software', 'saas', 'platform', 'application', 'api'],
            'service': ['service', 'terms of service', 'tos'],
            'employment': ['employment', 'employee', 'job', 'work', 'salary'],
            'lease': ['lease', 'rental', 'rent', 'property', 'premises'],
            'licensing': ['license', 'licensing', 'intellectual property', 'copyright'],
            'subscription': ['subscription', 'recurring', 'monthly', 'annual'],
            'privacy': ['privacy', 'data protection', 'personal information'],
            'general': ['agreement', 'contract']
        }
        
        text_lower = text.lower()
        scores = {}
        
        for contract_type, keywords in contract_types.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            if score > 0:
                scores[contract_type] = score
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0].title() + " Agreement"
        
        return "General Agreement"
    
    def _extract_key_terms(self, text: str) -> Dict[str, str]:
        """Extract key terms and definitions with better coverage"""
        key_terms = {}
        
        # Look for definitions with multiple patterns
        definition_patterns = [
            r'"([^"]{3,50})"\s*means\s+([^.]{10,200})',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\([^)]*\)\s*means\s+([^.]{10,200})',
            r'(?i)defined?\s+as\s*[:\-]?\s*([^.]{10,200})',
            r'(?i)([A-Z][A-Za-z\s]{3,30})\s*[:\-]\s*([^.\n]{10,200})',
            r'(?i)"([^"]{3,50})"\s*[:\-]\s*([^.\n]{10,200})',
        ]
        
        for pattern in definition_patterns:
            try:
                matches = re.findall(pattern, text[:3000])  # Search more text
                for match in matches:
                    if isinstance(match, tuple) and len(match) == 2:
                        term, definition = match
                        term = term.strip().strip('"').strip("'")
                        definition = definition.strip()
                        
                        # Filter out overly generic terms
                        if (len(term) > 2 and len(definition) > 5 and 
                            term.lower() not in ['this', 'the', 'you', 'we', 'us', 'service'] and
                            not definition.lower().startswith(('this', 'the', 'you', 'we'))):
                            key_terms[term] = definition[:100] + "..." if len(definition) > 100 else definition
            except:
                continue
        
        # If no formal definitions found, look for important terms
        if not key_terms:
            important_terms = {
                'Service': 'Software service or platform provided',
                'User': 'Individual or entity using the service', 
                'Agreement': 'This terms of service document',
                'Account': 'User account and associated data'
            }
            
            text_lower = text.lower()
            for term, default_def in important_terms.items():
                if term.lower() in text_lower:
                    key_terms[term] = default_def
                    if len(key_terms) >= 3:  # Limit to keep it clean
                        break
        
        return key_terms
    
    def _extract_obligations(self, text: str) -> Dict[str, List[str]]:
        """Extract key obligations for each party with improved detail"""
        obligations = {'User/Customer': [], 'Company/Provider': []}
        
        try:
            # Enhanced extraction with more specific patterns
            text_lower = text.lower()
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
            
            # More sophisticated user obligation patterns
            user_patterns = [
                ('payment', ['pay', 'payment', 'fee', 'subscription', 'billing']),
                ('compliance', ['comply', 'follow', 'abide by', 'adhere to', 'observe']),
                ('usage restrictions', ['not use', 'prohibited', 'restricted', 'forbidden']),
                ('data accuracy', ['accurate', 'complete', 'truthful', 'valid information']),
                ('account security', ['secure', 'protect', 'maintain', 'password', 'credentials'])
            ]
            
            # More sophisticated company obligation patterns  
            company_patterns = [
                ('service delivery', ['provide', 'deliver', 'supply', 'offer', 'make available']),
                ('support/maintenance', ['support', 'maintain', 'update', 'fix', 'resolve']),
                ('data protection', ['protect', 'secure', 'safeguard', 'encrypt', 'privacy']),
                ('uptime/availability', ['available', 'accessible', 'operational', 'uptime']),
                ('notification', ['notify', 'inform', 'communicate', 'alert', 'notice'])
            ]
            
            # Extract user obligations
            for sentence in sentences[:30]:  # Limit for performance
                sentence_lower = sentence.lower()
                
                # Check if sentence refers to user/customer
                if any(ref in sentence_lower for ref in ['user', 'customer', 'you', 'subscriber']):
                    for obligation_type, keywords in user_patterns:
                        if any(keyword in sentence_lower for keyword in keywords):
                            # Extract more specific obligation
                            obligation_text = sentence[:100] + "..." if len(sentence) > 100 else sentence
                            obligations['User/Customer'].append(f"{obligation_type.title()}: {obligation_text}")
                            break
                
                # Check if sentence refers to company/provider
                if any(ref in sentence_lower for ref in ['company', 'provider', 'we', 'service']):
                    for obligation_type, keywords in company_patterns:
                        if any(keyword in sentence_lower for keyword in keywords):
                            # Extract more specific obligation
                            obligation_text = sentence[:100] + "..." if len(sentence) > 100 else sentence
                            obligations['Company/Provider'].append(f"{obligation_type.title()}: {obligation_text}")
                            break
            
            # If no specific obligations found, add general ones
            if not obligations['User/Customer']:
                obligations['User/Customer'] = [
                    'Payment: Pay applicable fees as specified',
                    'Compliance: Follow all terms and conditions',
                    'Usage: Use service in accordance with agreement'
                ]
            
            if not obligations['Company/Provider']:
                obligations['Company/Provider'] = [
                    'Service Delivery: Provide agreed-upon services',
                    'Support: Maintain reasonable customer support',
                    'Compliance: Adhere to applicable laws and regulations'
                ]
        
        except:
            # Fallback obligations if extraction fails
            obligations = {
                'User/Customer': [
                    'Payment: Pay applicable fees as specified',
                    'Compliance: Follow all terms and conditions'
                ],
                'Company/Provider': [
                    'Service Delivery: Provide agreed-upon services',
                    'Support: Maintain customer support'
                ]
            }
        
        # Limit to 3 obligations per party for clean display
        obligations['User/Customer'] = obligations['User/Customer'][:3]
        obligations['Company/Provider'] = obligations['Company/Provider'][:3]
        
        return obligations
    
    def _extract_important_dates(self, text: str) -> List[Dict[str, str]]:
        """Extract important dates and deadlines"""
        dates = []
        
        for pattern in self.patterns['dates']:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    event, date = match
                    dates.append({'event': event.strip(), 'date': date.strip()})
        
        return dates[:10]  # Limit to 10 most relevant dates
    
    def _extract_financial_terms(self, text: str) -> Dict[str, str]:
        """Extract financial terms and amounts with better coverage"""
        financial = {}
        
        # Use comprehensive, safe patterns
        try:
            # Look for dollar amounts
            dollar_pattern = r'\$[\d,]+(?:\.\d{2})?'
            dollar_matches = re.findall(dollar_pattern, text[:3000])  # Search more text
            if dollar_matches:
                financial['amount'] = dollar_matches[0]
                if len(dollar_matches) > 1:
                    financial['additional_amounts'] = ', '.join(dollar_matches[1:3])
            
            # Look for payment frequency
            text_lower = text.lower()
            if 'monthly' in text_lower or 'per month' in text_lower:
                financial['frequency'] = 'Monthly'
            elif 'annual' in text_lower or 'yearly' in text_lower or 'per year' in text_lower:
                financial['frequency'] = 'Annually'
            elif 'weekly' in text_lower or 'per week' in text_lower:
                financial['frequency'] = 'Weekly'
            elif 'one-time' in text_lower or 'lump sum' in text_lower:
                financial['frequency'] = 'One-time'
                
            # Look for fee types
            if 'subscription' in text_lower:
                financial['type'] = 'Subscription Fee'
            elif 'license' in text_lower:
                financial['type'] = 'License Fee'
            elif 'service' in text_lower:
                financial['type'] = 'Service Fee'
            elif 'penalty' in text_lower or 'fine' in text_lower:
                financial['penalty_info'] = 'Penalty clauses present'
                
            # Look for payment terms
            payment_patterns = [
                r'(?i)payment\s+due\s+(?:within\s+)?(\d+)\s+(days?|weeks?|months?)',
                r'(?i)invoice\s+(?:within\s+)?(\d+)\s+(days?|weeks?)',
                r'(?i)net\s+(\d+)\s+days?',
            ]
            
            for pattern in payment_patterns:
                match = re.search(pattern, text[:2000])
                if match:
                    period = match.group(1)
                    unit = match.group(2) if len(match.groups()) > 1 else 'days'
                    financial['payment_terms'] = f"{period} {unit}"
                    break
                
        except:
            pass  # Ignore regex errors
        
        # If no financial info found, check for basic mentions
        if not financial:
            text_lower = text.lower()
            if any(term in text_lower for term in ['free', 'no cost', 'without charge']):
                financial['type'] = 'Free Service'
            elif any(term in text_lower for term in ['payment', 'fee', 'cost', 'price', 'charge']):
                financial['type'] = 'Payment Required'
        
        return financial
    
    def _extract_termination_conditions(self, text: str) -> List[str]:
        """Extract termination and cancellation conditions"""
        conditions = []
        
        for pattern in self.patterns['termination']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            conditions.extend(matches)
        
        return list(set(conditions))[:5]
    
    def _extract_governing_law(self, text: str) -> str:
        """Extract governing law and jurisdiction"""
        for pattern in self.patterns['governing_law']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Not specified"
    
    def _generate_executive_summary(self, contract_type: str, parties: List[str], 
                                  key_terms: Dict, financial: Dict, 
                                  termination: List[str]) -> str:
        """Generate executive summary of the contract"""
        
        summary_parts = []
        
        # Contract type and parties
        if parties:
            parties_str = " and ".join(parties[:2])
            summary_parts.append(f"This {contract_type.lower()} is between {parties_str}.")
        else:
            summary_parts.append(f"This is a {contract_type.lower()}.")
        
        # Financial terms
        if financial:
            financial_info = []
            for term, amount in list(financial.items())[:2]:
                financial_info.append(f"{term}: {amount}")
            if financial_info:
                summary_parts.append(f"Key financial terms include {', '.join(financial_info)}.")
        
        # Termination
        if termination:
            summary_parts.append(f"The agreement includes termination provisions regarding {', '.join(termination[:2])}.")
        
        # Key terms count
        if key_terms:
            summary_parts.append(f"The contract defines {len(key_terms)} key terms and concepts.")
        
        return " ".join(summary_parts)

    def generate_enhanced_contract_summary(self, contract_text: str, unfair_clauses: List[Any], risk_score: float) -> EnhancedContractSummary:
        """
        Generate enhanced analytical contract summary with risk prioritization and cross-references
        
        Args:
            contract_text: Full contract text
            unfair_clauses: List of detected unfair clauses
            risk_score: Overall risk score
            
        Returns:
            EnhancedContractSummary with comprehensive analysis
        """
        
        # Generate basic summary first
        basic_summary = self.summarize_contract(contract_text)
        
        # Extract structured sections
        sections = self._extract_contract_sections(contract_text)
        
        # Analyze clause relationships
        clause_relationships = self._analyze_clause_relationships(unfair_clauses)
        
        # Generate risk prioritization
        risk_analysis = self._generate_risk_prioritization(unfair_clauses, risk_score)
        
        # Create actionable insights
        actionable_insights = self._generate_actionable_insights(unfair_clauses, sections)
        
        # Generate negotiation strategy
        negotiation_strategy = self._generate_negotiation_strategy(unfair_clauses)
        
        # Identify compliance concerns
        compliance_concerns = self._identify_compliance_concerns(unfair_clauses)
        
        # Create executive summary
        executive_summary = self._create_executive_summary(risk_score, unfair_clauses, sections)
        
        return EnhancedContractSummary(
            executive_summary=executive_summary,
            sections_analysis=sections,
            risk_prioritization=risk_analysis,
            clause_relationships=clause_relationships,
            actionable_insights=actionable_insights,
            negotiation_strategy=negotiation_strategy,
            compliance_concerns=compliance_concerns,
            basic_summary=basic_summary
        )
    
    def _extract_contract_sections(self, contract_text: str) -> Dict[str, Dict]:
        """Extract and analyze key contract sections"""
        
        sections = {
            "service_terms": {"content": "", "risk_indicators": [], "key_points": []},
            "payment_terms": {"content": "", "risk_indicators": [], "key_points": []},
            "termination": {"content": "", "risk_indicators": [], "key_points": []},
            "liability": {"content": "", "risk_indicators": [], "key_points": []},
            "data_privacy": {"content": "", "risk_indicators": [], "key_points": []},
            "intellectual_property": {"content": "", "risk_indicators": [], "key_points": []},
            "dispute_resolution": {"content": "", "risk_indicators": [], "key_points": []}
        }
        
        # Section identification patterns
        section_patterns = {
            "service_terms": [r"service\s+terms", r"scope\s+of\s+services", r"software\s+license"],
            "payment_terms": [r"payment", r"billing", r"fees", r"subscription"],
            "termination": [r"termination", r"cancellation", r"end\s+of\s+agreement"],
            "liability": [r"liability", r"limitation", r"damages", r"warranties"],
            "data_privacy": [r"data", r"privacy", r"personal\s+information", r"confidential"],
            "intellectual_property": [r"intellectual\s+property", r"copyright", r"trademark", r"ownership"],
            "dispute_resolution": [r"dispute", r"arbitration", r"governing\s+law", r"jurisdiction"]
        }
        
        paragraphs = [p.strip() for p in contract_text.split('\n\n') if len(p.strip()) > 50]
        
        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()
            
            # Classify paragraph by section
            for section_name, patterns in section_patterns.items():
                if any(re.search(pattern, paragraph_lower) for pattern in patterns):
                    if not sections[section_name]["content"]:
                        sections[section_name]["content"] = paragraph[:500]
                    
                    # Identify risk indicators
                    risk_indicators = self._identify_section_risks(paragraph, section_name)
                    sections[section_name]["risk_indicators"].extend(risk_indicators)
                    
                    # Extract key points
                    key_points = self._extract_key_points(paragraph, section_name)
                    sections[section_name]["key_points"].extend(key_points)
                    break
        
        return sections
    
    def _identify_section_risks(self, paragraph: str, section_type: str) -> List[str]:
        """Identify specific risks within a contract section"""
        
        risk_patterns = {
            "service_terms": [
                ("unilateral changes", r"(?i)(?:may|can)\s+(?:modify|change|alter)\s+(?:at\s+any\s+time|without\s+notice)"),
                ("service limitations", r"(?i)(?:may|can)\s+(?:suspend|discontinue|limit)\s+(?:at\s+any\s+time|without\s+notice)"),
                ("broad discretion", r"(?i)(?:sole\s+discretion|absolute\s+discretion)")
            ],
            "payment_terms": [
                ("automatic renewal", r"(?i)automatically\s+(?:renew|extend|continue)"),
                ("penalty fees", r"(?i)(?:penalty|fine|charge).*\$[0-9,]+"),
                ("price changes", r"(?i)(?:increase|change|modify)\s+(?:fees|price|cost)")
            ],
            "termination": [
                ("immediate termination", r"(?i)terminate\s+(?:immediately|at\s+any\s+time|without\s+notice)"),
                ("no refund", r"(?i)no\s+refund"),
                ("data deletion", r"(?i)(?:delete|remove|destroy)\s+(?:all\s+)?data")
            ],
            "liability": [
                ("liability exclusion", r"(?i)(?:exclude|disclaim|limit)\s+(?:all\s+)?(?:liability|warranties)"),
                ("consequential damages", r"(?i)no\s+liability\s+for\s+(?:consequential|indirect|incidental)"),
                ("maximum liability", r"(?i)liability\s+(?:limited\s+to|shall\s+not\s+exceed)")
            ]
        }
        
        risks = []
        if section_type in risk_patterns:
            for risk_name, pattern in risk_patterns[section_type]:
                if re.search(pattern, paragraph):
                    risks.append(risk_name)
        
        return risks
    
    def _extract_key_points(self, paragraph: str, section_type: str) -> List[str]:
        """Extract key contractual points from a section"""
        
        # Split into sentences and find important ones
        sentences = [s.strip() for s in paragraph.split('.') if len(s.strip()) > 20]
        
        key_indicators = {
            "service_terms": ["license", "access", "use", "restrictions", "permissions"],
            "payment_terms": ["due", "billing", "subscription", "refund", "cancellation"],
            "termination": ["terminate", "end", "cancel", "notice", "effect"],
            "liability": ["liable", "responsible", "damages", "limit", "exclude"],
            "data_privacy": ["collect", "use", "share", "process", "store"],
            "intellectual_property": ["own", "license", "copyright", "trademark", "rights"]
        }
        
        key_points = []
        indicators = key_indicators.get(section_type, [])
        
        for sentence in sentences[:3]:  # Take top 3 sentences
            if any(indicator in sentence.lower() for indicator in indicators):
                key_points.append(sentence[:150] + "..." if len(sentence) > 150 else sentence)
        
        return key_points
    
    def _analyze_clause_relationships(self, unfair_clauses: List[Any]) -> Dict[str, List[str]]:
        """Analyze relationships and interactions between unfair clauses"""
        
        relationships = {
            "reinforcing_clauses": [],
            "conflicting_clauses": [],
            "escalating_risks": []
        }
        
        clause_types = []
        for clause in unfair_clauses:
            if hasattr(clause, 'clause_type'):
                clause_types.append(clause.clause_type)
            elif isinstance(clause, dict):
                clause_types.append(clause.get('clause_type', 'unknown'))
        
        # Define clause relationships
        reinforcing_combinations = [
            (["unilateral_termination", "broad_indemnification"], "Termination + indemnification creates one-sided risk"),
            (["limitation_of_liability", "mandatory_arbitration"], "Limited liability + forced arbitration reduces recourse"),
            (["automatic_renewal", "unreasonable_penalties"], "Auto-renewal + penalties trap customers"),
            (["unilateral_modification", "broad_indemnification"], "Modification rights + indemnification shift all risk")
        ]
        
        for combination, description in reinforcing_combinations:
            if all(clause_type in clause_types for clause_type in combination):
                relationships["reinforcing_clauses"].append(description)
        
        # Identify escalating risks
        if len(unfair_clauses) >= 5:
            relationships["escalating_risks"].append("Multiple unfair clauses create compound risk exposure")
        
        high_severity_count = 0
        for clause in unfair_clauses:
            if hasattr(clause, 'severity') and clause.severity == "high":
                high_severity_count += 1
            elif isinstance(clause, dict) and clause.get('severity') == "high":
                high_severity_count += 1
        
        if high_severity_count >= 3:
            relationships["escalating_risks"].append("Concentration of high-severity clauses indicates systematic unfairness")
        
        return relationships
    
    def _generate_risk_prioritization(self, unfair_clauses: List[Any], risk_score: float) -> Dict[str, Any]:
        """Generate prioritized risk assessment"""
        
        # Group clauses by severity and impact
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for clause in unfair_clauses:
            severity = "medium"  # default
            if hasattr(clause, 'severity'):
                severity = clause.severity
            elif isinstance(clause, dict):
                severity = clause.get('severity', 'medium')
            
            if severity == "high":
                high_priority.append(clause)
            elif severity == "medium":
                medium_priority.append(clause)
            else:
                low_priority.append(clause)
        
        # Calculate business impact
        business_impact = {
            "financial_risk": 0,
            "operational_risk": 0,
            "legal_risk": 0,
            "reputational_risk": 0
        }
        
        impact_mapping = {
            "unilateral_termination": {"operational_risk": 0.8, "financial_risk": 0.6},
            "broad_indemnification": {"legal_risk": 0.9, "financial_risk": 0.8},
            "limitation_of_liability": {"legal_risk": 0.7, "financial_risk": 0.6},
            "automatic_renewal": {"financial_risk": 0.5, "operational_risk": 0.3},
            "unreasonable_penalties": {"financial_risk": 0.7},
            "mandatory_arbitration": {"legal_risk": 0.6}
        }
        
        for clause in unfair_clauses:
            clause_type = ""
            if hasattr(clause, 'clause_type'):
                clause_type = clause.clause_type
            elif isinstance(clause, dict):
                clause_type = clause.get('clause_type', '')
            
            if clause_type in impact_mapping:
                for risk_type, impact in impact_mapping[clause_type].items():
                    business_impact[risk_type] += impact
        
        # Normalize scores
        max_impact = max(business_impact.values()) if business_impact.values() else 1
        if max_impact > 0:
            business_impact = {k: v/max_impact for k, v in business_impact.items()}
        
        return {
            "overall_risk_level": "High" if risk_score > 0.7 else "Medium" if risk_score > 0.3 else "Low",
            "risk_score": risk_score,
            "high_priority_clauses": len(high_priority),
            "medium_priority_clauses": len(medium_priority),
            "low_priority_clauses": len(low_priority),
            "business_impact": business_impact,
            "immediate_action_required": len(high_priority) > 0 or risk_score > 0.7
        }
    
    def _generate_actionable_insights(self, unfair_clauses: List[Any], sections: Dict) -> List[Dict[str, str]]:
        """Generate specific, actionable insights for contract negotiation"""
        
        insights = []
        
        # Section-specific insights
        for section_name, section_data in sections.items():
            if section_data["risk_indicators"]:
                insight = {
                    "section": section_name.replace('_', ' ').title(),
                    "risk_level": "High" if len(section_data["risk_indicators"]) >= 2 else "Medium",
                    "specific_concern": f"Found {len(section_data['risk_indicators'])} risk indicators: {', '.join(section_data['risk_indicators'])}",
                    "recommended_action": self._get_section_recommendation(section_name, section_data["risk_indicators"])
                }
                insights.append(insight)
        
        # Clause-specific insights
        clause_groups = {}
        for clause in unfair_clauses:
            clause_type = ""
            if hasattr(clause, 'clause_type'):
                clause_type = clause.clause_type
            elif isinstance(clause, dict):
                clause_type = clause.get('clause_type', 'unknown')
            
            if clause_type not in clause_groups:
                clause_groups[clause_type] = []
            clause_groups[clause_type].append(clause)
        
        for clause_type, clauses in clause_groups.items():
            if len(clauses) > 1:
                insight = {
                    "section": "Multiple Instances",
                    "risk_level": "High",
                    "specific_concern": f"Found {len(clauses)} instances of {clause_type.replace('_', ' ')} clauses",
                    "recommended_action": f"Address all {len(clauses)} instances collectively in negotiation"
                }
                insights.append(insight)
        
        return insights
    
    def _get_section_recommendation(self, section_name: str, risk_indicators: List[str]) -> str:
        """Get specific recommendation for a contract section"""
        
        recommendations = {
            "service_terms": "Negotiate mutual modification rights and service level guarantees",
            "payment_terms": "Request transparent pricing and reasonable cancellation terms",
            "termination": "Negotiate mutual termination rights with reasonable notice periods",
            "liability": "Seek balanced liability allocation with appropriate exclusions",
            "data_privacy": "Ensure compliance with data protection regulations and user rights",
            "intellectual_property": "Clarify ownership rights and usage permissions",
            "dispute_resolution": "Consider implications of mandatory arbitration clauses"
        }
        
        return recommendations.get(section_name, "Conduct detailed legal review of this section")
    
    def _generate_negotiation_strategy(self, unfair_clauses: List[Any]) -> Dict[str, Any]:
        """Generate strategic approach for contract negotiation"""
        
        strategy = {
            "primary_objectives": [],
            "negotiation_priorities": [],
            "fallback_positions": [],
            "deal_breakers": []
        }
        
        high_severity_clauses = []
        for clause in unfair_clauses:
            severity = "medium"
            if hasattr(clause, 'severity'):
                severity = clause.severity
            elif isinstance(clause, dict):
                severity = clause.get('severity', 'medium')
            
            if severity == "high":
                high_severity_clauses.append(clause)
        
        # Primary objectives
        if high_severity_clauses:
            strategy["primary_objectives"].append("Remove or significantly modify high-severity unfair clauses")
        
        if len(unfair_clauses) >= 5:
            strategy["primary_objectives"].append("Comprehensive clause review and rebalancing")
        
        # Negotiation priorities (ranked)
        clause_priority = {
            "broad_indemnification": 1,
            "unilateral_termination": 2,
            "limitation_of_liability": 3,
            "mandatory_arbitration": 4,
            "automatic_renewal": 5,
            "unreasonable_penalties": 6
        }
        
        priorities = []
        for clause in unfair_clauses:
            clause_type = ""
            if hasattr(clause, 'clause_type'):
                clause_type = clause.clause_type
            elif isinstance(clause, dict):
                clause_type = clause.get('clause_type', 'unknown')
            
            priority_level = clause_priority.get(clause_type, 10)
            priorities.append((clause_type, priority_level))
        
        priorities = sorted(priorities, key=lambda x: x[1])
        
        strategy["negotiation_priorities"] = [
            f"{i+1}. Address {clause_type.replace('_', ' ')} provisions"
            for i, (clause_type, _) in enumerate(priorities[:5])
        ]
        
        # Deal breakers
        for clause in unfair_clauses:
            clause_type = ""
            severity = "medium"
            if hasattr(clause, 'clause_type'):
                clause_type = clause.clause_type
            if hasattr(clause, 'severity'):
                severity = clause.severity
            elif isinstance(clause, dict):
                clause_type = clause.get('clause_type', '')
                severity = clause.get('severity', 'medium')
            
            if clause_type == "broad_indemnification" and severity == "high":
                strategy["deal_breakers"].append("Unlimited indemnification liability")
                break
        
        if len(high_severity_clauses) >= 3:
            strategy["deal_breakers"].append("Concentration of high-severity unfair terms")
        
        return strategy
    
    def _identify_compliance_concerns(self, unfair_clauses: List[Any]) -> List[Dict[str, str]]:
        """Identify potential regulatory compliance concerns"""
        
        concerns = []
        
        compliance_mapping = {
            "automatic_renewal": {
                "regulation": "Consumer Protection Laws",
                "concern": "Auto-renewal clauses may violate state consumer protection requirements",
                "jurisdiction": "State-level (varies by state)"
            },
            "broad_indemnification": {
                "regulation": "Unconscionability Doctrine",
                "concern": "Overly broad indemnification may be deemed unconscionable",
                "jurisdiction": "Federal and State Courts"
            },
            "mandatory_arbitration": {
                "regulation": "Consumer Arbitration Rules",
                "concern": "Arbitration clauses in consumer contracts face increasing scrutiny",
                "jurisdiction": "Federal (CFPB) and State"
            },
            "limitation_of_liability": {
                "regulation": "UCC and Consumer Protection",
                "concern": "Complete liability limitations may violate consumer protection standards",
                "jurisdiction": "State Commercial Law"
            }
        }
        
        for clause in unfair_clauses:
            clause_type = ""
            severity = "medium"
            if hasattr(clause, 'clause_type'):
                clause_type = clause.clause_type
            if hasattr(clause, 'severity'):
                severity = clause.severity
            elif isinstance(clause, dict):
                clause_type = clause.get('clause_type', '')
                severity = clause.get('severity', 'medium')
            
            if clause_type in compliance_mapping:
                concern_info = compliance_mapping[clause_type]
                concerns.append({
                    "clause_type": clause_type.replace('_', ' ').title(),
                    "regulation": concern_info["regulation"],
                    "concern": concern_info["concern"],
                    "jurisdiction": concern_info["jurisdiction"],
                    "severity": severity
                })
        
        return concerns
    
    def _create_executive_summary(self, risk_score: float, unfair_clauses: List[Any], sections: Dict) -> str:
        """Create executive summary of contract analysis"""
        
        risk_level = "High" if risk_score > 0.7 else "Medium" if risk_score > 0.3 else "Low"
        
        high_risk_clauses = 0
        for clause in unfair_clauses:
            severity = "medium"
            if hasattr(clause, 'severity'):
                severity = clause.severity
            elif isinstance(clause, dict):
                severity = clause.get('severity', 'medium')
            
            if severity == "high":
                high_risk_clauses += 1
        
        # Identify most problematic sections
        problematic_sections = [
            name.replace('_', ' ').title() 
            for name, data in sections.items() 
            if len(data.get("risk_indicators", [])) >= 2
        ]
        
        summary = f"""
EXECUTIVE SUMMARY

Overall Risk Assessment: {risk_level} Risk (Score: {risk_score:.1%})

Key Findings:
• {len(unfair_clauses)} potentially unfair clauses identified
• {high_risk_clauses} high-severity clauses requiring immediate attention
• {len(problematic_sections)} contract sections with significant risk indicators

Most Concerning Areas: {', '.join(problematic_sections[:3]) if problematic_sections else 'No major section-level risks identified'}

Recommendation: {'NEGOTIATE BEFORE SIGNING' if risk_score > 0.7 or high_risk_clauses > 0 else 'PROCEED WITH CAUTION' if risk_score > 0.3 else 'ACCEPTABLE RISK LEVEL'}
        """
        
        return summary.strip()
