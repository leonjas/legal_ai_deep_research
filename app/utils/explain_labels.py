"""
Label normalization and explanation mapping for unfair clause detection
"""

def normalize_label(label: str) -> str:
    """
    Normalize model output labels to canonical form for explanation mapping
    
    Args:
        label: Raw label from the model
        
    Returns:
        Normalized label that matches explanation keys
    """
    l = label.strip().lower().replace(" ", "_").replace("-", "_")
    
    # Common variants -> canonical keys mapping
    remap = {
        # Liability and warranty limitations
        "limitation_of_liability": "limitation_of_liability",
        "liability_limitation": "limitation_of_liability",
        "warranty_disclaimer": "limitation_of_liability",
        "warranty_limitation": "limitation_of_liability",
        
        # Termination clauses
        "unilateral_termination": "unilateral_termination",
        "termination_clause": "unilateral_termination",
        "termination": "unilateral_termination",
        
        # Content and user rights
        "content_removal": "content_removal",
        "content_modification": "content_removal",
        "user_content": "content_removal",
        
        # Auto-renewal and subscription
        "auto_renewal": "automatic_renewal",
        "automatic_renewal": "automatic_renewal",
        "renewal_clause": "automatic_renewal",
        
        # Arbitration and legal process
        "arbitration": "mandatory_arbitration", 
        "mandatory_arbitration": "mandatory_arbitration",
        "dispute_resolution": "mandatory_arbitration",
        "jury_waiver": "mandatory_arbitration",
        
        # Jurisdiction and governing law
        "choice_of_law": "choice_of_law",
        "governing_law": "choice_of_law",
        "jurisdiction": "choice_of_law",
        
        # Indemnification
        "indemnification": "broad_indemnification",
        "broad_indemnification": "broad_indemnification",
        "indemnity": "broad_indemnification",
        
        # Generic unfair (for binary models)
        "unfair": "unfair",
        "potentially_unfair": "unfair",
        "clearly_unfair": "unfair",
        "not_fair": "unfair",
        
        # Fair clauses (should be filtered out)
        "fair": "fair",
        "clearly_fair": "fair",
        "acceptable": "fair",
        
        # Penalty and fee clauses
        "penalty": "unreasonable_penalties",
        "fee": "unreasonable_penalties",
        "liquidated_damages": "unreasonable_penalties",
    }
    
    return remap.get(l, l)


def get_explanation_for_label(label: str) -> dict:
    """
    Get detailed explanation for a normalized label
    
    Args:
        label: Normalized label from normalize_label()
        
    Returns:
        Dictionary with explanation, severity, and recommendations
    """
    
    explanations = {
        "limitation_of_liability": {
            "explanation": "This clause unfairly limits the company's liability or disclaims warranties, potentially leaving you without recourse if the service causes harm.",
            "severity": "high",
            "category": "Liability Limitation",
            "recommendation": "Seek balanced liability limitations that protect both parties fairly."
        },
        
        "unilateral_termination": {
            "explanation": "This clause allows the company to terminate the agreement unilaterally without justification, creating an unfair power imbalance.",
            "severity": "high", 
            "category": "Termination Rights",
            "recommendation": "Negotiate mutual termination clauses with reasonable notice periods."
        },
        
        "content_removal": {
            "explanation": "This clause gives the company broad rights to remove or modify your content without adequate justification or appeal process.",
            "severity": "medium",
            "category": "Content Rights", 
            "recommendation": "Request clear content policies and fair appeal procedures."
        },
        
        "automatic_renewal": {
            "explanation": "This clause automatically renews your subscription without explicit consent, potentially trapping you in unwanted commitments.",
            "severity": "medium",
            "category": "Auto-Renewal",
            "recommendation": "Ensure explicit consent requirements for contract renewals."
        },
        
        "mandatory_arbitration": {
            "explanation": "This clause requires mandatory arbitration and may waive important legal rights like jury trials or class action participation.",
            "severity": "medium",
            "category": "Dispute Resolution",
            "recommendation": "Consider the implications of waiving jury trial and class action rights."
        },
        
        "choice_of_law": {
            "explanation": "This clause may require legal disputes to be resolved in an inconvenient jurisdiction or under unfavorable laws.",
            "severity": "low",
            "category": "Governing Law",
            "recommendation": "Verify that the chosen jurisdiction and laws are reasonable for your situation."
        },
        
        "broad_indemnification": {
            "explanation": "This clause requires you to indemnify the company for an unreasonably broad range of claims, potentially including their own negligence.",
            "severity": "high",
            "category": "Indemnification",
            "recommendation": "Negotiate limited indemnification with specific exclusions for company negligence."
        },
        
        "unreasonable_penalties": {
            "explanation": "This clause contains excessive penalties, liquidated damages, or fees that are disproportionate to actual harm.",
            "severity": "high",
            "category": "Penalties & Fees",
            "recommendation": "Review penalty clauses for reasonableness and proportionality to actual damages."
        },
        
        # Generic unfair (for binary models)
        "unfair": {
            "explanation": "This clause has been identified as potentially unfair to consumers based on legal analysis.",
            "severity": "medium",
            "category": "General Unfairness",
            "recommendation": "Review this clause carefully and consider seeking legal advice if needed."
        }
    }
    
    return explanations.get(label, {
        "explanation": f"This clause ({label}) requires review for potential unfairness.",
        "severity": "medium", 
        "category": "Other",
        "recommendation": "Consider having this clause reviewed by a legal professional."
    })


def get_severity_color(severity: str) -> str:
    """Get color code for severity level"""
    colors = {
        "high": "🔴",
        "medium": "🟡", 
        "low": "🟢"
    }
    return colors.get(severity, "⚪")


def format_unfair_result(result: dict) -> dict:
    """
    Format a model prediction result with explanations
    
    Args:
        result: Dictionary with 'clause', 'label', 'confidence' from model
        
    Returns:
        Enhanced result with explanations and formatting
    """
    normalized_label = normalize_label(result["label"])
    explanation = get_explanation_for_label(normalized_label)
    
    return {
        **result,
        "normalized_label": normalized_label,
        "explanation": explanation["explanation"],
        "severity": explanation["severity"],
        "category": explanation["category"],
        "recommendation": explanation["recommendation"],
        "severity_emoji": get_severity_color(explanation["severity"])
    }
