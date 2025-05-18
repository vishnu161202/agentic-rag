from typing import Dict, List

class HallucinationChecker:
    @staticmethod
    def should_abstain(verification_result: Dict, threshold: float = 0.7) -> bool:
        """Determine if the system should abstain from answering"""
        if not verification_result["is_consistent"]:
            return True
        
        if verification_result["confidence"] < threshold:
            return True
            
        if len(verification_result["unsupported_claims"]) > 0:
            return True
            
        return False
    
    @staticmethod
    def generate_warning(verification_result: Dict) -> str:
        """Generate warning message about potential hallucinations"""
        warnings = []
        
        if not verification_result["is_consistent"]:
            warnings.append("The response contains information inconsistent with the source document.")
        
        for claim in verification_result["unsupported_claims"]:
            if claim.lower() != "none":
                warnings.append(f"Unsupported claim detected: {claim}")
        
        for visual_ref in verification_result["visual_content_references"]:
            warnings.append(f"Potential unsubstantiated reference to visual content: {visual_ref}")
        
        if warnings:
            return "WARNING: " + " ".join(warnings)
        return ""