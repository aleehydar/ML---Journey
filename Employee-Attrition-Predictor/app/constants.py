from dataclasses import dataclass

@dataclass
class RiskThresholds:
    """Risk probability thresholds for attrition categorization."""
    LOW_THRESHOLD: float = 0.30  # Below 30% = low risk
    MEDIUM_THRESHOLD: float = 0.60  # Below 60% = medium risk
    HIGH_THRESHOLD: float = 1.00  # Above 60% = high risk
    
    def categorize(self, probability: float) -> str:
        """Categorize risk level based on probability.
        
        Args:
            probability: Attrition probability (0-1)
            
        Returns:
            Risk level: "Low", "Medium", or "High"
        """
        if probability < self.LOW_THRESHOLD:
            return "Low"
        elif probability < self.MEDIUM_THRESHOLD:
            return "Medium"
        else:
            return "High"

risk_thresholds = RiskThresholds()
