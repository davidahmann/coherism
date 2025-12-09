"""
Consensus Engine
================

Multi-agent decision system that arbitrates between:
1. Semantic Agent: BEM-based risk/success/coverage signals
2. Heuristic Agent: Deterministic rule checks

Outputs one of four actions:
- TRUST: Use backbone output directly
- ABSTAIN: Decline with explanation
- ESCALATE: Route to human review
- QUERY: Request specific information to reduce uncertainty

The Query action is the key extension that enables active learning:
instead of passively abstaining, the system identifies what information
would help and requests it.

Author: David Ahmann
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from constants import DEFAULT_RISK_THRESHOLD, DEFAULT_CONFIDENCE_THRESHOLD


class Action(Enum):
    """Possible Consensus Engine outputs."""
    TRUST = "trust"
    ABSTAIN = "abstain"
    ESCALATE = "escalate"
    QUERY = "query"


class QueryType(Enum):
    """Types of queries the system can generate."""
    CLARIFICATION = "clarification"      # Input is ambiguous
    PRECEDENT = "precedent"              # No similar past cases
    DOMAIN_KNOWLEDGE = "domain_knowledge"  # Outside known domains
    VALIDATION = "validation"            # Low confidence, need check


@dataclass
class HeuristicResult:
    """Result from heuristic rule checking."""
    violations: List[str]        # List of violated rules
    is_critical: bool           # Whether any violation is critical (hard veto)
    details: Dict[str, Any]     # Additional context


@dataclass
class ConsensusDecision:
    """Full decision output from Consensus Engine."""
    action: Action
    confidence: float                    # How confident in this decision
    risk_score: float                    # From BEM
    success_score: float                 # From BEM
    coverage_score: float                # From BEM (OOD indicator)
    heuristic_result: HeuristicResult
    explanation: str                     # Human-readable explanation
    query_type: Optional[QueryType] = None  # If action is QUERY
    query_content: Optional[str] = None     # Specific query to ask
    query_id: Optional[str] = None          # Unique ID for tracking response


class SemanticAgent:
    """
    Uses BEM signals to provide risk-based intuition.
    
    This agent says: "Based on past experiences, this looks risky/safe/unfamiliar."
    """
    
    def __init__(
        self,
        risk_threshold: float = DEFAULT_RISK_THRESHOLD,
        success_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        coverage_threshold: float = 0.3
    ):
        self.risk_threshold = risk_threshold
        self.success_threshold = success_threshold
        self.coverage_threshold = coverage_threshold
    
    def evaluate(
        self,
        risk: float,
        success: float,
        coverage: float
    ) -> Tuple[str, float, str]:
        """
        Evaluate BEM signals and return recommendation.
        
        Returns:
            (recommendation, confidence, explanation)
        """
        # High risk -> suggest abstain/escalate
        if risk > self.risk_threshold:
            severity = "critical" if risk > 0.85 else "high"
            return (
                "abstain",
                risk,
                f"Similar to past failures ({severity} risk: {risk:.2f})"
            )
        
        # Low coverage -> suggest query
        if coverage < self.coverage_threshold:
            return (
                "query",
                1.0 - coverage,
                f"Outside experience distribution (coverage: {coverage:.2f})"
            )
        
        # High success evidence -> suggest trust
        if success > self.success_threshold:
            return (
                "trust",
                success,
                f"Similar to past successes (confidence: {success:.2f})"
            )
        
        # Moderate signals -> uncertain
        return (
            "uncertain",
            0.5,
            f"Mixed signals (risk: {risk:.2f}, success: {success:.2f}, coverage: {coverage:.2f})"
        )


class HeuristicAgent:
    """
    Executes deterministic rules that must be satisfied.
    
    This agent says: "These hard constraints are violated/satisfied."
    """
    
    def __init__(self):
        self.rules: List[Dict[str, Any]] = []
    
    def add_rule(
        self,
        name: str,
        check_fn,  # Callable[[Dict], bool] - returns True if violated
        is_critical: bool = False,
        description: str = ""
    ):
        """Add a heuristic rule."""
        self.rules.append({
            "name": name,
            "check": check_fn,
            "is_critical": is_critical,
            "description": description
        })
    
    def evaluate(self, context: Dict[str, Any]) -> HeuristicResult:
        """
        Check all rules against the context.
        
        Returns HeuristicResult with violations and criticality.
        """
        violations = []
        is_critical = False
        details = {}
        
        for rule in self.rules:
            try:
                violated = rule["check"](context)
                if violated:
                    violations.append(rule["name"])
                    details[rule["name"]] = rule["description"]
                    if rule["is_critical"]:
                        is_critical = True
            except Exception as e:
                # Rule check failed — log but don't block
                details[f"{rule['name']}_error"] = str(e)
        
        return HeuristicResult(
            violations=violations,
            is_critical=is_critical,
            details=details
        )


class ConsensusEngine:
    """
    Arbitrates between Semantic and Heuristic agents to produce final decision.
    
    Decision logic:
    1. If heuristic finds CRITICAL violation → ESCALATE (hard veto)
    2. If coverage is very low → QUERY (active learning)
    3. If risk is high → ABSTAIN or ESCALATE
    4. If success is high and no violations → TRUST
    5. Otherwise → QUERY or ABSTAIN based on uncertainty type
    """
    
    def __init__(
        self,
        semantic_weight: float = 0.6,
        heuristic_weight: float = 0.4,
        confidence_boost_threshold: float = 0.9
    ):
        self.semantic_agent = SemanticAgent()
        self.heuristic_agent = HeuristicAgent()
        self.semantic_weight = semantic_weight
        self.heuristic_weight = heuristic_weight
        self.confidence_boost_threshold = confidence_boost_threshold
    
    def add_heuristic_rule(self, *args, **kwargs):
        """Delegate to heuristic agent."""
        self.heuristic_agent.add_rule(*args, **kwargs)
    
    def decide(
        self,
        risk: float,
        success: float,
        coverage: float,
        context: Dict[str, Any]
    ) -> ConsensusDecision:
        """
        Make a decision based on all signals.
        
        Args:
            risk: BEM risk signal
            success: BEM success signal
            coverage: BEM coverage signal
            context: Context dict for heuristic rules
        
        Returns:
            ConsensusDecision with action and explanation
        """
        # Get agent recommendations
        semantic_rec, semantic_conf, semantic_expl = self.semantic_agent.evaluate(
            risk, success, coverage
        )
        heuristic_result = self.heuristic_agent.evaluate(context)
        
        # Decision logic
        
        # 1. Critical heuristic violation → ESCALATE (hard veto)
        if heuristic_result.is_critical:
            return ConsensusDecision(
                action=Action.ESCALATE,
                confidence=1.0,
                risk_score=risk,
                success_score=success,
                coverage_score=coverage,
                heuristic_result=heuristic_result,
                explanation=f"Critical rule violation: {heuristic_result.violations}"
            )
        
        # 2. Very low coverage → QUERY (outside experience)
        if coverage < 0.2:
            query_type = self._diagnose_query_type(risk, success, coverage, heuristic_result)
            return ConsensusDecision(
                action=Action.QUERY,
                confidence=1.0 - coverage,
                risk_score=risk,
                success_score=success,
                coverage_score=coverage,
                heuristic_result=heuristic_result,
                explanation=f"Outside experience distribution (coverage: {coverage:.2f})",
                query_type=query_type,
                query_content=self._generate_query(query_type, context)
            )
        
        # 3. High risk → ABSTAIN or ESCALATE
        if risk > 0.6:
            action = Action.ESCALATE if risk > 0.85 else Action.ABSTAIN
            return ConsensusDecision(
                action=action,
                confidence=risk,
                risk_score=risk,
                success_score=success,
                coverage_score=coverage,
                heuristic_result=heuristic_result,
                explanation=semantic_expl
            )
        
        # 4. High success, no violations → TRUST
        if success > 0.7 and not heuristic_result.violations:
            confidence = success
            # Boost confidence if semantic agent is very confident
            if semantic_conf > self.confidence_boost_threshold:
                confidence = min(1.0, confidence * 1.2)
            
            return ConsensusDecision(
                action=Action.TRUST,
                confidence=confidence,
                risk_score=risk,
                success_score=success,
                coverage_score=coverage,
                heuristic_result=heuristic_result,
                explanation=semantic_expl
            )
        
        # 5. Non-critical violations → ABSTAIN with explanation
        if heuristic_result.violations:
            return ConsensusDecision(
                action=Action.ABSTAIN,
                confidence=0.7,
                risk_score=risk,
                success_score=success,
                coverage_score=coverage,
                heuristic_result=heuristic_result,
                explanation=f"Rule violations: {heuristic_result.violations}"
            )
        
        # 6. Uncertain → QUERY or TRUST with low confidence
        if semantic_rec == "uncertain":
            if coverage < 0.5:
                query_type = QueryType.VALIDATION
                return ConsensusDecision(
                    action=Action.QUERY,
                    confidence=0.5,
                    risk_score=risk,
                    success_score=success,
                    coverage_score=coverage,
                    heuristic_result=heuristic_result,
                    explanation="Uncertain - requesting validation",
                    query_type=query_type,
                    query_content=self._generate_query(query_type, context)
                )
            else:
                return ConsensusDecision(
                    action=Action.TRUST,
                    confidence=0.5,
                    risk_score=risk,
                    success_score=success,
                    coverage_score=coverage,
                    heuristic_result=heuristic_result,
                    explanation="Low confidence trust (proceed with caution)"
                )
        
        # Default: trust with moderate confidence
        return ConsensusDecision(
            action=Action.TRUST,
            confidence=0.6,
            risk_score=risk,
            success_score=success,
            coverage_score=coverage,
            heuristic_result=heuristic_result,
            explanation="Default trust (no strong signals)"
        )
    
    def _diagnose_query_type(
        self,
        risk: float,
        success: float,
        coverage: float,
        heuristic_result: HeuristicResult
    ) -> QueryType:
        """Determine what type of query would help."""
        if risk > 0.5 and success < 0.3:
            return QueryType.PRECEDENT  # Know failures but no successes
        if coverage < 0.1:
            return QueryType.DOMAIN_KNOWLEDGE  # Completely novel
        if heuristic_result.violations:
            return QueryType.CLARIFICATION  # Rule issues
        return QueryType.VALIDATION  # General uncertainty
    
    def _generate_query(
        self,
        query_type: QueryType,
        context: Dict[str, Any]
    ) -> str:
        """Generate a specific query based on type."""
        context_preview = str(context)[:100] if context else "N/A"
        
        templates = {
            QueryType.CLARIFICATION: 
                f"Input is ambiguous. Please clarify: {context_preview}...",
            QueryType.PRECEDENT: 
                f"No successful precedent found. How should this be handled? Context: {context_preview}...",
            QueryType.DOMAIN_KNOWLEDGE: 
                f"Outside known domains. What applies here? Context: {context_preview}...",
            QueryType.VALIDATION: 
                f"Low confidence. Please verify this approach: {context_preview}..."
        }
        
        return templates.get(query_type, "Please provide additional guidance.")
