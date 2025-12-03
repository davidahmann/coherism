"""
Healthcare Claim Simulator
==========================

Simulates a stream of medical claims submitted to a Payer with hidden rejection rules.
ALFM-BEM acts as the "Pre-Submission Scrubber", learning to predict rejections
based on past outcomes.

Rules (Hidden from Agent):
1. Medical Necessity: CPT 99213 (Office Visit) requires Diagnosis J01.90 (Strep) or J20.9 (Bronchitis).
2. Age Limit: CPT 90658 (Flu Shot) only allowed for age > 3.
3. Modifier Required: CPT 25111 (Removal of Ganglion Cyst) requires modifier RT or LT.

Author: David Ahmann
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bem import BidirectionalExperienceMemory, CoverageMode

# =============================================================================
# 1. Domain Entities
# =============================================================================

@dataclass
class Claim:
    id: str
    provider_id: str
    cpt_code: str
    diagnosis_code: str
    patient_age: int
    modifier: str = ""

    def __str__(self):
        return f"Claim({self.cpt_code}, {self.diagnosis_code}, Age={self.patient_age}, Mod={self.modifier})"

# =============================================================================
# 2. Synthetic Data Generator
# =============================================================================

class ClaimGenerator:
    def __init__(self):
        self.cpt_codes = ["99213", "90658", "25111", "99203", "71045"]
        self.diagnosis_codes = ["J01.90", "J20.9", "R05", "M67.4", "Z23"]
        self.modifiers = ["", "RT", "LT", "25", "59"]
        
    def generate(self, claim_id: int) -> Claim:
        # Generate random base attributes
        cpt = random.choice(self.cpt_codes)
        
        # Correlate diagnosis slightly to make it realistic (but still noisy)
        if cpt == "99213":
            dx = random.choice(["J01.90", "J20.9", "R05"]) # Mostly respiratory
        elif cpt == "25111":
            dx = "M67.4" # Ganglion
        elif cpt == "90658":
            dx = "Z23" # Vaccination
        else:
            dx = random.choice(self.diagnosis_codes)
            
        age = random.randint(1, 80)
        mod = random.choice(self.modifiers) if random.random() > 0.5 else ""
        
        return Claim(
            id=str(claim_id),
            provider_id=f"PROV_{random.randint(1, 5)}",
            cpt_code=cpt,
            diagnosis_code=dx,
            patient_age=age,
            modifier=mod
        )

# =============================================================================
# 3. Payer Rules Engine (Ground Truth)
# =============================================================================

class PayerEngine:
    """The 'World' that provides feedback."""
    
    def adjudicate(self, claim: Claim) -> Tuple[float, str]:
        """Returns (outcome, reason). Outcome: +1.0 (Paid), -1.0 (Rejected)."""
        
        # Rule 1: Medical Necessity
        if claim.cpt_code == "99213":
            if claim.diagnosis_code not in ["J01.90", "J20.9"]:
                return -1.0, "DENIAL: Medical Necessity (Dx mismatch)"
        
        # Rule 2: Age Limit
        if claim.cpt_code == "90658":
            if claim.patient_age <= 3:
                return -1.0, "DENIAL: Age Limit (< 3 years)"
                
        # Rule 3: Modifier Required
        if claim.cpt_code == "25111":
            if claim.modifier not in ["RT", "LT"]:
                return -1.0, "DENIAL: Missing/Invalid Modifier"
                
        return 1.0, "PAID"

# =============================================================================
# 4. ALFM-BEM Integration
# =============================================================================

class SymbolicProjector:
    """Maps categorical claims to dense vectors for BEM."""
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        # Random projection matrices for each feature
        np.random.seed(42)
        self.cpt_proj = {c: np.random.randn(dim) for c in ["99213", "90658", "25111", "99203", "71045"]}
        self.dx_proj = {d: np.random.randn(dim) for d in ["J01.90", "J20.9", "R05", "M67.4", "Z23"]}
        self.mod_proj = {m: np.random.randn(dim) for m in ["", "RT", "LT", "25", "59"]}
        
    def project(self, claim: Claim) -> np.ndarray:
        # Simple additive embedding: vec = E(cpt) + E(dx) + E(mod) + E(age_bucket)
        v_cpt = self.cpt_codes_vec(claim.cpt_code)
        v_dx = self.dx_codes_vec(claim.diagnosis_code)
        v_mod = self.mod_codes_vec(claim.modifier)
        
        # Age bucket embedding (random vector for each bucket)
        age_bucket = claim.patient_age // 10
        np.random.seed(age_bucket) # Deterministic per bucket
        v_age = np.random.randn(self.dim)
        
        vec = v_cpt + v_dx + v_mod + v_age
        return vec / (np.linalg.norm(vec) + 1e-10)

    def cpt_codes_vec(self, code):
        if code not in self.cpt_proj: return np.zeros(self.dim)
        return self.cpt_proj[code]
    
    def dx_codes_vec(self, code):
        if code not in self.dx_proj: return np.zeros(self.dim)
        return self.dx_proj[code]

    def mod_codes_vec(self, code):
        if code not in self.mod_proj: return np.zeros(self.dim)
        return self.mod_proj[code]


class HealthcareBEM:
    def __init__(self):
        self.projector = SymbolicProjector(dim=64)
        self.bem = BidirectionalExperienceMemory(
            dim=64,
            similarity_threshold=0.85, # High threshold for specific rules
            coverage_mode=CoverageMode.KDE
        )
        self.risk_threshold = 0.6
        
    def process(self, claim: Claim) -> Tuple[str, float]:
        """Returns (Action, RiskScore). Action in {SUBMIT, ABSTAIN}."""
        vec = self.projector.project(claim)
        risk, _ = self.bem.risk_signal(vec)
        
        if risk > self.risk_threshold:
            return "ABSTAIN", risk
        return "SUBMIT", risk
    
    def learn(self, claim: Claim, outcome: float):
        vec = self.projector.project(claim)
        # Store experience
        self.bem.add_experience(vec, outcome, str(claim))

# =============================================================================
# 5. Simulation Loop
# =============================================================================

def run_simulation(n_claims: int = 2000):
    print(f"Starting Healthcare Simulation (N={n_claims})...")
    
    generator = ClaimGenerator()
    payer = PayerEngine()
    agent = HealthcareBEM()
    
    history = []
    
    # Metrics
    window_size = 100
    rejection_rates = []
    abstain_rates = []
    
    recent_outcomes = [] # 1=Paid, 0=Rejected (for submitted claims)
    recent_actions = [] # 1=Abstain, 0=Submit
    
    for i in range(n_claims):
        claim = generator.generate(i)
        
        # 1. Agent Decision
        action, risk = agent.process(claim)
        
        # 2. Outcome
        if action == "ABSTAIN":
            # Human review simulates "fixing" the claim or confirming rejection
            # For simplicity, we assume human review catches the error (costly but safe)
            # We don't send to payer, so no rejection recorded.
            # But we query payer "offline" to learn.
            outcome, reason = payer.adjudicate(claim)
            final_status = "ABSTAINED"
            recent_actions.append(1)
        else:
            # Submit to payer
            outcome, reason = payer.adjudicate(claim)
            final_status = "PAID" if outcome > 0 else "REJECTED"
            recent_actions.append(0)
            
            if final_status == "REJECTED":
                recent_outcomes.append(0)
            else:
                recent_outcomes.append(1)
        
        # 3. Learn (BEM learns from ALL outcomes, even if abstained/simulated)
        agent.learn(claim, outcome)
        
        history.append({
            "id": i,
            "cpt": claim.cpt_code,
            "risk": risk,
            "action": action,
            "status": final_status,
            "reason": reason
        })
        
        # 4. Metrics (Rolling Window)
        if len(recent_outcomes) > window_size: recent_outcomes.pop(0)
        if len(recent_actions) > window_size: recent_actions.pop(0)
        
        if i % 50 == 0 and i > 0:
            rej_rate = 1.0 - (sum(recent_outcomes) / len(recent_outcomes) if recent_outcomes else 1.0)
            abs_rate = sum(recent_actions) / len(recent_actions) if recent_actions else 0.0
            
            rejection_rates.append(rej_rate)
            abstain_rates.append(abs_rate)
            
            print(f"Step {i}: Rejection Rate={rej_rate:.2f}, Abstain Rate={abs_rate:.2f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    steps = np.arange(len(rejection_rates)) * 50
    plt.plot(steps, rejection_rates, label="Rejection Rate (Submitted)", color="red")
    plt.plot(steps, abstain_rates, label="Human Review Rate", color="blue", linestyle="--")
    plt.xlabel("Claims Processed")
    plt.ylabel("Rate")
    plt.title("ALFM-BEM Learning Curve: Healthcare Claims")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("learning_curve.pdf")
    print("Saved learning_curve.pdf")
    
    # Analysis
    df = pd.DataFrame(history)
    print("\nTop Rejection Reasons (Initial):")
    print(df[df.index < 200][df.status == "REJECTED"].reason.value_counts())
    
    print("\nTop Rejection Reasons (Final):")
    print(df[df.index > n_claims - 200][df.status == "REJECTED"].reason.value_counts())

if __name__ == "__main__":
    run_simulation()
