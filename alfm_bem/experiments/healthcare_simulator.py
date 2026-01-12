"""
Healthcare Claim Simulator (Comparative Study)
==============================================

Simulates a stream of medical claims submitted to a Payer with hidden rejection rules.
Compares three approaches for Pre-Submission Scrubbing:
1. ALFM-BEM: Risk based on outcome-weighted neighbors.
2. RAG: Retrieval of top-k similar failures.
3. NEP: Density estimation (OOD detection).

Rules (Hidden from Agent):
1. Medical Necessity: CPT 99213 requires Dx J01.90/J20.9.
2. Age Limit: CPT 90658 only allowed for age > 3.
3. Modifier Required: CPT 25111 requires modifier RT/LT.
"""

from __future__ import annotations

import os
import json
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import random
from abc import ABC, abstractmethod
from pathlib import Path

_CACHE_ROOT = Path(__file__).resolve().parents[1] / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "fontconfig").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alfm_bem.bem import BidirectionalExperienceMemory, CoverageMode

# ... (Domain Entities & Generator - Same as before)
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

class ClaimGenerator:
    def __init__(self):
        self.cpt_codes = ["99213", "90658", "25111", "99203", "71045"]
        self.diagnosis_codes = ["J01.90", "J20.9", "R05", "M67.4", "Z23"]
        self.modifiers = ["", "RT", "LT", "25", "59"]
        
    def generate(self, claim_id: int) -> Claim:
        cpt = random.choice(self.cpt_codes)
        if cpt == "99213": dx = random.choice(["J01.90", "J20.9", "R05"]) 
        elif cpt == "25111": dx = "M67.4"
        elif cpt == "90658": dx = "Z23"
        else: dx = random.choice(self.diagnosis_codes)
        age = random.randint(1, 80)
        if cpt == "25111":
            # Make missing modifiers a realistic-but-not-dominant failure mode so
            # baseline rejection rates are in a plausible regime (~10–15%).
            mod = random.choices(["RT", "LT", ""], weights=[0.4, 0.4, 0.2], k=1)[0]
        else:
            mod = random.choice(self.modifiers) if random.random() > 0.5 else ""
        return Claim(str(claim_id), f"PROV_{random.randint(1,5)}", cpt, dx, age, mod)

class PayerEngine:
    def adjudicate(self, claim: Claim) -> Tuple[float, str]:
        if claim.cpt_code == "99213" and claim.diagnosis_code not in ["J01.90", "J20.9"]:
            return -1.0, "DENIAL: Medical Necessity"
        if claim.cpt_code == "90658" and claim.patient_age <= 3:
            return -1.0, "DENIAL: Age Limit"
        if claim.cpt_code == "25111" and claim.modifier not in ["RT", "LT"]:
            return -1.0, "DENIAL: Missing Modifier"
        return 1.0, "PAID"

class SymbolicProjector:
    def __init__(self, dim: int = 64):
        self.dim = dim
        self._rng = np.random.default_rng(42)
        self.cpt_proj = {c: self._rng.standard_normal(dim) for c in ["99213", "90658", "25111", "99203", "71045"]}
        self.dx_proj = {d: self._rng.standard_normal(dim) for d in ["J01.90", "J20.9", "R05", "M67.4", "Z23"]}
        self.mod_proj = {m: self._rng.standard_normal(dim) for m in ["", "RT", "LT", "25", "59"]}
        self.age_seeds = {a: self._rng.standard_normal(dim) for a in range(10)} # Age buckets
        
    def project(self, claim: Claim) -> np.ndarray:
        v_cpt = self.cpt_proj.get(claim.cpt_code, np.zeros(self.dim))
        v_dx = self.dx_proj.get(claim.diagnosis_code, np.zeros(self.dim))
        v_mod = self.mod_proj.get(claim.modifier, np.zeros(self.dim))
        bucket = claim.patient_age // 10
        if bucket not in self.age_seeds:
            self.age_seeds[bucket] = self._rng.standard_normal(self.dim)
        v_age = self.age_seeds[bucket]
        vec = v_cpt + v_dx + v_mod + v_age
        return vec / (np.linalg.norm(vec) + 1e-10)

# --- Agents ---

class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.projector = SymbolicProjector()
        
    @abstractmethod
    def process(self, claim: Claim) -> Tuple[str, float]:
        pass
    
    @abstractmethod
    def learn(self, claim: Claim, outcome: float):
        pass

class TrustAllAgent(BaseAgent):
    """Baseline: always submit, never abstain."""

    def __init__(self):
        super().__init__("TrustAll")

    def process(self, claim: Claim) -> Tuple[str, float]:
        return "SUBMIT", 0.0

    def learn(self, claim: Claim, outcome: float):
        return

class BEMAgent(BaseAgent):
    def __init__(self):
        super().__init__("BEM")
        # High similarity for precision
        self.bem = BidirectionalExperienceMemory(dim=64, similarity_threshold=0.85)
        self.risk_threshold = 0.6
        
    def process(self, claim: Claim) -> Tuple[str, float]:
        vec = self.projector.project(claim)
        risk, _ = self.bem.risk_signal(vec)
        if risk > self.risk_threshold: return "ABSTAIN", risk
        return "SUBMIT", risk
        
    def learn(self, claim: Claim, outcome: float):
        vec = self.projector.project(claim)
        self.bem.add_experience(vec, outcome, str(claim))

class RAGAgent(BaseAgent):
    """
    Retrieves Top-K failures. If any match is found > threshold, Abstain.
    This mimics a 'Retrieval-Based Guardrail'.
    """
    def __init__(self):
        super().__init__("RAG")
        self.failures = [] # List of (embedding, claim_str)
        self.threshold = 0.85 # Same as BEM sim threshold
        self.k = 5
        
    def process(self, claim: Claim) -> Tuple[str, float]:
        vec = self.projector.project(claim)
        # Naive linear scan for top-k (simulation scale is small)
        if not self.failures: return "SUBMIT", 0.0
        
        sims = [np.dot(vec, f_vec) for f_vec, _ in self.failures]
        if not sims: return "SUBMIT", 0.0
        
        max_sim = max(sims)
        if max_sim > self.threshold:
            # RAG says: "This looks like a known failure"
            return "ABSTAIN", max_sim
        return "SUBMIT", max_sim
        
    def learn(self, claim: Claim, outcome: float):
        if outcome < 0:
            vec = self.projector.project(claim)
            self.failures.append((vec, str(claim)))

class NEPAgent(BaseAgent):
    """
    Novelty Detection (NEP).
    Abstains if coverage (density) is low.
    Assumes low density = Risk.
    Does NOT use failure labels directly for decision, only coverage.
    """
    def __init__(self):
        super().__init__("NEP")
        self.bem = BidirectionalExperienceMemory(dim=64, coverage_mode=CoverageMode.KDE)
        self.coverage_threshold = 0.3 # If coverage < 0.3, Abstain
        
    def process(self, claim: Claim) -> Tuple[str, float]:
        vec = self.projector.project(claim)
        success, _ = self.bem.success_signal(vec) # Unused
        risk, _ = self.bem.risk_signal(vec) # Unused
        coverage = self.bem.coverage_signal(vec)
        
        if coverage < self.coverage_threshold:
            return "ABSTAIN", (1.0 - coverage) # Inverse coverage as risk
        return "SUBMIT", (1.0 - coverage)
        
    def learn(self, claim: Claim, outcome: float):
        # NEP simply observes distribution
        vec = self.projector.project(claim)
        self.bem.add_experience(vec, outcome, str(claim))

def run_simulation(
    n_claims: int = 2000,
    *,
    seed: int = 42,
    window: int = 100,
    save_json: bool = True,
) -> Dict[str, Dict[str, float]]:
    print(f"Starting Multi-Agent Simulation (N={n_claims}, seed={seed})...")
    random.seed(seed)
    np.random.seed(seed)
    
    gen = ClaimGenerator()
    payer = PayerEngine()
    
    agents = [TrustAllAgent(), BEMAgent(), RAGAgent(), NEPAgent()]
    
    # Store results per agent
    results = {a.name: {"rej": [], "abs": []} for a in agents}
    history = {a.name: {"abstain": [], "submitted": [], "rejected": []} for a in agents}
    
    # Use SAME random seed for claim generation per step could be tricky if we want them to see SAME claims.
    # We will generate a list of claims upfront.
    claims = [gen.generate(i) for i in range(n_claims)]
    
    for i, claim in enumerate(claims):
        
        # Ground truth (what happens if submitted)
        gt_outcome, reason = payer.adjudicate(claim)
        
        for agent in agents:
            # Decision
            action, risk = agent.process(claim)
            
            # Record Action
            if action == "ABSTAIN":
                history[agent.name]["abstain"].append(1.0)
                history[agent.name]["submitted"].append(0.0)
                history[agent.name]["rejected"].append(0.0)
            else:
                history[agent.name]["abstain"].append(0.0)
                history[agent.name]["submitted"].append(1.0)
                history[agent.name]["rejected"].append(1.0 if gt_outcome < 0 else 0.0)
            
            # Learn
            agent.learn(claim, gt_outcome)
            
        # Logging
        if i % 50 == 0 and i > 0:
            for agent in agents:
                abst = history[agent.name]["abstain"][-window:]
                submitted = history[agent.name]["submitted"][-window:]
                rejected = history[agent.name]["rejected"][-window:]
                
                abstain_rate = float(sum(abst) / len(abst))
                submitted_count = max(1.0, float(sum(submitted)))
                rejection_rate = float(sum(rejected) / submitted_count)
                
                results[agent.name]["rej"].append(rejection_rate)
                results[agent.name]["abs"].append(abstain_rate)
            
            # Print BEM stats as proxy
            print(f"Step {i}: BEM Rej={results['BEM']['rej'][-1]:.2f} Abs={results['BEM']['abs'][-1]:.2f}")

    # Plot
    plt.figure(figsize=(12, 5))
    steps = np.arange(len(results['BEM']['rej'])) * 50
    
    # Subplot 1: Failure Rate (Lower is Better)
    plt.subplot(1, 2, 1)
    for name in results:
        plt.plot(steps, results[name]['rej'], label=name, linewidth=2)
    plt.title("Rejection Rate (Submitted Claims)")
    plt.xlabel("Claims")
    plt.ylabel("Rejection Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Abstention Rate (Cost)
    plt.subplot(1, 2, 2)
    for name in results:
        plt.plot(steps, results[name]['abs'], label=name, linewidth=2, linestyle='--')
    plt.title("Abstention Rate (Cost)")
    plt.xlabel("Claims")
    plt.ylabel("Abstain Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = Path(__file__).parent / "learning_curve.pdf"
    plt.savefig(out_path)
    print(f"Saved {out_path}")

    summary: Dict[str, Dict[str, float]] = {}
    for agent in agents:
        name = agent.name
        abstain = np.asarray(history[name]["abstain"], dtype=float)
        submitted = np.asarray(history[name]["submitted"], dtype=float)
        rejected = np.asarray(history[name]["rejected"], dtype=float)
        if rejected.size == 0:
            continue
        submitted_total = max(1.0, float(np.sum(submitted)))
        submitted_final = max(1.0, float(np.sum(submitted[-window:])))
        summary[name] = {
            "rejection_rate_submitted_overall": float(np.sum(rejected) / submitted_total),
            "rejection_rate_submitted_final_window": float(np.sum(rejected[-window:]) / submitted_final),
            "abstain_rate_overall": float(np.mean(abstain)),
            "abstain_rate_final_window": float(np.mean(abstain[-window:])),
            "effective_rejection_rate_overall": float(np.sum(rejected) / float(len(rejected))),
            "effective_rejection_rate_final_window": float(np.sum(rejected[-window:]) / float(len(rejected[-window:]))),
        }

    if "TrustAll" in summary and "BEM" in summary:
        baseline = summary["TrustAll"]["rejection_rate_submitted_final_window"]
        bem_final = summary["BEM"]["rejection_rate_submitted_final_window"]
        summary["__comparison__"] = {
            "baseline_rejection_rate_submitted_final_window": baseline,
            "bem_rejection_rate_submitted_final_window": bem_final,
            "relative_reduction": float((baseline - bem_final) / (baseline + 1e-12)),
        }

    if save_json:
        metrics_path = Path(__file__).parent / "healthcare_metrics.json"
        metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {metrics_path}")

    return summary

if __name__ == "__main__":
    run_simulation()
