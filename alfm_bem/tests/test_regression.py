"""
Regression Tests for ALFM-BEM
=============================

Verifies critical system functionality:
1. Experience logic (success/failure definitions).
2. Anti-Poisoning (conflict detection).
3. System Integration (infer -> record_outcome loop).
"""

import unittest
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bem import BidirectionalExperienceMemory, Experience
from alfm_bem import create_alfm_bem, ALFMConfig

class TestALFMBEM(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        self.dim = 64
        
    def test_experience_logic(self):
        """Test basic experience properties."""
        exp_fail = Experience(
            embedding=np.random.randn(self.dim),
            outcome=-0.8,
            context_hash="fail_1",
            timestamp=datetime.now()
        )
        self.assertTrue(exp_fail.is_failure)
        self.assertFalse(exp_fail.is_success)
        self.assertEqual(exp_fail.severity, 4) # -0.8 -> Severity 4
        
        exp_succ = Experience(
            embedding=np.random.randn(self.dim),
            outcome=0.9,
            context_hash="succ_1",
            timestamp=datetime.now()
        )
        self.assertTrue(exp_succ.is_success)
        self.assertFalse(exp_succ.is_failure)
        self.assertEqual(exp_succ.severity, 1) # 0.9 -> Severity 1

    def test_anti_poisoning(self):
        """Test conflict detection mechanism (Task 4.1)."""
        bem = BidirectionalExperienceMemory(dim=self.dim, similarity_threshold=0.5)
        
        z = np.random.randn(self.dim)
        z /= np.linalg.norm(z)  # Normalize
        
        # 1. Establish a high-confidence failure
        # Add 3 identical experiences to confirm it
        for i in range(3):
            bem.add_experience(z, -1.0, f"failure_ctx_{i}")
        
        # Hack confirmation count for the last one to be 3 (or rely on logic if implemented)
        # BEM deduplication logic increases confirmation count if context_hash matches.
        # But we used unique hashes here. Let's just create one with confirmation=3 manually.
        bem.experiences = []
        exp_confirmed = Experience(
            embedding=z,
            outcome=-1.0,
            context_hash="confirmed_fail",
            timestamp=datetime.now(),
            confirmation_count=5
        )
        bem.experiences.append(exp_confirmed)
        bem._ensure_index()
        
        # 2. Try to add a success with HIGH similarity (same vector)
        conflict = bem.check_conflict(z, 1.0) # Outcome 1.0 (Success) vs stored -1.0
        
        # Should detect conflict
        self.assertIsNotNone(conflict)
        self.assertIn("Conflict detected", conflict)
        self.assertIn("confirmed 5x", conflict)
        
        # 3. Try adding failure (Consistent) -> No conflict
        conflict_ok = bem.check_conflict(z, -0.9)
        self.assertIsNone(conflict_ok)
        
        # 4. Try adding weak success (0.1) -> No conflict (ambiguous, not strong success)
        conflict_weak = bem.check_conflict(z, 0.1)
        self.assertIsNone(conflict_weak)

    def test_system_loop(self):
        """Integration test for full ALFM-BEM loop."""
        config = ALFMConfig(
            backbone_dim=self.dim,
            projection_dim=self.dim, # Identity projection for simplicity
            adapter_hidden_dim=32
        )
        system = create_alfm_bem(backbone_dim=self.dim, projection_dim=self.dim)
        
        z = np.random.randn(self.dim)
        context = {"text": "test context"}
        
        # 1. Infer
        res = system.infer(z, context, tenant_id="test_tenant")
        self.assertIsNotNone(res.decision)
        
        # 2. Record Outcome
        # Should pass
        system.record_outcome(z, str(context), -1.0, tenant_id="test_tenant")
        
        # Check if added to BEM
        stats = system.get_statistics("test_tenant")
        self.assertEqual(stats["bem"]["total"], 1)

    def test_vacuum(self):
        """Test memory vacuuming (Task 4.3)."""
        bem = BidirectionalExperienceMemory(dim=self.dim)
        z = np.random.randn(self.dim)
        
        # Add 15 experiences
        for i in range(15):
            bem.add_experience(z, 1.0, f"ctx_{i}")
            
        # Vacuum to max 10
        bem.vacuum(max_size=10)
        
        self.assertEqual(len(bem.experiences), 10)
        # Vacuum logic sorts by (confirmation, timestamp) desc, so keeps "best"/"newest"

    def test_scalable_index(self):
        """Test sklearn KNN index (Task 4.4)."""
        # Create BEM with scalable indexing enabled
        bem = BidirectionalExperienceMemory(
            dim=self.dim, 
            similarity_threshold=0.5,
            use_knn_index=True
        )
        
        # Add 100 random vectors
        vectors = np.random.randn(100, self.dim)
        for i, vec in enumerate(vectors):
            bem.add_experience(vec, 1.0, f"vec_{i}")
            
        # Target vector = vec_0 + small noise
        target = vectors[0] + np.random.randn(self.dim) * 0.1
        
        # Risk signal should use KNN fast path
        # We just verify it returns a result and doesn't crash
        # (Correctness relying on sklearn logic)
        sims = bem._compute_similarities(target)
        self.assertGreater(len(sims), 0)
        self.assertGreater(max(sims), 0.8) # Should find the neighbor

if __name__ == '__main__':
    unittest.main()
