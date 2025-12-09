"""
ALFM-BEM System Constants
=========================

Centralized definition of system thresholds and parameters.
"""

# Unified Thresholds
DEFAULT_SIMILARITY_THRESHOLD = 0.5  # For BEM neighbor filtering
DEFAULT_FAILURE_THRESHOLD = -0.3    # Outcome < -0.3 is Failure
DEFAULT_SUCCESS_THRESHOLD = 0.3     # Outcome > 0.3 is Success
DEFAULT_RISK_THRESHOLD = 0.6        # Risk > 0.6 triggers intervention
DEFAULT_CONFIDENCE_THRESHOLD = 0.7  # High confidence > 0.7

# KDE Parameters
KDE_BANDWIDTH = 0.3                 # Justified via analysis/bandwidth_selection.py (robustness preference)
