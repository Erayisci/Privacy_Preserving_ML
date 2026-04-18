"""Privacy-preserving ML pipeline with MIA evaluation.

Split-model chest X-ray pneumonia classifier. Teammates plug in
DP/BIE/SMPC modules via a shared PrivacyMechanism Protocol; this
package owns the encoder, head, MIA attacks, metrics, and runner.

Design: docs/superpowers/specs/2026-04-18-mia-design.md
"""

__version__ = "0.1.0"
