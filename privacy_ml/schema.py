"""Encoder schema version for cache invalidation.

Bump SCHEMA_VERSION whenever the encoder architecture or embedding
semantics change; the cache's encoder_hash incorporates it, so every
cached encoder + embedding bundle becomes unreachable and will be
regenerated on the next run.

Kept in its own tiny module (no TensorFlow import) so that cache.py
and other lightweight code can consume it without pulling TF in.
"""
SCHEMA_VERSION: int = 2
