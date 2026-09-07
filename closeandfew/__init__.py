"""Leakage-resistant closed-set and few-shot registration experiments."""

from .protocol import ProtocolSpec, build_protocol_manifests, load_metadata

__all__ = ["ProtocolSpec", "build_protocol_manifests", "load_metadata"]
