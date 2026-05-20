"""Compatibility helpers for Biomodals app configuration."""

from __future__ import annotations

from modal import Volume

from biomodals.schema.app import AppConfig as SchemaAppConfig


def get_output_volume(config: SchemaAppConfig) -> Volume:
    """Volume for storing outputs."""
    vol_name = f"{config.name}-outputs"
    return Volume.from_name(vol_name, create_if_missing=True, version=2)


class AppConfig(SchemaAppConfig):
    """App configuration with Modal-specific compatibility helpers."""

    def get_out_volume(self) -> Volume:
        """Volume for storing outputs."""
        return get_output_volume(self)


__all__ = ["AppConfig", "get_output_volume"]
