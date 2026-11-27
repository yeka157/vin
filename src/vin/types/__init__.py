"""Vin's named types, one module per area.

Import from the module that owns the type, for example `from vin.types.data import is_record`.
Each module imports only the standard library, Rich and other `vin.types` modules, so any
part of Vin can depend on it without an import cycle.
"""
