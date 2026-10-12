"""
Alias module to support standard Python imports without spaces.
Delegates everything directly to 'dymaic pipiles_.py'.
"""
import os
import importlib.util

_current_dir = os.path.dirname(os.path.abspath(__file__))
_target_path = os.path.join(_current_dir, "dymaic pipiles_.py")

if os.path.exists(_target_path):
    _spec = importlib.util.spec_from_file_location("dymaic_pipiles_module", _target_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    # Expose all functions, classes, and variables
    globals().update({k: v for k, v in _mod.__dict__.items() if not k.startswith("__")})


