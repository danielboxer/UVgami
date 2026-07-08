import os as _os
import sys as _sys

# bundled runtime dlls (gmp, mpfr, yaml-cpp) live next to the extension
if _sys.platform == "win32":
    _libs = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".libs")
    if _os.path.isdir(_libs):
        _os.add_dll_directory(_libs)

try:
    from ._core import *           # compiled extension
except ImportError:
    print("ERROR: _core.so not found")
    print("HINT: are your running python -m demo.partuv_demo at the root folder of the project? This sometimes routes `import partuv` to the local path. Go to demo folder if needed.")
    if _sys.platform == "win32":
        print("HINT: on Windows this also fails without the latest VC++ redistributable (msvcp140, libomp140)")
    raise ImportError("ImportError: _core.so not found")
__all__ = [n for n in dir() if not n.startswith("_")] + ["preprocess"]


# lazy: .preprocess pulls the torch stack, absent without the [ai] extra
def __getattr__(name):
    if name == "preprocess":
        from .preprocess import preprocess

        return preprocess
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
