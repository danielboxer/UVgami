import importlib.util as _util
import os as _os
import sys as _sys

# bundled runtime dlls (gmp, mpfr, yaml-cpp) live next to the extension
if _sys.platform == "win32":
    _libs = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".libs")
    if _os.path.isdir(_libs):
        _os.add_dll_directory(_libs)

# python -m partuv must import even when the compiled core can't load, so the
# cli can surface the reinstall hint. record the failure instead of raising at
# import.
_CORE_ERROR = None
try:
    from ._core import *           # compiled extension
except ImportError as _error:
    _CORE_ERROR = _error
    __all__ = ["preprocess"]
else:
    __all__ = [n for n in dir() if not n.startswith("_")] + ["preprocess"]


# lazy: .preprocess pulls the torch stack, absent without the [ai] extra
def __getattr__(name):
    if name == "preprocess":
        from .preprocess import preprocess

        return preprocess
    # a name that would have come from the compiled core, accessed while the
    # core failed to load: surface the build hints. submodules (cli, common,
    # ...) and dunders must still resolve, so let those fall through to
    # AttributeError
    if (
        _CORE_ERROR is not None
        and not name.startswith("_")
        and _util.find_spec(f"{__name__}.{name}") is None
    ):
        print("ERROR: _core.so not found")
        print("HINT: are your running python -m demo.partuv_demo at the root folder of the project? This sometimes routes `import partuv` to the local path. Go to demo folder if needed.")
        if _sys.platform == "win32":
            print("HINT: on Windows this fails without the VC++ redistributable (msvcp140); dev builds also need libomp140 (release wheels bundle it in .libs)")
        raise ImportError("_core.so not found") from _CORE_ERROR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
