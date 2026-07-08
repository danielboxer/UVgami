try:
    from ._core import *           # compiled extension
except ImportError:
    print("ERROR: _core.so not found")
    print("HINT: are your running python -m demo.partuv_demo at the root folder of the project? This sometimes routes `import partuv` to the local path. Go to demo folder if needed.")
    raise ImportError("ImportError: _core.so not found")
__all__ = [n for n in dir() if not n.startswith("_")] + ["preprocess"]


# lazy: .preprocess pulls the torch stack, absent without the [ai] extra
def __getattr__(name):
    if name == "preprocess":
        from .preprocess import preprocess

        return preprocess
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
