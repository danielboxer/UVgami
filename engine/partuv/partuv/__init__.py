try:
    from ._core import *           # compiled extension
except ImportError:
    print("ERROR: _core.so not found")
    print("HINT: are your running python -m demo.partuv_demo at the root folder of the project? This sometimes routes `import partuv` to the local path. Go to demo folder if needed.")
    raise ImportError("ImportError: _core.so not found")
from .preprocess import preprocess
__all__ = [n for n in dir() if not n.startswith("_")]
