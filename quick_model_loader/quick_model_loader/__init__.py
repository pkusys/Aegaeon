# __init__.py for quick model loader

# Version of the package
__version__ = "0.0.1"

# Importing submodules or symbols to make them available at the package level
from quick_model_loader.model_loader import QuickModelLoader
from quick_model_loader.meta import TensorInfo, TensorsMeta

__all__ = ["QuickModelLoader", "TensorsMeta", "TensorInfo"]
