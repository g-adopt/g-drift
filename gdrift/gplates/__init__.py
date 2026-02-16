try:
    from .coastlines import CoastlineVTKFile
except ImportError:
    CoastlineVTKFile = None

__all__ = [
    "CoastlineVTKFile"
]
