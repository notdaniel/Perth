from pathlib import Path
import importlib.resources as resources

try:
    PREPACKAGED_MODELS_DIR = str(resources.files(__name__) / "pretrained")
except AttributeError:
    import pkg_resources
    PREPACKAGED_MODELS_DIR = pkg_resources.resource_filename(__name__, "pretrained")

from .perth_net_implicit.perth_watermarker import PerthImplicitWatermarker
