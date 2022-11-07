import logging
import warnings
from tempfile import mkdtemp

import pandas as pd

__all__ = ['pipeline_memory', 'clustering', 'benchmark', 'ICA', 'PCA', 'RP']

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)

# Keep a cache for the pipelines to speed things up
pipeline_cachedir = mkdtemp()
# pipeline_memory = Memory(cachedir=pipeline_cachedir, verbose=10)
pipeline_memory = None
