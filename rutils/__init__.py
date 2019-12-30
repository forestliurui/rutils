from .utils import tensor_to_image, fix_random_seed, visualize_dataset, decode_captions
from .eval import rel_error, compute_numeric_gradient
from . import data
from .gpu import get_free_gpu