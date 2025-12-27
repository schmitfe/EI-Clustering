"""Package for simulating binary balanced networks."""

from . import mean_field
from . import weights

try:
    from . import network  # Optional: requires compiled extensions
except ImportError:
    network = None

__all__ = ['mean_field', 'weights', 'network']
