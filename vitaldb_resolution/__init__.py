"""vitaldb_resolution: every recorded VitalDB time and value as a probability distribution of the true value.

```python
import vitaldb_resolution as vr
reg = vr.load()
d = reg.distribution('intraop_uo', 100)
d.mean(), d.interval(0.95), d.sample(1000), d.to_dict()
```
"""
from . import versions
from .cache import Cache
from .distribution import Component, Distribution
from .registry import Registry, load

__all__ = ['Cache', 'Component', 'Distribution', 'Registry', 'load']

__version__ = '0.6.0'
