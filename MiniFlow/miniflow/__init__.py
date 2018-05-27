from .function import *
from .graph import *
from .mnist import *
from .operator import *
from .optimizer import *
from .placeholder import *
from .session import *
from .variable import *

import builtins
DEFAULT_GRAPH = builtins.DEFAULT_GRAPH = Graph()
