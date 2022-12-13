# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os.path

from jax_ipu_addons.utils import cppimport_append_include_dirs

from jax_ipu_research.utils import env_cpath_append

# Update default `cppimport` external libraries directory.
cppimport_append_include_dirs([os.path.join(os.path.dirname(__file__), "../external")])

# Local include PATH update.
env_cpath_append(os.path.join(os.path.dirname(__file__), "vertex"))
