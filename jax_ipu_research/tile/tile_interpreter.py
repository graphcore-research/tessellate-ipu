# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
"""Building IPU tile MPMD programming as a custom JAX interpreter (https://github.com/google/jax/tree/main/jax/interpreters).

In particular, we need a registry mapping JAX primitives to IPU vertex (and additionally support custom IPU vertex).
"""
from dataclasses import dataclass
from typing import Callable, Dict, List

from jax.core import Primitive
from jax.interpreters.xla import ShapedArray


@dataclass
class IpuVertexInfo:
    """Ipu vertex info: all required fields for adding a vertex to a Poplar graph and compute set.

    Args:
        name: (Short) name of the vertex.
        fullname: Full Poplar name of the vertex.
    """

    name: str
    fullname: str


IpuVertexTranslation = Callable[[Primitive, List[ShapedArray], List[ShapedArray]], IpuVertexInfo]
"""Ipu vertex translation: callable translating a JAX primitive (with inputs/outputs) into a full
vertex info data structure.
"""

_ipu_vertex_registry: Dict[Primitive, IpuVertexTranslation]
"""Global registry mapping JAX primitives to IPU vertex translation rules.
"""


def register_ipu_vertex_primitive(primitive: Primitive, translation: IpuVertexTranslation):
    """Register an IPU tile vertex translation from JAX primitive.

    Args:
        primitive: JAX primitive.
        translation: IPU vertex translation rule.
    """
    global _ipu_vertex_registry
    _ipu_vertex_registry[primitive] = translation
