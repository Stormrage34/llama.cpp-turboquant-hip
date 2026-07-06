# SPDX-FileCopyrightText: 2025 llama.cpp authors
# SPDX-License-Identifier: MIT

"""
Shared ROCTx marker context manager for all MNLN simulation engines.

Usage::

    from _roctx import mark, roctx

    with mark("occupancy_solver_vgpr"):
        result = solve_occupancy(...)

All calls are no-ops when libroctx64.so is absent (common).

Environment:
    ROCM_PATH    Path to ROCm install (default: /opt/rocm)
"""

import os
import ctypes
from contextlib import contextmanager


class _ROCTx:
    """Lazy-loaded ROCTx library bridge (singleton pattern)."""

    def __init__(self):
        self.lib = None
        self._push = None
        self._pop = None

    def _ensure(self):
        if self.lib is not None:
            return
        rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
        candidates = [
            os.path.join(rocm_path, "lib", "libroctx64.so"),
            "libroctx64.so",
        ]
        for path in candidates:
            try:
                self.lib = ctypes.CDLL(path)
                break
            except OSError:
                continue

        if self.lib is None:
            return  # no-ops forever

        try:
            fn = self.lib.roctxRangePushA
            fn.argtypes = [ctypes.c_char_p]
            self._push = fn
        except AttributeError:
            self._push = None

        try:
            fn = self.lib.roctxRangePop
            fn.argtypes = []
            self._pop = fn
        except AttributeError:
            self._pop = None

    def push_range(self, name: str):
        self._ensure()
        if self._push:
            self._push(name.encode("utf-8"))

    def pop_range(self):
        self._ensure()
        if self._pop:
            self._pop()


# Global singleton — importers use this directly.
roctx = _ROCTx()


@contextmanager
def mark(name: str):
    """Context manager / decorator: push *name* on enter, pop on exit.

    Example::

        with mark("my_computation"):
            ...
    """
    roctx.push_range(name)
    try:
        yield
    finally:
        roctx.pop_range()
