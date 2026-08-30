#!/usr/bin/env python3
"""Count the storage buffers each ONNX node would bind in a WebGPU kernel.

    pip install onnx
    python3 tools/onnx-buffers.py model.onnx [...]

ONNX Runtime's WebGPU backend binds roughly one storage buffer per runtime
input plus one per output, and aborts when a single kernel exceeds the device's
maxStorageBuffersPerShaderStage:

    numbers_storage_buffers_ <= limits_.maxStorageBuffersPerShaderStage
    Too many storage buffers in shader. Current: 11, Max is 10

Initialisers do not count — they are constants, uploaded once. Wide Concat is
the usual culprit: every input is its own buffer.

Measured with this script:

    studioludens/birefnet-lite-512   max 7    (what the site ships)
    onnx-community/BiRefNet_lite     max 1025 (a Concat with 1024 inputs)

which is why the 512 export runs on WebGPU and the 1024 one cannot.
"""

import sys
from collections import Counter

import onnx


def report(path: str) -> None:
    model = onnx.load(path, load_external_data=False)
    initialisers = {i.name for i in model.graph.initializer}

    rows = []
    per_op = Counter()
    for node in model.graph.node:
        runtime_inputs = [i for i in node.input if i and i not in initialisers]
        buffers = len(runtime_inputs) + len(node.output)
        per_op[node.op_type] = max(per_op[node.op_type], buffers)
        rows.append((buffers, node.op_type, node.name, len(runtime_inputs), len(node.output)))

    rows.sort(reverse=True)
    print(f"=== {path} ===")
    print(f"  {len(model.graph.node)} nodes, worst kernel binds {rows[0][0]} storage buffers")
    for buffers, op, name, ins, outs in rows[:5]:
        print(f"    {buffers:5d}  {op:<10} {ins} runtime inputs, {outs} outputs   {name[:44]}")
    print(f"  per op type: {dict(sorted(per_op.items(), key=lambda kv: -kv[1])[:5])}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for path in sys.argv[1:]:
        report(path)
