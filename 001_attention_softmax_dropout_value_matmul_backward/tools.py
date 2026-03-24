#!/usr/bin/env python3
"""Tools for managing solution.json and performing basic checks."""

import json
import os
import sys
import glob

TASK_DIR = os.path.dirname(os.path.abspath(__file__))


def build_solution(round_name: str, description: str = "", name: str = "cc_attention_softmax_dropout_value_matmul_backward"):
    """Build solution.json from source files in the task directory.

    Reads all .cpp, .cu, .h, .cuh files from TASK_DIR and packages them
    into the solution.json format.

    Args:
        round_name: e.g. "round_0", "round_1"
        description: description of this solution attempt
        name: solution name
    """
    source_extensions = {'.cpp', '.cu', '.h', '.cuh'}
    sources = []

    for f in sorted(os.listdir(TASK_DIR)):
        _, ext = os.path.splitext(f)
        if ext in source_extensions:
            filepath = os.path.join(TASK_DIR, f)
            with open(filepath, 'r') as fh:
                content = fh.read()
            sources.append({"path": f, "content": content})

    if not sources:
        print("ERROR: No source files (.cpp, .cu, .h, .cuh) found in", TASK_DIR)
        sys.exit(1)

    # Check for required main.cpp with run entry point
    main_found = any(s["path"] == "main.cpp" for s in sources)
    if not main_found:
        print("WARNING: main.cpp not found - entry_point main.cpp::run will fail")

    solution = {
        "name": name,
        "definition": "attention_softmax_dropout_value_matmul_backward",
        "author": "cc",
        "spec": {
            "languages": ["cuda_cpp"],
            "target_hardware": ["B200", "LOCAL"],
            "entry_point": "main.cpp::run",
            "dependencies": [],
            "destination_passing_style": False,
            "binding": "torch"
        },
        "sources": sources,
        "description": description,
        "round_dir": round_name
    }

    solution_path = os.path.join(TASK_DIR, "solution.json")
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    print(f"Built solution.json with {len(sources)} source file(s):")
    for s in sources:
        lines = s["content"].count('\n') + 1
        print(f"  {s['path']} ({lines} lines)")
    print(f"Round: {round_name}")
    print(f"Saved to: {solution_path}")
    return solution


def validate_solution(solution_path=None):
    """Validate solution.json structure and basic content checks."""
    if solution_path is None:
        solution_path = os.path.join(TASK_DIR, "solution.json")

    with open(solution_path, 'r') as f:
        sol = json.load(f)

    errors = []
    warnings = []

    # Check required fields
    for field in ["name", "definition", "spec", "sources"]:
        if field not in sol:
            errors.append(f"Missing required field: {field}")

    # Check spec
    spec = sol.get("spec", {})
    if spec.get("entry_point") != "main.cpp::run":
        errors.append(f"entry_point should be 'main.cpp::run', got '{spec.get('entry_point')}'")
    if spec.get("binding") != "torch":
        errors.append(f"binding should be 'torch', got '{spec.get('binding')}'")

    # Check sources
    sources = sol.get("sources", [])
    if not sources:
        errors.append("No source files in solution")

    source_paths = [s["path"] for s in sources]
    if "main.cpp" not in source_paths:
        errors.append("main.cpp is missing from sources")

    # Check main.cpp has PYBIND11_MODULE and run function
    for s in sources:
        if s["path"] == "main.cpp":
            if "PYBIND11_MODULE" not in s["content"]:
                errors.append("main.cpp missing PYBIND11_MODULE")
            if 'def("run"' not in s["content"] and '"run"' not in s["content"]:
                warnings.append("main.cpp may be missing run function binding")

    # Check for .cu files (kernel must exist)
    cu_files = [s for s in sources if s["path"].endswith(".cu")]
    if not cu_files:
        warnings.append("No .cu kernel files found")

    # Report
    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  ✗ {e}")
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  ! {w}")
    if not errors and not warnings:
        print("✓ Solution looks valid")

    print(f"\nSolution: {sol.get('name', '?')}")
    print(f"Round: {sol.get('round_dir', '?')}")
    print(f"Sources: {', '.join(source_paths)}")

    return len(errors) == 0


def copy_to_round(round_name: str):
    """Copy solution.json to round directory."""
    round_dir = os.path.join(TASK_DIR, round_name)
    os.makedirs(round_dir, exist_ok=True)

    src = os.path.join(TASK_DIR, "solution.json")
    dst = os.path.join(round_dir, "solution.json")

    with open(src, 'r') as f:
        content = f.read()
    with open(dst, 'w') as f:
        f.write(content)

    print(f"Copied solution.json -> {round_dir}/solution.json")


def show_workloads():
    """Display workload configurations from workload.jsonl."""
    workload_path = os.path.join(TASK_DIR, "workload.jsonl")
    with open(workload_path, 'r') as f:
        for i, line in enumerate(f):
            wl = json.loads(line.strip())
            axes = wl["axes"]
            tol = wl["tolerance"]
            print(f"  WL {i:2d}: batch={axes['batch_size']:3d}  seq_q={axes['seq_len_q']:5d}  seq_kv={axes['seq_len_kv']:5d}  "
                  f"atol={tol['max_atol']:.0e}  rtol={tol['max_rtol']:.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python tools.py build <round_name> [description]  - Build solution.json from source files")
        print("  python tools.py validate                          - Validate solution.json")
        print("  python tools.py copy <round_name>                 - Copy solution.json to round dir")
        print("  python tools.py workloads                         - Show workload configs")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "build":
        round_name = sys.argv[2] if len(sys.argv) > 2 else "round_0"
        desc = sys.argv[3] if len(sys.argv) > 3 else ""
        build_solution(round_name, desc)
    elif cmd == "validate":
        validate_solution()
    elif cmd == "copy":
        round_name = sys.argv[2] if len(sys.argv) > 2 else "round_0"
        copy_to_round(round_name)
    elif cmd == "workloads":
        show_workloads()
    else:
        print(f"Unknown command: {cmd}")
