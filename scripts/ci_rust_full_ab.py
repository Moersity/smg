#!/usr/bin/env python3
"""Fork-only full SMG target experiment; preserve unrelated Actions caches.

Cold measurements use empty private Cargo homes and target directories, no
compiler wrapper, and separate fetch/build/test timings. Warm measurements
reuse each variant's artifacts and change the same library source in both.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def measure(root, variant, scenario, iteration, command, env, cwd):
    """Record elapsed time, exit status, resources, and the entire Cargo log."""
    results = root / "results"
    label = f"{variant}-{scenario}-{iteration}"
    log = results / f"{label}.log"
    resources = results / f"{label}-resources.txt"
    print(f"START {label}: {' '.join(command)}", flush=True)
    started = time.perf_counter()
    with log.open("w") as out:
        result = subprocess.run(
            ["/usr/bin/time", "-v", "-o", str(resources), *command],
            cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT,
        )
    elapsed = time.perf_counter() - started
    with (results / "timings.csv").open("a") as out:
        csv.writer(out).writerow([variant, scenario, iteration, f"{elapsed:.2f}", result.returncode])
    timing = Path(env["CARGO_TARGET_DIR"]) / "cargo-timings/cargo-timing.html"
    if timing.exists() and "--timings" in command:
        shutil.copyfile(timing, results / f"{label}.html")
    print(f"END {label}: {elapsed:.2f}s, exit {result.returncode}", flush=True)
    if result.returncode:
        print("\n".join(log.read_text().splitlines()[-60:]), flush=True)
        raise SystemExit(result.returncode)


def environment(root, variant, iteration, cold):
    """Isolate compilation caches without deleting repository or user caches."""
    env = os.environ.copy()
    state = root / f"{variant}-environment.json"
    if cold:
        target = root / f"target-{variant}"
        if target.exists():
            # This path is created exclusively by this benchmark under RUNNER_TEMP.
            shutil.rmtree(target)
        home = root / f"cargo-{variant}-{iteration}"
        home.mkdir()
        state.write_text(json.dumps({"CARGO_HOME": str(home), "CARGO_TARGET_DIR": str(target)}))
    env.update(json.loads(state.read_text()))
    env.update({
        "RUSTC_WRAPPER": "", "RUSTC_WORKSPACE_WRAPPER": "",
        "SCCACHE_GHA_ENABLED": "false", "CARGO_INCREMENTAL": "0",
    })
    return env


def prepare(root, base):
    """Create two worktrees from one frozen community revision."""
    root.mkdir()
    (root / "results").mkdir()
    (root / "results/timings.csv").write_text("variant,scenario,iteration,seconds,exit_code\n")
    for variant in ("before", "after"):
        source = root / variant
        subprocess.run(["git", "worktree", "add", "--detach", str(source), base], check=True)
        if variant == "after":
            manifest = source / "model_gateway/Cargo.toml"
            text = manifest.read_text()
            block = '[[bin]]\nname = "amg"\npath = "src/main.rs"\n\n'
            assert text.count(block) == 1
            manifest.write_text(text.replace(block, "", 1))
        shutil.copyfile(source / "model_gateway/src/lib.rs", root / f"{variant}-lib.rs")
    (root / "results/method.txt").write_text(
        f"base={base}\nSame runner and compiler profile, actual manifest removal.\n"
        "Scope: cargo test -p smg (library, CLI, integration targets and doctests).\n"
        "Cold: empty native target + empty Cargo home; wrappers disabled.\n"
        "Fetch recorded separately; build uses --offline. No global cache deletion.\n"
        "No target fingerprint manipulation or build timestamp suppression.\n"
    )


def run(root, variant, scenario, iteration):
    """Measure real Cargo commands under cold, unchanged, or changed conditions."""
    source = root / variant
    env = environment(root, variant, iteration, scenario == "cold")
    if scenario == "cold":
        measure(root, variant, "fetch", iteration, ["cargo", "fetch", "--locked"], env, source)
    if scenario == "changed":
        lib = source / "model_gateway/src/lib.rs"
        lib.write_text((root / f"{variant}-lib.rs").read_text()
                       + f"\n#[allow(dead_code)]\nconst CI_SOURCE_PROBE: u32 = {iteration};\n")
    measure(root, variant, scenario, iteration,
            ["cargo", "test", "--locked", "--offline", "-p", "smg", "--no-run", "--timings"], env, source)
    if scenario == "cold":
        # Match normal cargo test coverage; no name filters or added skip flags.
        measure(root, variant, "tests", iteration,
                ["cargo", "test", "--locked", "--offline", "-p", "smg"], env, source)
    disk = shutil.disk_usage(root)
    print(f"Free disk after {variant}/{scenario}: {disk.free / 2**30:.1f} GiB", flush=True)


def summarize(root):
    """Publish every sample without presenting a one-sample cold run as a median."""
    import statistics
    rows = list(csv.DictReader((root / "results/timings.csv").open()))
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as out:
        out.write("Full SMG targets; no global cache deletion. Fetch is separate from compilation.\n\n")
        out.write("| Variant | Scenario | Samples | Median seconds |\n|---|---|---:|---:|\n")
        for variant in ("before", "after"):
            for scenario in ("fetch", "cold", "tests", "unchanged", "changed"):
                values = [float(r["seconds"]) for r in rows if r["variant"] == variant
                          and r["scenario"] == scenario and r["exit_code"] == "0"]
                if values:
                    out.write(f"| {variant} | {scenario} | {len(values)} | {statistics.median(values):.2f} |\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "summarize"))
    parser.add_argument("--base", default="59f72433fe04a19755b17b595fdcbd12daab4a44")
    parser.add_argument("--variant", choices=("before", "after"))
    parser.add_argument("--scenario", choices=("cold", "unchanged", "changed"))
    parser.add_argument("--iteration", type=int, default=1)
    args = parser.parse_args()
    task_root = Path(os.environ["RUNNER_TEMP"]) / "smg-full-rust-ab"
    if args.action == "prepare":
        prepare(task_root, args.base)
    elif args.action == "run":
        run(task_root, args.variant, args.scenario, args.iteration)
    else:
        summarize(task_root)
