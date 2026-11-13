"""Helpers for writing ngspice netlists, running them in parallel and reading
the `wrdata` output back."""
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from config import MODEL_FILE, NGSPICE, N_WORKERS, PROC_PARAMS, SPICE_OPTIONS


def header(title):
    return (f"* {title}\n"
            f".include \"{MODEL_FILE.resolve().as_posix()}\"\n"
            f"{SPICE_OPTIONS}\n\n")


def corner_cmds(c):
    """ngspice control lines that apply one corner (row of corners.csv).
    All NMOS share one process draw, all PMOS share another."""
    lines = [f"altermod nmos {p}={c['n_' + p]:.10e}" for p in PROC_PARAMS]
    lines += [f"altermod pmos {p}={c['p_' + p]:.10e}" for p in PROC_PARAMS]
    lines += [f"alter vdd dc={c['vdd']:.10e}",
              f"option temp={c['temp']:.10e}",
              f"let tc = {c['temp']:.10e}"]
    return "\n".join(lines) + "\n"


def control_block(body):
    return (".control\nset wr_singlescale\nset appendwrite\nset numdgt=12\n"
            + body + ".endc\n.end\n")


def _run_one(job):
    net, out, expected_rows, n_vec = job
    net, out = Path(net), Path(out)
    if out.exists():
        out.unlink()
    log = net.with_suffix(".log")
    subprocess.run([NGSPICE, "-b", net.name, "-o", log.name], cwd=net.parent,
                   capture_output=True, text=True)
    if not out.exists():
        raise RuntimeError(f"ngspice produced no output for {net}. See {log}")
    data = np.loadtxt(out, ndmin=2)
    # with wr_singlescale the first column is the scale (a copy of vector 1)
    if data.shape != (expected_rows, n_vec + 1):
        raise RuntimeError(f"{net}: expected {(expected_rows, n_vec + 1)}, got "
                           f"{data.shape}. See {log}")
    return data[:, 1:]


def run_jobs(jobs, label="", stack=True):
    """jobs: list of (netlist path, wrdata output path, expected rows, n vectors).
    Runs them in parallel and returns the results in job order (stacked into
    one array if `stack`, else as a list)."""
    results = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_run_one, j): i for i, j in enumerate(jobs)}
        done = 0
        for f in futs:
            results[futs[f]] = f.result()
            done += 1
            print(f"\r  {label} {done}/{len(jobs)} netlists done", end="", flush=True)
    print()
    return np.vstack(results) if stack else results
