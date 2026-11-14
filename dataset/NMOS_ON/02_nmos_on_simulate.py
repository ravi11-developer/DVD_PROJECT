import argparse
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE       = Path(__file__).resolve().parent
NGSPICE    = os.environ.get("NGSPICE") or shutil.which("ngspice_con") or shutil.which("ngspice") or "ngspice"
MODEL_FILE = HERE.parents[1] / "model" / "22nm_MGK.pm"
INPUT_CSV  = HERE / "nmos_on_inputs.csv"
OUTPUT_CSV = HERE / "nmos_on.csv"
SIM_DIR    = HERE / "sim_nmos_on"
BATCH_SIZE = 1000
L          = 22e-9

COLUMNS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb",
           "vth0", "toxe", "ndep", "u0", "xj",
           "id", "ig", "is", "ib"]

WRDATA_VECTORS = ("v(ref) temp_c @mn[w] v(drain) v(source) v(gate) v(body) "
                  "@nmos[vth0] @nmos[toxe] @nmos[ndep] @nmos[u0] @nmos[xj] "
                  "i(vd) i(vg) i(vs) i(vb)")


def sample_commands(row, out_name):
    return f"""altermod nmos vth0={row.vth0:.10e}
altermod nmos toxe={row.toxe:.10e}
altermod nmos ndep={row.ndep:.10e}
altermod nmos u0={row.u0:.10e}
altermod nmos xj={row.xj:.10e}
alter mn w={row.w:.10e}
alter vdd dc={row.vdd:.10e}
alter vd dc={row.vd:.10e}
alter vg dc={row.vg:.10e}
alter vs dc={row.vs:.10e}
alter vb dc={row.vb:.10e}
option temp={row.temp:.10e}
op
let temp_c = {row.temp:.10e}
wrdata {out_name} {WRDATA_VECTORS}
"""


def make_netlist(batch, model_path, out_name):
    header = f"""* NMOS ON - 22nm_MGK
.include "{model_path}"
.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9

Mn drain gate source body nmos W={L:.4e} L={L:.4e}
Vd  drain  0 0
Vg  gate   0 0
Vs  source 0 0
Vb  body   0 0
Vdd ref    0 0

.control
set wr_singlescale
set appendwrite
"""
    body = "".join(sample_commands(row, out_name) for row in batch.itertuples())
    return header + body + ".endc\n.end\n"


def run_batch(batch, batch_no, sim_dir, model_path):
    net_file = sim_dir / f"batch_{batch_no:03d}.net"
    log_file = sim_dir / f"batch_{batch_no:03d}.log"
    out_file = sim_dir / f"batch_{batch_no:03d}_out.txt"   # one file per batch: batches can run in parallel

    if out_file.exists():
        out_file.unlink()

    net_file.write_text(make_netlist(batch, model_path, out_file.name))

    subprocess.run([NGSPICE, "-b", net_file.name, "-o", log_file.name],
                   cwd=sim_dir, capture_output=True, text=True)

    if not out_file.exists():
        raise RuntimeError(f"Batch {batch_no}: no output. Check {log_file}")

    data = np.loadtxt(out_file, ndmin=2)
    out_file.unlink()

    if data.shape != (len(batch), len(COLUMNS) + 1):
        raise RuntimeError(f"Batch {batch_no}: expected {len(batch)} rows x "
                           f"{len(COLUMNS) + 1} cols, got {data.shape}. "
                           f"Check {log_file}")

    return data[:, 1:]


def main():
    ap = argparse.ArgumentParser(description="Simulate the input samples in ngspice (batches of 1000).")
    ap.add_argument("--inputs", type=Path, default=INPUT_CSV, help="input samples CSV")
    ap.add_argument("--output", type=Path, default=OUTPUT_CSV, help="output dataset CSV")
    ap.add_argument("--simdir", type=Path, default=SIM_DIR, help="folder for batch netlists / logs")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="ngspice runs in parallel")
    args = ap.parse_args()

    inputs = pd.read_csv(args.inputs)
    sim_dir = Path(args.simdir)
    sim_dir.mkdir(exist_ok=True)
    model_path = Path(MODEL_FILE).resolve().as_posix()

    n_batches = int(np.ceil(len(inputs) / BATCH_SIZE))
    batches = [inputs.iloc[b * BATCH_SIZE:(b + 1) * BATCH_SIZE] for b in range(n_batches)]

    def job(b):
        out = run_batch(batches[b], b, sim_dir, model_path)
        print(f"Batch {b + 1}/{n_batches} done", flush=True)
        return out

    with ThreadPoolExecutor(max_workers=args.workers) as ex:      # results come back in batch order
        results = list(ex.map(job, range(n_batches)))

    df = pd.DataFrame(np.vstack(results), columns=COLUMNS)

    for col in ["id", "ig", "is", "ib"]:
        df[col] = -df[col]

    input_cols = COLUMNS[:12]
    rel_diff = ((df[input_cols] - inputs[input_cols]).abs()
                / (inputs[input_cols].abs() + 1e-30))
    print("Largest input read-back mismatch:", float(rel_diff.max().max()))

    df["itotal"] = df["id"] + df["ig"] + df["is"] + df["ib"]
    kcl_sum = df["itotal"]
    largest = df[["id", "ig", "is", "ib"]].abs().max(axis=1)
    kcl_rel = kcl_sum.abs() / largest
    kcl_pass = (kcl_rel < 1e-6) | (kcl_sum.abs() < 1e-14)
    print("KCL relative residual  median:", float(kcl_rel.median()),
          " max:", float(kcl_rel.max()))
    print("KCL absolute residual  max (A):", float(kcl_sum.abs().max()))
    print(f"KCL pass: {int(kcl_pass.sum())}/{len(df)} rows")

    df.to_csv(args.output, index=False)
    print(f"Saved {args.output.name} with shape {df.shape}")


if __name__ == "__main__":
    main()
