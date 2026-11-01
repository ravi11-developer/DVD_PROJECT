import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

NGSPICE    = r"D:\ngspice-47_64\Spice64\bin\ngspice_con.exe"
MODEL_FILE = r"D:\DVD_PROJECT\model\22nm_MGK.pm"
INPUT_CSV  = "nmos_off_inputs.csv"
OUTPUT_CSV = "nmos_off.csv"
SIM_DIR    = "sim_nmos_off"
BATCH_SIZE = 1000
L          = 22e-9

COLUMNS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb",
           "vth0", "toxe", "ndep", "u0", "xj",
           "id", "ig", "is", "ib"]

WRDATA_VECTORS = ("v(ref) temp_c @mn[w] v(drain) v(source) v(gate) v(body) "
                  "@nmos[vth0] @nmos[toxe] @nmos[ndep] @nmos[u0] @nmos[xj] "
                  "i(vd) i(vg) i(vs) i(vb)")


def sample_commands(row):
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
wrdata batch_out.txt {WRDATA_VECTORS}
"""


def make_netlist(batch, model_path):
    header = f"""* NMOS OFF - 22nm_MGK
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
    body = "".join(sample_commands(row) for row in batch.itertuples())
    return header + body + ".endc\n.end\n"


def run_batch(batch, batch_no, sim_dir, model_path):
    net_file = sim_dir / f"batch_{batch_no:03d}.net"
    log_file = sim_dir / f"batch_{batch_no:03d}.log"
    out_file = sim_dir / "batch_out.txt"

    if out_file.exists():
        out_file.unlink()

    net_file.write_text(make_netlist(batch, model_path))

    subprocess.run([NGSPICE, "-b", net_file.name, "-o", log_file.name],
                   cwd=sim_dir, capture_output=True, text=True)

    if not out_file.exists():
        raise RuntimeError(f"Batch {batch_no}: no output. Check {log_file}")

    data = np.loadtxt(out_file, ndmin=2)

    if data.shape != (len(batch), len(COLUMNS) + 1):
        raise RuntimeError(f"Batch {batch_no}: expected {len(batch)} rows x "
                           f"{len(COLUMNS) + 1} cols, got {data.shape}. "
                           f"Check {log_file}")

    return data[:, 1:]


def main():
    inputs = pd.read_csv(INPUT_CSV)
    sim_dir = Path(SIM_DIR)
    sim_dir.mkdir(exist_ok=True)
    model_path = Path(MODEL_FILE).resolve().as_posix()

    results = []
    n_batches = int(np.ceil(len(inputs) / BATCH_SIZE))
    for b in range(n_batches):
        batch = inputs.iloc[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        results.append(run_batch(batch, b, sim_dir, model_path))
        print(f"Batch {b + 1}/{n_batches} done")

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

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV} with shape {df.shape}")


if __name__ == "__main__":
    main()
