# Part B (gates) and Part C (3-bit CLA): leakage, SPICE vs ML

Run every script **from this folder** (`cd gate_circuit`). The scripts need
`ngspice` (tested with 42) and `numpy`, `pandas` and `matplotlib`. The ML step
also needs whatever packages your models need.

| Step | Script | What it does | Output |
|---|---|---|---|
| 1 | `01_make_corners.py` | Draws the 200 PVT corners (seed 2026), shared by the gates and the CLA | `data/corners.csv` |
| 2 | `02_gate_spice.py` | SPICE of INV/NAND2/NOR2/AND2/OR2, 18 cases × 200 corners. Writes example netlists. | `data/gate_spice.csv`, `netlists/{inv,nand2,nor2,and2,or2}.net` |
| 3 | `03_gate_transistor_oracle.py` | Splits every gate into transistors and re-simulates each one alone at the same voltages, to check the bookkeeping | `data/gate_transistors.csv` |
| 4 | `04_cla_spice.py` | SPICE of the full 3-bit CLA netlist, 128 vectors × 200 corners | `data/cla_spice.csv`, `netlists/cla3.net` |
| 5 | `05_gate_ml.py` | ML gate leakage from the 4 transistor models | `data/gate_ml.csv`, `data/gate_transistors_ml.csv` |
| 6 | `06_gate_analysis.py` | Gate table and parity plot (deliverables 6 and 7) plus error analysis | `results/gate/` |
| 7 | `07_cla_ml.py` | Hierarchical CLA leakage, summed from the gate ML results | `data/cla_ml.csv` |
| 8 | `08_cla_analysis.py` | CLA table and parity plot (deliverable 8) plus error decomposition | `results/cla/` |

Steps 1–4 are SPICE only. They have already been run, and their CSVs are in `data/`.

## Running the ML and analysis steps (5–8)

1. Fill in the four `predict_*` functions in `leakage_models.py`. Each one
   loads the final saved model for its case, applies that model's own
   preprocessing, and returns `[id, ig, is, ib]` with the dataset sign
   convention.
2. Run:

```bash
cd gate_circuit
python3 05_gate_ml.py            # add --side snk to use the ground-side sum
python3 06_gate_analysis.py
python3 07_cla_ml.py
python3 08_cla_analysis.py
```

To test the pipeline before the models are ready, run
`python3 05_gate_ml.py --backend oracle`. It uses the SPICE single-transistor
currents instead of ML, so the errors come out at about 0%. Re-run steps 5–8
with the real models before reporting anything.

## Definitions

- **Corner:** one draw of VDD (0.9 V ±10%), temperature (−40 to 125 °C), the
  NMOS (vth0, toxe, ndep, u0, xj) and, separately, the PMOS (vth0, toxe, ndep,
  u0, xj). Each parameter is its nominal value ±10%, uniform. All NMOS in a
  circuit share one draw and all PMOS share another.
- **Sizing:** Wn = 2·Lmin = 44 nm and Wp = 4·Lmin = 88 nm for every transistor;
  L = 22 nm.
- **Leakage:** current delivered by the VDD source plus current delivered by
  every input source held at logic 1. SPICE: `-i(Vdd) - Σ_{inputs=1} i(Vin)`.
  The same definition is used for gates and the CLA.
- **ML gate leakage:** each transistor's vd and vs come from the recorded
  SPICE node voltages. vg is the logic level (0 or VDD) of its gate node, and
  that level also picks the model case. The predicted currents are summed over
  the terminals on VDD-level source nodes (`src`, the default).
  - `snk` instead sums the currents leaving at terminals on GND or on logic-0
    inputs.
  - In SPICE both sums are equal (KCL), and step 3 confirms it.
- **Gate topologies:** NAND2 has its series NMOS stack out→x→GND. NOR2 has its
  series PMOS stack VDD→y→out. AND2 is NAND2 (output `nd`) followed by INV, and
  OR2 is NOR2 (output `nr`) followed by INV. Bodies are tied to GND (NMOS) and
  VDD (PMOS).
- **3-bit CLA:** 7 inputs (A2..0, B2..0, Cin). P = A⊕B, G = A·B, and full
  lookahead carries C1–C3 built from AND2/OR2. S = P⊕C. Each XOR is 4 NAND2.
  The total is 24 NAND2, 9 AND2 and 6 OR2: 39 gates and 186 transistors.
  Step 4 checks that the circuit adds correctly for all 128 vectors.
- **Metrics:**
  - bias = median of (ML−SPICE)/SPICE
  - median error and p95 error are taken over |(ML−SPICE)/SPICE|
  - mean accuracy = 100 − mean(min(|err|, 100))
