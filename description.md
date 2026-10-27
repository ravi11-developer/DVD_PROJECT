# ML-Driven PVT Variation Aware Leakage Current Estimation

## Project Description

This project addresses a critical challenge in modern VLSI design: **estimating leakage current in CMOS circuits** under **Process, Voltage, and Temperature (PVT) variations** using **Machine Learning (ML)** as a fast and accurate alternative to full SPICE simulations.

As transistor technology scales down to deep sub-micron nodes (45nm, 32nm, 22nm, 16nm), leakage current becomes a dominant component of total power consumption. Traditionally, leakage estimation relies on SPICE simulations, which are accurate but extremely time-consuming for large circuits at numerous PVT corners. This project replaces the SPICE simulation bottleneck with trained ML models that predict terminal currents (Id, Ig, Is, Ib) of individual transistors, and then hierarchically composes those predictions to estimate leakage for logic gates and complete circuits.

### Core Idea

1. **Train ML models on single-transistor SPICE data** — Four separate models are trained for NMOS ON, NMOS OFF, PMOS ON, and PMOS OFF states, each predicting the four terminal currents (drain, gate, source, body) given input parameters like VDD, temperature, width, terminal voltages, and process parameters (vth0, toxe, ndep, u0, xj).
2. **Compose transistor models to predict gate leakage** — Using the trained transistor-level models, predict leakage for standard CMOS gates (INV, NAND2, NOR2, AND2, OR2) for every input combination.
3. **Scale to circuit level** — Build a larger test circuit (3-bit Carry-Lookahead Adder or 8-bit Parity Checker) from gate-level predictions and validate against full SPICE simulation.

### Target Accuracy
Up to **99% mean accuracy** compared to SPICE simulation results.

---

## Directory Structure & File Descriptions

### Root-Level Files

| File | Type | Description |
|------|------|-------------|
| [`22n_NMOS_ON.net`](file:///home/ravi-sahu/Documents/5_sem/DVD/Project/22n_NMOS_ON.net) | SPICE Netlist | A sample NGSPICE netlist for simulating a **single NMOS transistor in the ON state** at 22nm technology. It includes the 22nm MGK model file, defines minimum channel length and width (Lmin = Wmin = 22nm), sets up drain, gate, source, and body voltage sources, and sweeps VDD from 0 to 1.2V while performing a DC temperature sweep (30°C to 90°C). The simulation prints terminal voltages and currents (Id, Ig, Is, Ib). This serves as a **template** for creating similar netlists for NMOS OFF, PMOS ON, and PMOS OFF cases. |
| [`DVD_ML_PROJECT.pdf`](file:///home/ravi-sahu/Documents/5_sem/DVD/Project/DVD_ML_PROJECT.pdf) | PDF Document | The **main project specification** document from the Digital VLSI Design course. It defines the complete workflow in three parts — (A) Transistor model training, (B) Gate-level validation, (C) Test-circuit validation — along with parameter ranges, deliverables, gate topologies, and accuracy expectations. |
| [`Project-1_leakage_estimation_annotated.pdf`](file:///home/ravi-sahu/Documents/5_sem/DVD/Project/Project-1_leakage_estimation_annotated.pdf) | PDF Document | An **annotated reference/tutorial** explaining the theory behind leakage current estimation in CMOS circuits. It covers leakage current setup in single MOSFETs, sample database generation for 4 MOS cases, stack structures with intermediate node voltages, and worked examples for INV, NAND2, AND2, and complex cells (e.g., AO31). It illustrates sub-threshold leakage, gate leakage, and body leakage estimation equations. |



### `models/` Directory — PTM Model Files (Set 2)

A second set of PTM BSIM4 model files, including **16nm technology node** and **Low Power (LP) variants** not present in the `model/` directory. This is the set referenced in the project assignment for team-specific model selection.

| File | Technology | Process Type | Description |
|------|-----------|-------------|-------------|
| [`22nm_MGK.pm`](file:///home/ravi-sahu/Documents/5_sem/DVD/Project/models/22nm_MGK.pm) | 22nm | Metal Gate / High-K | Standard process at 22nm. |

---

## Key Technical Details

### SPICE Model Parameters (from `.pm` files)
The PTM model files use **BSIM4v4.0 (level 54)** and contain parameters critical for leakage estimation:
- **`vth0`** — Threshold voltage (determines sub-threshold leakage)
- **`toxe`** — Electrical oxide thickness (determines gate leakage)
- **`ndep`** — Channel doping concentration (affects threshold voltage)
- **`u0`** — Low-field mobility
- **`xj`** — Junction depth

These five process parameters are varied ±10% from their nominal values during data generation to model process variations.

### SPICE Netlist Structure (from `.net` file)
- Uses `.INCLUDE` to reference a specific PTM model file
- Sets `.OPTIONS GMIN=1e-020 ABSTOL=1e-18` for accurate leakage current measurement (currents can be in the femtoampere range)
- Defines a single MOSFET with independent voltage sources on all four terminals
- Performs DC sweeps across voltage and temperature to generate I-V data

### Four Transistor Cases
| Case | Gate Voltage | Body Voltage | Purpose |
|------|-------------|-------------|---------|
| NMOS ON | VDD | 0 | Conduction current through ON NMOS |
| NMOS OFF | 0 | 0 | Leakage current through OFF NMOS |
| PMOS ON | 0 | VDD | Conduction current through ON PMOS |
| PMOS OFF | VDD | VDD | Leakage current through OFF PMOS |
