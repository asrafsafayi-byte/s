
# Desmond Molecular Dynamics Module

## Overview
Desmond is a high-performance molecular dynamics (MD) program designed for efficient simulation of biomolecular systems. It utilizes the GROMOS force field and OPLS4.

## Command Line Usage
To run a simulation, use the `desmond` command with specific switches:
```bash
desmond -c input.cfg -o output.dae -cpu 48
```

## Configuration Parameters
Key parameters in the `.cfg` file:
- FORCE_FIELD OPLS4
- INTEGRATOR reversible_reference_propagator
- TIMESTEP 0.002
- TEMPERATURE 300.0
- PRESSURE 1.01325
- ENSEMBLE NPT

## Advanced Switches
-use_gpus true
-gpu_ids 0,1,2,3
-checkpoint_interval 10.0
-max_sim_time 100.0
                