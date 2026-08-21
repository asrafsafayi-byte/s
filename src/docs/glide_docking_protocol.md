
# Glide Molecular Docking

## Docking Protocols
Glide performs ligand docking using XP (Extra Precision) and SP (Standard Precision) modes.

## Execution
glide -HOST localhost:20 -WAIT -JOBNAME dock_run_01

## Input Parameters
LIGAND_FILE ligands.mae
RECEPTOR_FILE protein.mae
GRID_FILE grid.zip
DOCKING_MODE XP
SAMPLE_LIGANDS 1000

## Scoring
Emodel weight 1.0
Gscore weight 0.5
                