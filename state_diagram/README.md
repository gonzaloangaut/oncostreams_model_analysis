# State diagram

Initial study of the oncostreams model in the density–interaction-strength plane $(\rho, \widetilde\kappa)$, focusing on cell morphology, collective organization, and mobility.

## Simulation protocol

- $\rho \in \{0.60, 0.75, 0.85, 0.90, 0.98\}$; $\widetilde\kappa \in \{0.75, 1.50, 2.50, 4.00\}$.
- 64 independent realizations per point (20 points; 1280 simulations), with $N=1000$ cells.
- $\Delta t=0.01$, final time $T=2000$, periodic boundary conditions, and no additive noise.
- $AR_{\max}=2.7$, $\lambda_{\rm core}=1$, five deformation attempts, and contraction-overlap safety ratio $\xi_{\rm safe}=0.8$. Morphology is updated at every integration step.

## Data

- `processed/`: Parquet datasets for global and local order, motion, deformation, overlap, and clusters.
- `ovito_final/`: final configurations at $T=2000$ for all realizations.

These data directories are not tracked in Git. Complete raw outputs, intermediate OVITO configurations, and checkpoints are retained on the CCAD cluster. Processing scripts are in `../scripts/processing/`, and the exact scripts used for this campaign are archived under `processed/scripts_used/` locally.

## Analysis

- `00_data_validation.ipynb`: short reproducible integrity check.
- `01_temporal_analysis.ipynb`: relaxation and stationarity.
- `02_final_configurations.ipynb`: spatial organization at final time.
- `03_state_diagram.ipynb`: late-time observables and parameter-space maps.

The initial grid is exploratory; possible transition regions will require more focused simulations and, where justified, finite-size analysis.
