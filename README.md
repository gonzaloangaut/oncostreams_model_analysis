# Oncostreams Model Analysis

This repository contains the numerical validation and systematic analysis of the computational model developed to study the emergence of collective migration in glioma cells.

The simulation model is implemented separately in the [`oncostreams`](https://github.com/gonzaloangaut/oncostreams) repository. This repository focuses on the analysis required to define, validate, and characterize the model.

## Repository structure

### `preliminary/`

Analyses used to define the model and justify numerical and modeling choices before studying its physical behavior.

Current studies include:

- `model/`: model definition, nondimensionalization, and parameter summary.
- `neighbors/`: definition and justification of the cell-neighbor criterion.
- `forces/`: force regularization and contraction-overlap safety criterion.
- `delta_t/`: numerical convergence with respect to the integration time step.

The goal of this stage is to obtain a well-defined and numerically validated set of model parameters.

### `scripts/processing/`

Scripts used to process raw simulation outputs into analysis-ready datasets.

## Analysis roadmap

After the preliminary validation is completed, the model will be studied through:

1. a state diagram in density and interaction strength, $(\rho,\kappa)$;
2. the response to translational and rotational noise;
3. a high-resolution study of selected transitions, including finite-size effects when appropriate.

These analyses will be added progressively as the numerical protocol is finalized.

## Data

Raw simulation outputs and processed datasets are not versioned in this repository because of their size.

Local analysis directories such as `processed/` and `ovito_final/` are excluded through `.gitignore`.

## Status

The project is currently in the numerical-validation stage.
