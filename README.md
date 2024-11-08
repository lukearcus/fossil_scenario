# Data-Driven Neural Certificates

We propose a technique for the sound and automated synthesis of probabilistic certificates. The tool makes use of teh scenario approach theory for verification of these certificates. This work builds upon the FOSSIL tool, which is described in this [tool paper](https://doi.org/10.1145/3447928.3456646), or can be downloaded [here](https://github.com/oxford-oxcav/fossil).

## Requirements

Install:

We suggest using conda for environment management, and provide an environment file for easy installation as:

> conda env create -f requirements.txt

## Running Experiments

Experiments (found in the experiments folder) may be run with the following syntax:

> python3 -m experiments.scenapp\_tests.benchmarks.barr\_1\_alt

Some useful options are listed below:

> --plot
> --record

## Exploring the Code

For the interested reader, the bulk of the problem formulation is found in fossil/certificates.py, this file contains the loss functions for all certificates as well as the learning loop used to train each certificate.

