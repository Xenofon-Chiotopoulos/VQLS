# Variational Quantum Linear System Solver (VQLS) Repository

## Overview

This repository contains Python code for a Variational Quantum Linear System Solver (VQLS). The VQLS algorithm is designed to find a quantum state vector that approximates the solution to a linear system of equations by optimizing variational parameters in a quantum circuit.

## Motivation

This repository was created to facilitate the seamless usage and implementation of the Variational Quantum Linear Solver (VQLS). Building upon the principles outlined in the original tutorial at [https://pennylane.ai/qml/demos/tutorial_vqls/](https://pennylane.ai/qml/demos/tutorial_vqls/), our adaptation extends its capabilities to handle general matrices and ansatz circuits. For a more comprehensive exploration of the foundational concepts, the original paper is available [here](https://quantum-journal.org/papers/q-2023-11-22-1188/) Hello sir.



### Documentation

For detailed documentation, please refer to the [official documentation](vqls.readthedocs.io).

## Files

### `vqls_utils.py`

This file contains utility functions and definitions for the VQLS algorithm. Key functions include:

- `make_CA`: Constructs the controlled unitary component of the problem matrix.
- `construct_matrix`: Constructs the matrix representation of the problem using Pauli matrices.
- `calc_c_probs`: Calculates classical probabilities based on the linear system solution.
- `process_vqls_output`: Processes the output of the VQLS algorithm, calculating classical and quantum probabilities.
- `run_vqls`: Runs the VQLS algorithm to optimize variational parameters.
- `plot_vqls_results`: Plots the results of the VQLS algorithm, including classical and quantum probabilities.
- `count_variational_gates`: Counts the number of variational gates in a given ansatz circuit.
- `custom_variational_block`: Implements a custom variational block using provided ansatz information.
- `convert_to_pauli_string`: Converts a `PauliSentence` object to a list of Pauli strings and corresponding coefficients.
- `load_from_json` and `save_to_json`: Utility functions to load and save data from/to JSON files.

### `examples.py`

This file provides an example of how to use the VQLS algorithm with a specific linear system. It demonstrates the steps to set up the problem, define an ansatz circuit, run the VQLS algorithm, and visualize the results.

## Usage

### Clone this repository

```shell
git clone https://github.com/Xenofon-Chiotopoulos/VQLS.git
```

### Set up virtual environment

> ⚠️ We support Python 3.11 and above. While older versions might work, we do not 
> guarantee a bug free experience.

```shell
python -m venv venv
venv\Scripts\activate  # Windows
. venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```
