# Importing necessary libraries
from pennylane import numpy as np
from src.vqls_utils import construct_matrix, plot_vqls_results, run_vqls, process_vqls_output

# Define coefficients and Pauli strings for the linear system
c = np.array([3.0, -0.5, -0.5]) 
pauli_string = [["I", "I", "I"], ["X", "I", "I"], ["X", "Z", "Z"]]

# Determine the number of qubits in the system
n_qubits = len(pauli_string[0])

# Set the number of quantum measurements
n_shots = 10 ** 6

# Define the variational ansatz circuit
ansatz = [
    {"gate": "H", "wires": [0, 1, 2]},
    {"gate": "RY", "wires": [0, 1, 2]},
    {"gate": "CNOT", "wires": [[0, 1], [1, 2]]},
]

# Construct the matrix representation of the linear system
A_num = construct_matrix(c, pauli_string)

# Define the target vector
b = np.ones(2 ** len(pauli_string[0])) 

# Run the VQLS algorithm to optimize variational parameters
w, cost_history = run_vqls(n_qubits, pauli_string, c, ansatz, rng_seed=0, steps=201, eta=0.8, q_delta=0.5, print_step=True)

# Calculate classical and quantum probabilities
c_probs, q_probs = process_vqls_output(n_qubits, w, ansatz, A_num, b, n_shots, print_output=True)

# Plot and visualize the results of the VQLS algorithm
plot_vqls_results(n_qubits, c_probs, q_probs, cost_history, file_name="quantum_probabilities_1")
