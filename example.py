from pennylane import numpy as np
from src.vqls_utils import construct_matrix, plot_vqls_results, run_vqls, process_vqls_output

c = np.array([3.0, -0.5, -0.5]) 
pauli_string=[["I","I","I"],["X","I","I"],["X","Z","Z"]]

n_qubits = len(pauli_string[0]) # Number of system qubits.
n_shots = 10 ** 6  # Number of quantum measurements.

ansatz = [
    {"gate": "H", "wires": [0,1,2]},
    {"gate": "RY", "wires": [0,1,2]},
    {"gate": "CNOT", "wires": [[0,1], [1,2]]},
]

params = []
A_num = construct_matrix(c,pauli_string)
b = np.ones(2**len(pauli_string[0])) 

w, cost_history = run_vqls(n_qubits, pauli_string, c, ansatz, rng_seed = 0, steps = 201, eta = 0.8 ,q_delta = 0.5, print_step = True)

c_probs, q_probs = process_vqls_output(n_qubits, w, ansatz, A_num, b, n_shots, print_output = True)

results_list = [w, cost_history, c_probs, q_probs, ansatz]

plot_vqls_results(n_qubits, c_probs, q_probs, cost_history, file_name="quantum_probabilities")