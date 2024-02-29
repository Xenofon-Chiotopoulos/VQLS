# Importing necessary libraries
from pennylane import numpy as np
from src.vqls_utils import VQLS, load_from_json
import time
start = time.time()
# Define coefficients and Pauli strings for the linear system
#c = np.array([3.0, -0.5, -0.5]) 
#pauli_string = [["I", "I", "I"], ["X", "I", "I"], ["X", "Z", "Z"]]

pauli_string = load_from_json('pauli_string.json')
c = load_from_json('coeffs_list.json')
print(c,pauli_string)

# Define the target vector
b = np.ones(2 ** len(pauli_string[0])) 

# Set the number of quantum measurements
n_shots = 10 ** 6
ansatz = [{"gate": "H", "wires": [0,1,2,3]},]

vqls_instance = VQLS(c, pauli_string, b, n_shots)
vqls_instance.steps = 10
vqls_instance.print_step = True

# Run the VQLS algorithm to optimize variational parameters
w, cost_history = vqls_instance.run_vqls(None, ansatz)

# Calculate classical and quantum probabilities
c_probs, q_probs = vqls_instance.process_vqls_output(w, ansatz)

# Plot and visualize the results of the VQLS algorithm
vqls_instance.plot_vqls_results(c_probs, q_probs, cost_history, file_name="quantum_probabilities_1")

print(time.time()-start)