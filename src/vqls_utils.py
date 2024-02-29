import json
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from pennylane.pauli import PauliSentence
from typing import List, Any, Tuple

class VQLS:
    def __init__(self, c: list[float], pauli_string: list[list[str]], b: list[float], n_shots: int) -> None:
        self.c = c
        self.pauli_string = pauli_string
        self.b = b
        self.n_shots = n_shots
        self.rng_seed = 0
        self.steps = 251
        self.eta = 0.8
        self.q_delta = 0.5
        self.print_step = False
        self.print_output = False
        self.A = self.construct_matrix()
        self.n_qubits = len(pauli_string[0])
        

    def make_CA(self, idx: int, ancilla_idx: int) -> None:
        """Constructs the controlled unitary component A_l of the problem matrix A.

        Args:
            idx (int): Index of the Pauli string in the Pauli string list.
            ancilla_idx (int): Index of the ancilla qubit.
            pauli_string (list): List of Pauli strings.

        """
        # Iterate over the Pauli string and apply the appropriate controlled operations
        for id in range(len(self.pauli_string)):
            if id == idx:
                sub_pauli = self.pauli_string[id]
                for gate_index ,gate in enumerate(sub_pauli):
                    # Apply Identity, CNOT or CZ based on the Pauli symbol
                    if gate == "I":
                        continue  
                    elif gate == "X":
                        qml.CNOT(wires=[ancilla_idx, gate_index])
                    elif gate == "Y":
                        qml.CY(wires=[ancilla_idx, gate_index])
                    elif gate == "Z":
                        qml.CZ(wires=[ancilla_idx, gate_index])

    def construct_matrix(self) -> np.tensor:
        """
        Construct a matrix given coefficients and Pauli strings.

        Parameters:
        - c (array): Coefficients for each Pauli term.
        - pauli_strings (list): List of tuples representing Pauli strings.

        Returns:
        - np.array: Constructed matrix.
        """
        # Define Pauli matrices
        Id = np.identity(2,dtype='complex128')
        Z = np.array([[1, 0], [0, -1]],dtype='complex128')
        X = np.array([[0, 1], [1, 0]],dtype='complex128')
        Y = np.array([[0,complex(0,-1)], [complex(0,1), 0]])

        # Construct the matrix using vectorized operations
        matrix = 0
        for pauli_string, coeff in zip(self.pauli_string, self.c):
            pauli_term = 0
            for gate_index, gate in enumerate(pauli_string):
                if gate_index == 0:
                    if gate == "I":
                        pauli_term = Id
                    elif gate == "X":
                        pauli_term = X
                    elif gate == "Y":
                        pauli_term = Y
                    elif gate == "Z":
                        pauli_term = Z
                else:
                    if gate == "I":
                        pauli_term = np.kron(pauli_term, Id)
                    elif gate == "X":
                        pauli_term = np.kron(pauli_term, X)
                    elif gate == "Y":
                        pauli_term = np.kron(pauli_term, Y)
                    elif gate == "Z":
                        pauli_term = np.kron(pauli_term, Z)
            matrix += coeff * pauli_term
        return matrix

    def calc_c_probs(self) -> np.tensor:#A_num: np.tensor, b: np.tensor) -> np.tensor:
        """
        Calculates the vector `x` that satisfies the equation `Ax = b`.

        Args:
            A (pennylane.numpy.tensor.tensor): tensor representation of the matrix `A`.
            b (numpy.ndarray): NumPy array representation of the vector `b`.

        Returns:
            pennylane.numpy.tensor.tensor: Calculated vector `x`.
        """
        A_inv = np.linalg.inv(self.A)
        x = np.dot(A_inv, self.b)
        c_probs = (x / np.linalg.norm(x)) ** 2
        return c_probs

    def process_vqls_output(self, w: np.tensor, ansatz: list[dict], quantum_circuit=None) -> np.tensor:
        """
        Process the output of a variational quantum linear system algorithm (VQLS).

        Args:
            n_qubits (int): Number of qubits.
            w (pennylane.numpy.tensor.tensor): Coefficients of the Pauli operators.
            ansatz (list[dict]): Ansatz circuit description.
            A (pennylane.numpy.tensor.tensor): Pauli operators matrix.
            b (numpy.array): Vector of coefficients for each Pauli term.
            n_shots (int): Number of shots for the quantum simulation.
            print: (bool, optional): Whether to print the calculated probabilities. Defaults to False.

        Returns:
            tuple[pennylane.numpy.tensor.tensor, pennylane.numpy.tensor.tensor]: Calculated classical and quantum probabilities.
        """
        dev_x = qml.device("lightning.qubit", wires=self.n_qubits, shots=self.n_shots)

        @qml.qnode(dev_x, interface="autograd")
        def prepare_and_sample(weights):
            self.custom_variational_block(weights, ansatz, quantum_circuit=quantum_circuit)
            return qml.sample()

        c_probs = self.calc_c_probs()

        raw_samples = prepare_and_sample(w)

        samples = []
        for sam in raw_samples:
            samples.append(int("".join(str(bs) for bs in sam), base=2))

        for qubit in range(2**self.n_qubits): # fixes bincount issue
            if qubit not in np.unique(samples):
                samples.append(qubit)

        q_probs = np.bincount(samples) / self.n_shots

        c_probs = 1/np.max(c_probs)*c_probs
        q_probs = 1/np.max(q_probs)*q_probs

        if self.print_output == True:
            print("x_n^2 =\n", c_probs)
            print("|<x|n>|^2=\n", q_probs)
            print('Difference=\n', c_probs-q_probs)

        return c_probs, q_probs

    def count_variational_gates(self, ansatz: list[dict], all_gates = False) -> int:
        """
        Counts the number of variational gates in the provided ansatz circuit.

        Args:
            ansatz (list[dict]): List of gate information dictionaries.

        Returns:
            int: Total number of variational gates in the ansatz.
        """
        variational_gate_count = 0
        non_variational_gate_count = 0
        for gate_info in ansatz:
            if gate_info["gate"] == "RX" or gate_info["gate"] == "RY" or gate_info["gate"] == "RZ" or gate_info["gate"] == "P":
                variational_gate_count += len(gate_info["wires"])

            if gate_info["gate"] == "CNOT" or gate_info["gate"] == "H" :
                non_variational_gate_count += len(gate_info["wires"])

        if all_gates == True:
            return variational_gate_count, non_variational_gate_count
        else:
            return variational_gate_count

    def custom_variational_block(self, weights: np.tensor, quantum_circuit, ansatz) -> None:
        """
        Implements a custom variational block using the provided ansatz.

        Args:
            weights: pennylane.numpy.tensor.tensor representing the weights of the variational parameters.
            ansatz: List of gate information dictionaries describing the variational circuit.

        Returns:
            None

        """
        if quantum_circuit is None:
            # We first prepare an equal superposition of all the states of the computational basis.
            for idx in range(self.n_qubits):
                qml.Hadamard(wires=idx)
            # Let's define our guess with a minimal variational circuit.
            # A single rotation qubit is added on each qubit on which angles are0 provided len(angles)<n_qubits
            for idx, element in enumerate(weights):
                qml.RY(element, wires=idx)
            qml.CNOT(wires=[1, 0])
            qml.CNOT(wires=[2, 3])

        else:

            def circuit(parameters):
                parameters = parameters
                i = 0
                for instr, qubits, clbits in quantum_circuit.data:
                    name = instr.name.lower()

                    if name == "rx":
                        if ansatz == 'all':
                            qml.RX(instr.params[0], wires=qubits[0].index)
                        else:
                            qml.RX(parameters[i], wires=qubits[0].index)
                            i += 1
                    elif name == "ry":
                        if ansatz == 'all':
                            qml.RY(instr.params[0], wires=qubits[0].index)
                        else:
                            qml.RY(parameters[i], wires=qubits[0].index)
                            i += 1
                    elif name == "rz":
                        if ansatz == 'all':
                            qml.RZ(parameters[i], wires=qubits[0].index)
                        else:
                            qml.RZ(parameters[i], wires=qubits[0].index)
                            i += 1
                    elif name == "h":
                        qml.Hadamard(wires=qubits[0].index)
                    elif name == "cx":
                        qml.CNOT(wires=[qubits[0].index, qubits[1].index])
                return qml

            return circuit(parameters=weights)
                

    def run_vqls(self, quantum_circuit, ansatz) -> np.tensor:
        tot_qubits = self.n_qubits + 1  
        ancilla_idx = self.n_qubits 

        def U_b():
            """Unitary matrix rotating the ground state to the problem vector |b> = U_b |0>."""
            for idx in range(self.n_qubits):
                qml.Hadamard(wires=idx)

        def CA(idx,params=None):
            self.make_CA(idx,ancilla_idx)

        dev_mu = qml.device("lightning.qubit", wires=tot_qubits)

        @qml.qnode(dev_mu, interface="autograd")
        def local_hadamard_test(weights, l=None, lp=None, j=None, part=None , params=None):

            qml.Hadamard(wires=ancilla_idx)

            if part == "Im" or part == "im":
                qml.PhaseShift(-np.pi / 2, wires=ancilla_idx)

            self.custom_variational_block(weights, ansatz, quantum_circuit=quantum_circuit)

            CA(l,params)

            U_b()

            if j != -1:
                qml.CZ(wires=[ancilla_idx, j])

            U_b()

            CA(lp,params)

            qml.Hadamard(wires=ancilla_idx)

            # Expectation value of Z for the ancillary qubit.
            return qml.expval(qml.PauliZ(wires=ancilla_idx))

        def mu(weights, l=None, lp=None, j=None, params=None):
            """Generates the coefficients to compute the "local" cost function C_L."""

            mu_real = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Re", params=params)
            mu_imag = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Im", params=params)

            return mu_real + 1.0j * mu_imag

        def psi_norm(weights, params=None):
            """Returns the normalization constant <psi|psi>, where |psi> = A |x>."""
            norm = 0.0

            for l in range(0, len(self.c)):
                for lp in range(0, len(self.c)):
                    norm = norm + self.c[l] * np.conj(self.c[lp]) * mu(weights, l, lp, -1,params)

            return abs(norm)

        def cost_loc(weights,params=None):
            """Local version of the cost function. Tends to zero when A|x> is proportional to |b>."""
            mu_sum = 0.0

            for l in range(0, len(self.c)):
                for lp in range(0, len(self.c)):
                    for j in range(0, self.n_qubits):
                        mu_sum = mu_sum + self.c[l] * np.conj(self.c[lp]) * mu(weights, l, lp, j,params)

            mu_sum = abs(mu_sum)

            # Cost function C_L
            return 0.5 - 0.5 * mu_sum / (self.n_qubits * psi_norm(weights))

        return cost_loc
        
    def costFunc(self, weights, quantum_circuit=None, ansatz=''):
        cost = self.run_vqls(quantum_circuit, ansatz=ansatz)
        return cost(weights)

    def getReward(self, weights, quantum_circuit=None, ansatz=''):
        return np.exp(-10*self.costFunc(weights, quantum_circuit, ansatz))

    def gradient_descent(self, quantum_circuit):
        opt = qml.AdamOptimizer()
        parameters = get_parameters(quantum_circuit)
        theta = np.array(parameters, requires_grad=True)

        # store the values of the cost function

        def prova(params):
            return self.costFunc(params=params, quantum_circuit=quantum_circuit, ansatz='')

        cost = [prova(theta)]

        # store the values of the circuit parameter
        angle = [theta]

        conv_tol = 1e-08  # default -06

        for n in range(self.steps):
            theta, prev_energy = opt.step_and_cost(prova, theta)
            cost.append(prova(theta))
            angle.append(theta)

            conv = np.abs(cost[-1] - prev_energy)

            if n % 2 == 0:
                print(f"Step = {n},  Cost = {cost[-1]:.8f}")

            if conv <= conv_tol:
                print('Landscape is flat')
                break
        return cost

    def plot_vqls_results(self, c_probs: list[float], q_probs: list[float], cost_history: list[np.tensor], file_name: str = "quantum_probabilities") -> None:
        """
        Plots the results of a variational quantum linear system algorithm (VQLS) using classical and quantum probabilities.

        Args:
            n_qubits (int): Number of qubits in the system.
            c_probs (list[float]): List of classical probabilities.
            q_probs (list[float]): List of quantum probabilities.
            cost_history (list[pennylane.numpy.tensor.tensor]): List of cost function values during optimization.

        Returns:
            None
        """
        plt.style.use("ggplot")  
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

        # Plot 1: Classical probabilities
        ax1.bar(np.arange(0, 2 ** self.n_qubits), c_probs, color="blue")
        ax1.set_xlim(-0.5, 2 ** self.n_qubits - 0.5)
        ax1.set_xlabel("Vector space basis")
        ax1.set_title("Classical probabilities")

        # Plot 2: Quantum probabilities
        ax2.bar(np.arange(0, 2 ** self.n_qubits), q_probs, color="green")
        ax2.set_xlim(-0.5, 2 ** self.n_qubits - 0.5)
        ax2.set_xlabel("Hilbert space basis")
        ax2.set_title("Quantum probabilities")

        # Plot 3: Cost function optimization
        ax3.plot(cost_history, color='green', marker='o', linestyle='-', linewidth=2, markersize=8, label='Cost Function')
        ax3.set_title("Optimization Progress")
        ax3.set_xlabel("Optimization Steps")
        ax3.set_ylabel("Cost Function Value")

        plt.tight_layout()
        if file_name != "quantum_probabilities":
            plt.savefig(f"{file_name}.png")
        elif file_name == "quantum_probabilities": 
            plt.show()

    def filter_comb(combinations_: List[List[int]]) -> List[List[int]]:
        """
        Filters combinations to include only those with unique elements.

        Args:
            cnot_combinations (List[List[int]]): List of combinations.

        Returns:
            List[List[int]]: Filtered list of combinations.
        """
        combinations = []
        for cnot_list in combinations_:
            if len(cnot_list) == 1:
                # Single-element combinations are always included
                combinations.append(cnot_list)
            if len(cnot_list) == 2:
                # Two-element combinations are included if the elements are distinct
                if cnot_list[0] != cnot_list[1]:
                    combinations.append(cnot_list)
            if len(cnot_list) == 3:
                # Three-element combinations are included if all elements are distinct
                if cnot_list[0] != cnot_list[1] and cnot_list[0] != cnot_list[2] and cnot_list[1] != cnot_list[2]:
                    combinations.append(cnot_list)
        return combinations

def load_from_json(file_path: str) -> Any:
    """
    Load data from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Any: Loaded data from the JSON file.
    """
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def save_to_json(data: Any, file_path: str) -> None:
    """
    Save data to a JSON file.

    Args:
        data (Any): Data to be saved to the JSON file.
        file_path (str): Path to the JSON file.

    Returns:
        None
    """
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)

def convert_to_pauli_string(pauli_sentence: PauliSentence) -> Tuple[List[List[str]], List[float]]:
    """
    Convert a PauliSentence object to a list of Pauli strings and corresponding coefficients.

    Args:
        pauli_sentence (PauliSentence): Input PauliSentence object.

    Returns:
        Tuple[List[List[str]], List[float]]: Tuple containing the list of Pauli strings and coefficients.
    """
    pauli_string = []
    coeffs_list = []
    for term, coefficient in pauli_sentence.items():
        pauli_term = []
        for qubit in range(len(pauli_sentence.wires)):
            op = term[qubit]
            if op == "I":
                pauli_term.append("I")
            elif op == "X":
                pauli_term.append("X")
            elif op == "Y":
                pauli_term.append("Y")
            elif op == "Z":
                pauli_term.append("Z")
        pauli_string.append(pauli_term)
        coeffs_list.append(coefficient)
    return pauli_string, coeffs_list

def get_parameters(quantum_circuit):
    parameters = []
    # Iterate over all gates in the circuit
    for instr, qargs, cargs in quantum_circuit.data:

        # Extract parameters from gate instructions
        if len(instr.params) > 0:
            parameters.append(instr.params[0])
    return parameters