"""
Quantum Galton Board Circuit implementations.

"""
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister

def galton_board_circuit_gaussian(n_layers):
    """
    Construct a Gaussian-weighted quantum Galton board circuit.

    This function simulates a Galton board using a Gaussian distribution
    by applying Hadamard gates, controlled swaps (Fredkin gates), and CX
    gates to move a 'ball' qubit through layers of pegs.

    Args:
        n_layers (int): Number of layers (rows of pegs) in the Galton board.

    Returns:
        QuantumCircuit: The constructed quantum circuit with measurements
        on odd-numbered qubits representing the bins.
    
    """
    n_qubits = 2 * (n_layers + 1)
    midpoint = int((n_qubits + 1) / 2)
    qc = QuantumCircuit(n_qubits)
    qc.x(midpoint)

    #first layer always the same
    qc.h(0)
    qc.cswap(0, midpoint - 1, midpoint)
    qc.cx(midpoint, 0)
    qc.cswap(0, midpoint, midpoint + 1)
    #between layers
    if n_layers != 1 or n_layers != 2 :
        for i in range(2,n_layers+1):
            qc.reset(0)
            qc.h(0)
            #first two fredkins and cnot
            qc.cswap(0, midpoint - i, midpoint - i + 1)
            qc.cx(midpoint - i + 1, 0)
            qc.cswap(0, midpoint - i + 1, midpoint - i + 2)
            qc.cx(midpoint - i + 2, 0)

            #iterate from back to mid
            lower_cnot_2 = midpoint - i + 2
            upper_cnot = midpoint + i - 1
            while upper_cnot != lower_cnot_2:
                qc.cswap(0, upper_cnot, upper_cnot + 1)
                qc.cx(upper_cnot, 0)
                upper_cnot -= 1
            qc.cswap(0, upper_cnot, upper_cnot + 1)

    qc.barrier()
    odd_qubits = [i for i in range(n_qubits) if i % 2 == 1]
    c_odd = ClassicalRegister(len(odd_qubits))
    qc.add_register(c_odd)
    for i, q in enumerate(odd_qubits):
        qc.measure(q, c_odd[i])
    return qc

def galton_board_circuit_fine_grained(n_layers, theta):
    """
    Enables bias within the Galton board by shifting the probability all of
    the pegs by a constant theta using RX gates.

    Args:
        n_layers (int): Number of layers (rows of pegs) in the Galton board.
        theta (float): Rotation angle (in radians) for RX gates at each peg.

    Returns:
        QuantumCircuit: The constructed quantum circuit with measurements
        on odd-numbered qubits (representing bins).
    """
    n_qubits = 2 * (n_layers + 1)
    midpoint = int((n_qubits + 1) / 2)
    qc = QuantumCircuit(n_qubits)
    qc.x(midpoint)

    for layer in range(1,n_layers + 1):
        for peg_number in range(1, layer + 1):
            qc.reset(0)
            qc.rx(theta, 0)
            qc.cswap(0, midpoint - layer + (2 * peg_number) - 2,
                    midpoint - layer + (2 * peg_number) - 1)
            qc.cx(midpoint - layer + (2 * peg_number) - 1, 0)
            qc.cswap(0, midpoint - layer + (2 * peg_number) - 1,
                    midpoint - layer + (2 * peg_number))
        qc.barrier()
        for peg_number in range(1, layer):
            qc.cx(midpoint - layer + 1 + (2 * peg_number), midpoint - layer + (2 * peg_number))   
            qc.reset(midpoint - layer + 1 + (2 * peg_number))

    qc.barrier()
    odd_qubits = [i for i in range(n_qubits) if i % 2 == 1]
    c_odd = ClassicalRegister(len(odd_qubits))
    qc.add_register(c_odd)
    for i, q in enumerate(odd_qubits):
        qc.measure(q, c_odd[i])
    return qc

def theta(peg_no, decay_constant):
    """
    Calculates the RX rotation angle for exponential decay probability.

    Args:
        peg_no (int): Peg number in the current layer.
        decay_constant (float): Decay rate controlling probability drop-off.

    Returns:
        float: Rotation angle in radians.
    """
    return 2 * np.arcsin(np.exp(- decay_constant * peg_no))

def galton_board_exponential(n_layers, decay_constant):
    """
    Construct a fine-grained exponential-decay quantum Galton board.
    With a tunable decay constant which represents the expoenetial decay per peg.

    Args:
        n_layers (int): Number of layers in the Galton board.
        decay_constant (float): Decay constant controlling peg rotation.

    Returns:
        QuantumCircuit: Circuit with exponential bias, measuring odd-numbered qubits.
    """
    n_qubits = 2 * (n_layers + 1)
    midpoint = int((n_qubits + 1) / 2)
    qc = QuantumCircuit(n_qubits)
    qc.x(midpoint)

    for layer in range(1,n_layers + 1):
        for peg_number in range(1, layer + 1):
            qc.reset(0)
            qc.rx(theta(midpoint + peg_number - layer,decay_constant), 0)
            qc.cswap(0, midpoint - layer + (2 * peg_number) - 2,
                    midpoint - layer + (2 * peg_number) - 1)
            qc.cx(midpoint - layer + (2 * peg_number) - 1, 0)
            qc.cswap(0, midpoint - layer + (2 * peg_number) - 1,
                    midpoint - layer + (2 * peg_number))
        qc.barrier()
        for peg_number in range(1, layer):
            qc.cx(midpoint - layer + 1 + (2 * peg_number), midpoint - layer + (2 * peg_number))   
            qc.reset(midpoint - layer + 1 + (2 * peg_number))

    qc.barrier()
    odd_qubits = [i for i in range(n_qubits) if i % 2 == 1]
    c_odd = ClassicalRegister(len(odd_qubits))
    qc.add_register(c_odd)
    for i, q in enumerate(odd_qubits):
        qc.measure(q, c_odd[i])
    return qc

def galton_board_hadamard_walk(n_layers):
    """
    Construct a Hadamard-walk quantum Galton board.

    This method uses a Hadamard gate as a coin flip to produce
    a balanced probability distribution at each layer. The 
    coin is not reset to allow interference within the board.

    Args:
        n_layers (int): Number of layers in the Galton board.

    Returns:
        QuantumCircuit: Circuit implementing the Hadamard random walk.
    """
    n_data_qubits = (2 * n_layers) + 1
    midpoint = int((n_data_qubits + 1) / 2)

    qc = QuantumCircuit(n_data_qubits + 1,)

    qc.x(midpoint)

    for layer in range(1, n_layers + 1):
        for i in range(0,layer):
            qc.h(0)
            qc.cswap(0, midpoint - layer + (2*i) + 1, midpoint - layer + (2*i) + 2)
            qc.cx(midpoint - layer + (2*i) + 1, 0)
            qc.cswap(0, midpoint - layer + (2*i) + 1, midpoint - layer + (2*i))

        qc.barrier()

    # Measurement on odd-positioned data qubits only
    odd_qubits = [i for i in range(1, n_data_qubits + 1) if i % 2 == 1]
    c_odd = ClassicalRegister(len(odd_qubits))
    qc.add_register(c_odd)

    for i, q in enumerate(odd_qubits):
        qc.measure(q, c_odd[i])

    return qc
