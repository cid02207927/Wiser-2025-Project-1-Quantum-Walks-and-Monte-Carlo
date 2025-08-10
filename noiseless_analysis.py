"""
Quantum Galton Board Circuit Visualisation & Simulations

"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from circuit_distributions import (
    galton_board_circuit_gaussian,
    galton_board_circuit_fine_grained,
    galton_board_exponential,
    galton_board_hadamard_walk,
)

def qc_visualise(n_layers):
    """
    Visualizes a Gaussian-weighted quantum Galton board circuit.

    Constructs a Gaussian distribution Galton board using quantum gates
    and renders the circuit diagram.

    Args:
        n_layers (int): Number of layers (rows of pegs) in the Galton board.

    Returns:
        None: Displays the quantum circuit.
    """
    qc = galton_board_circuit_gaussian(n_layers)
    qc.draw('mpl', fold=-1)
    plt.show()

def qc_gaussian_histogram(n_layers, shots=10000):
    """
    Simulates and plots the output histogram of a Gaussian-weighted Galton board.

    Runs the Gaussian circuit on the Aer simulator and produces a histogram
    of measurement outcomes.

    Args:
        n_layers (int): Number of layers in the Galton board.

    Returns:
        None: Displays a histogram plot of simulation results.
    """
    qc = galton_board_circuit_gaussian(n_layers)
    simulator = AerSimulator()
    circ = transpile(qc, simulator)
    result = simulator.run(circ, shots=shots).result()
    counts = result.get_counts(circ)
    plot_histogram(counts, title='Bell-State counts')
    plt.show()

def normal_distribution(x, mu, sig, amp):
    return amp * np.exp(- 0.5 * ((x -mu) / sig) ** 2) / (np.sqrt(np.pi * 2))


def chi_2_gaussian_fit(n_layers, shots=10000):
    """
    Does a chi sqiared goodness of fit test to see if the output of the 
    gaussian galton board fits a normal distribution.
    Args:
        n_layers (int): Number of layers in the quantum Galton board.

    Returns:
        None
    """
    qc = galton_board_circuit_gaussian(n_layers)
    simulator = AerSimulator()
    circ = transpile(qc, simulator)
    result = simulator.run(circ, shots=shots).result()
    counts = result.get_counts(circ)

    sorted_bins = sorted(counts.keys(), key=lambda x: int(x, 2))
    sorted_counts = [counts[bin_str] for bin_str in sorted_bins]

    bins_int = []
    for i in range(0,n_layers + 1):
        bins_int.append(2*i)

    xdata = np.array(bins_int)
    ydata = np.array(sorted_counts)
    amp_guess = np.max(ydata)
    mu_guess = np.sum(xdata * ydata) / np.sum(ydata)
    sigma_guess = 1

    popt, pcov = curve_fit(normal_distribution, xdata, ydata, p0=[mu_guess, sigma_guess, amp_guess])
    mu_fit, sig_fit, amp_fit = popt

    expected = normal_distribution(xdata, mu_fit, sig_fit, amp_fit)
    chi2_stat = np.sum((ydata - expected) ** 2 / expected)
    dof = len(xdata) - len(popt)
    p_value = chi2.sf(chi2_stat, dof)

    plt.bar(xdata, ydata, width=1.0, alpha=0.6, label='Observed counts')
    xfit = np.linspace(min(xdata), max(xdata), 200)
    yfit = normal_distribution(xfit, mu_fit, sig_fit, amp_fit)
    plt.plot(xfit, yfit, 'r-', label='Fitted Normal Distribution')
    plt.xlabel('Bin index (integer)')
    plt.ylabel('Counts')
    plt.title('Counts vs Bin index with Normal Fit')
    plt.legend()
    plt.show()

    print(f"Chi-square statistic: {chi2_stat:.3f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p_value:.4f}")
    print(f"Fitted mu: {mu_fit:.3f}, sigma: {sig_fit:.3f}, amplitude: {amp_fit:.3f}")

def qc_exp_visualise(n_layers, decay_constant = 1):
    """
    Visualizes an exponential-decay fine-grained quantum Galton board circuit.

    Args:
        n_layers (int): Number of layers in the Galton board.
        decay_constant (float): Controls the steepness of exponential decay.

    Returns:
        None: Displays the quantum circuit.
    """
    qc = galton_board_exponential(n_layers, decay_constant)
    qc.draw('mpl', fold=-1)
    plt.show()

def qc_exponential_histogram(n_layers,decay_constant = 1, shots=10000):
    """
    Simulates and plots the histogram of an exponential-decay fine-grained Galton board.

    Args:
        n_layers (int): Number of layers in the Galton board.
        decay_constant (float): Controls the steepness of exponential decay.

    Returns:
        None: Displays a histogram plot of simulation results.
    """
    qc = galton_board_exponential(n_layers, decay_constant)
    simulator = AerSimulator()
    circ = transpile(qc, simulator)
    result = simulator.run(circ, shots=shots).result()
    counts = result.get_counts(circ)
    plot_histogram(counts, title='Bell-State counts')
    plt.show()

def qc_finegrain_histogram(n_layers, theta, shots=10000):
    """
    Simulates and plots the histogram of a fine-grained adjustable-theta Galton board.

    Args:
        n_layers (int): Number of layers in the Galton board.
        theta (float): Rotation angle applied to gates for controlling distribution.

    Returns:
        None: Displays a histogram plot of simulation results.
    """
    qc = galton_board_circuit_fine_grained(n_layers, theta)
    simulator = AerSimulator()
    circ = transpile(qc, simulator)
    result = simulator.run(circ, shots=shots).result()
    counts = result.get_counts(circ)
    plot_histogram(counts, title='Bell-State counts')
    plt.show()

def qc_hadamard_walk_hist(n_layers, shots=10000):
    """
    Simulates and plots the histogram of a Hadamard quantum walk circuit.

    Args:
        n_layers (int): Number of layers (steps) in the quantum walk.

    Returns:
        None: Displays a histogram plot of simulation results.
    """
    qc = galton_board_hadamard_walk(n_layers)
    sim = AerSimulator()
    tcirc = transpile(qc, sim)
    result = sim.run(tcirc, shots=shots).result()
    counts = result.get_counts()

    plot_histogram(counts)
    plt.show()

def qc_hadamard_walk_visualise(n_layers, shots=10000):
    """
    Visualizes a Hadamard quantum walk circuit.

    Args:
        n_layers (int): Number of layers (steps) in the quantum walk.

    Returns:
        None: Displays a matplotlib rendering of the quantum circuit.
    """
    qc = galton_board_hadamard_walk(n_layers)
    qc.draw('mpl', fold=-1)
    plt.show()

if __name__ == "__main__":

    #Example usage of each function uncomment to test
    #By default the numbers used to create images on presentation

    # Gaussian
    # qc_visualise(4)
    # qc_gaussian_histogram(4)

    # chi_2_gaussian_fit(7)

    # Exponential
    # qc_exp_visualise(3)
    # qc_exponential_histogram(5, 1/5)

    # Fine-grained
    # qc_finegrain_histogram(4, 2 * np.pi / 3)

    # Hadamard walk
    # qc_hadamard_walk_hist(4)
    # qc_hadamard_walk_visualise(4)
    pass
