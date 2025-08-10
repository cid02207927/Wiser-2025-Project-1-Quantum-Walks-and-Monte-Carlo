# Wiser-2025-Project-1-Quantum-Walks-and-Monte-Carlo
## Team Name
zeta

## Team Members (solo project)
Muhammad Nauman gst-4u0NZJaROvxcug3

## Project Summary 

This project aimed to  simulate a quantum galton board of n layers to see if we  could match classical boards. Then further manipulate the gates to achieve an exponential distribution as well as simulate a Hadamard quantum walk. The method used was to understand and generalise the technique used in the Universal Statistical Simulator paper which contained balanced galton boards of  1 and 2 layers. 
This implementation consists of two Python modules one with separate functions consisting of each distrubutions circuit and the other with the analysis and visualisation. 

'circuit_distributions.py'
Contains of the contructions of each circuit type.
Gaussian uses hadamard gates to be symettrical.
Fine_grained introduces bias from RX rotations by a fixed angle on all pegs.
Exponetial includes different biases at each peg to create an exponential decay pattern.
Hadamard_walk simulates a hadamard walk by keeping the interference between each collision.

'noiseless_analysis.py'
Contains the circuit diagram visualisations for all distributions.
Simulations using Qiskits AerSimulator with adjustable layers, biases and shot numbers for all of the circuits.
Histogram plots of all of the output distributions.
A chi^2 goodness of fit test to show how closely the Gaussian circuit matches and ideal normalised distribution, with parameter fitting.

Approach
Each board was modelled as a layered system of pegs, with the ball represented by a flipped qubit.
The gates were then used to give each peg probabilities to move the ball around. It was important to correctly map out the circuit so that the probabilities are correct.

Findings
Gaussian circuits correctly produced normal distributions with chi test idicating a good fit for many layers.
Exponential decay bias was correctly implemented and shows that desired distributions can be found by changing the bias on the pegs.
Hadamand walk circuit correctly gave the bimodal distribution we were seeking.

This project can be imporved by using noisy backends to simulate realworld uses of the circuits. And further optimising the gate counts to reduce circuit depth.

## Project Presentation Deck

[Click here to view the presentation deck](./Project%20presentation%20-%20Muhammad%20Nauman.pdf)
