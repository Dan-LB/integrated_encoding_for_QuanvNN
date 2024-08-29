import torch

import math

from qiskit.circuit import QuantumCircuit, Parameter


from quanvs.quanv_util import default_encoding_config

import torch.nn as nn


import numpy as np

from qiskit.quantum_info import Statevector

from tqdm import tqdm


def random_unitary(N):
    """
        Return a Haar distributed random unitary from U(N)
    """

    Z = np.random.randn(N, N) + 1.0j * np.random.randn(N, N)
    [Q, R] = np.linalg.qr(Z)
    D = np.diag(np.diagonal(R) / np.abs(np.diagonal(R)))
    return np.dot(Q, D)

def haar_integral(num_qubits, samples):
    """
        Return calculation of Haar Integral for a specified number of samples.
    """

    N = 2**num_qubits
    randunit_density = np.zeros((N, N), dtype=complex)

    
    zero_state = np.zeros(N, dtype=complex)
    zero_state[0] = 1
    
    for _ in tqdm(range(samples)):
        A = np.matmul(zero_state, random_unitary(N)).reshape(-1, 1)
        randunit_density += np.kron(A, A.conj().T)

    randunit_density /= samples

    return randunit_density


def pqc_integral(num_qubits, circuit, samples, n_params=4):
    """
        Return calculation of Integral for a PQC over the uniformly sampled 
        the parameters θ for the specified number of samples.
    """

    N = 2**num_qubits
    randunit_density = np.zeros((N, N), dtype=complex)

    for _ in range(samples):
        params = np.random.uniform(0, 1, n_params)

        circuit_ass = circuit.assign_parameters({f"p{i}": params[i] for i in range(n_params)})
        circuit_ass.remove_final_measurements()
        U = Statevector.from_instruction(circuit_ass).data
        randunit_density += np.kron(U.reshape(-1, 1), U.conj().T.reshape(1, -1))

    return randunit_density / samples

def get_fidelity(psi, phi):
    """
    Compute the fidelity between two pure quantum states.
    """
    return np.abs(np.vdot(psi, phi)) ** 2

def compute_avg_fidelity(num_qubits, circuit, samples, n_params=4):
    """
        Return calculation of Integral for a PQC over the uniformly sampled 
        the parameters θ for the specified number of samples.
    """

    N = 2**num_qubits
    fidelities = []

    for _ in range(samples):
        params1 = np.random.uniform(0, 1, n_params)
        params2 = np.random.uniform(0, 1, n_params)

        c1 = circuit.assign_parameters({f"p{i}": params1[i] for i in range(n_params)})
        c2 = circuit.assign_parameters({f"p{i}": params2[i] for i in range(n_params)})

        c1.remove_final_measurements()
        c2.remove_final_measurements()

        U1 = Statevector.from_instruction(c1).data
        U2 = Statevector.from_instruction(c2).data

        fidelities.append(get_fidelity(U1, U2))

    return fidelities


