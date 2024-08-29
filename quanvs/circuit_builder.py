import torch

#from qiskit import QuantumCircuit, Aer, transpile

from qiskit.circuit import QuantumCircuit, Parameter
import random
import math
import numpy as np
import constants

from qiskit import transpile
from qiskit_aer import AerSimulator


def generate_circuit_integrated(encoding_config, kernel_size): 
    n_qubits = encoding_config["n_qubits"]
    L = encoding_config["L"]
    activation = encoding_config["activation"]

    circuit = QuantumCircuit(n_qubits)
    data = [Parameter(f"p{i}") for i in range(kernel_size**2)]
    for par in data:
        circuit.rx(par*0, 0)

    gates = []
    while len(gates) < L:
        q1 = random.randint(0, n_qubits-1)
        q2 = random.randint(0, n_qubits-1)
        if q1 != q2:
            beta = random.random()
            #select random data in kernel_size**2
            if len(gates) < kernel_size**2:
                index = len(gates)
            else:
                index = random.randint(0, kernel_size**2-1)
            #select random generator in I, X, Y, Z


            g1 = "I"
            g2 = "I"
            while(g1 == "I" and g2 == "I"):
                g1 = random.choice(["I", "X", "Y", "Z"])
                g2 = random.choice(["I", "X", "Y", "Z"])

            gates.append({"q1": q1, "q2": q2, "g1" : g1, "g2": g2, "beta": beta, "index": index})

    random.shuffle(gates)
    for gate in gates:
        q1 = gate["q1"]
        q2 = gate["q2"]
        g1 = gate["g1"]
        g2 = gate["g2"]

        if activation == constants.QuanvActivation.FULL.value:
            param = data[gate["index"]]*beta*2*math.pi
        elif activation == constants.QuanvActivation.HALF.value:
            param = data[gate["index"]]*beta*math.pi
        elif activation == constants.QuanvActivation.SHIFTED.value:
            param = data[gate["index"]]*beta*math.pi + math.pi/2
        elif activation == constants.QuanvActivation.RANDOM.value:
            param = data[gate["index"]]*beta*math.pi + random.random()*math.pi
        elif activation == constants.QuanvActivation.FIXED.value:
            param = data[gate["index"]]*math.pi
        else:
            raise ValueError("Activation not supported")

        beta = gate["beta"]
        index = gate["index"]

        if g1 == g2:
            if g1 == "X":
                circuit.rxx(param, q1, q2)
            elif g1 == "Y":
                circuit.ryy(param, q1, q2)
            elif g1 == "Z":
                circuit.rzz(param, q1, q2)
        elif g1 == "I":
            if g2 == "X":
                circuit.rx(param, q2)
            elif g2 == "Y":
                circuit.ry(param, q2)
            elif g2 == "Z":
                circuit.rz(param, q2)
        elif g2 == "I":
            if g1 == "X":
                circuit.rx(param, q1)
            elif g1 == "Y":
                circuit.ry(param, q1)
            elif g1 == "Z":
                circuit.rz(param, q1)
        elif g1 == "Z" and g2 == "X":
            circuit.rzx(param, q1, q2)
        elif g1 == "X" and g2 == "Z":
            circuit.rzx(param, q2, q1)

        #gates such as rxy, and ryz are not standard in qiskit, so they are implemented as rxx and rzz
        elif g1 == "X" and g2 == "Y": 
            circuit.rzz(math.pi/4, q1, q2)
            circuit.rxx(param, q1, q2)
            circuit.rzz(-math.pi/4, q1, q2)
        elif g1 == "Y" and g2 == "X":
            circuit.rzz(math.pi/4, q2, q1)
            circuit.rxx(param, q2, q1)
            circuit.rzz(-math.pi/4, q2, q1)
        elif g1 == "Y" and g2 == "Z":
            circuit.rzz(math.pi/4, q2, q1)
            circuit.ryy(param, q2, q1)
            circuit.rzz(-math.pi/4, q2, q1)
        elif g1 == "Z" and g2 == "Y":
            circuit.rzz(math.pi/4, q1, q2)
            circuit.ryy(param, q2, q1)
            circuit.rzz(-math.pi/4, q1, q2)
    return circuit


def generate_circuit_rotational(encoding_config):
        connection_prob = encoding_config["probability"]
        n_qubits = encoding_config["n_qubits"]
        activation = encoding_config["activation"]
        one_qb_list = ["X", "Y", "Z",  "P", "T", "H"]
        two_qb_list = ["Cnot", "Swap", "SqrtSwap"]
        gate_list = []

        for i in range(n_qubits):
            for j in range(n_qubits):
                if i != j and random.random() < connection_prob:
                    g_index = random.randint(0, len(two_qb_list)-1)
                    gate_list.append({"gate": two_qb_list[g_index], "first_q": i, "second_q": j})

        n_one_qg =   random.randint(0, 2*n_qubits)   
        for i in range(n_one_qg):
            q = random.randint(0, n_qubits-1) 
            g_index = random.randint(0, len(one_qb_list)-1)  
            gate_list.append({"gate": one_qb_list[g_index], "first_q": q})
        random.shuffle(gate_list)


        circuit = QuantumCircuit(n_qubits)
        
        data = [Parameter(f"p{i}") for i in range(n_qubits)]
        if activation == constants.QuanvActivation.FULL.value:
            for q in range(n_qubits):
                circuit.rx(data[q]*2*math.pi, q)
        elif activation == constants.QuanvActivation.HALF.value:
            for q in range(n_qubits):
                circuit.rx(data[q]*math.pi, q)
        else:
            raise ValueError("Activation not supported")

        for gate in gate_list:
            theta = random.random()*math.pi 
            if gate["gate"] == "Cnot":
                circuit.cx(gate["first_q"], gate["second_q"])
            elif gate["gate"] == "Swap":
                circuit.swap(gate["first_q"], gate["second_q"])
            elif gate["gate"] == "SqrtSwap":
                # Implementing SQRTSWAP using Qiskit's standard gates
                circuit.rxx(math.pi / 2, gate["first_q"], gate["second_q"])
                circuit.ryy(math.pi / 2, gate["first_q"], gate["second_q"])
            elif gate["gate"] == "RX":
                circuit.rx(theta, gate["first_q"])
            elif gate["gate"] == "RY":
                circuit.ry(theta, gate["first_q"])
            elif gate["gate"] == "RZ":
                circuit.rz(theta, gate["first_q"])
            elif gate["gate"] == "P":
                circuit.p(math.pi / 2, gate["first_q"])  # Phase gate
            elif gate["gate"] == "T":
                circuit.t(gate["first_q"])
            elif gate["gate"] == "H":
                circuit.h(gate["first_q"])
        return circuit




