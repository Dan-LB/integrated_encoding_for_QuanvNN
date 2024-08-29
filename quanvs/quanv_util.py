import sys
sys.path.append(".") 

import constants
import math

def default_encoding_config(encoding_approach):
    """
    Returns the default encoding configuration based on the specified encoding approach.

    Parameters:
    - encoding_approach (str): The encoding approach to use. Possible values are 'ROTATIONAL', 'UGATE', or 'INTEGRATED'.

    Returns:
    - dict: A dictionary containing the default encoding configuration.

    Raises:
    - Exception: If the specified encoding approach is not recognized.
    """

    if encoding_approach == constants.CircuitEncoding.ROTATIONAL:
        return {
            "kernel_size": 3,
            "n_qubits": 9,
            "probability": 0.15,
            "n_shots": 1000,
            "activation": constants.QuanvActivation.HALF.value
        }

    elif encoding_approach == constants.CircuitEncoding.INTEGRATED:
        return {
            "kernel_size": 3,
            "n_qubits": 4,
            "L": 25,
            "activation": constants.QuanvActivation.FULL.value,
            "n_shots": 1000
        }

    elif encoding_approach == constants.CircuitEncoding.PASS:
        return {
            "info": "No encoding needed. This is a pass-through layer."
        }
    else:
        raise Exception(f"Encoding approach '{encoding_approach}' not recognized.")



def check_config_integrity(encoding_approach, encoding_config, kernel_size, verbose=False):  
        """
        Check the integrity of the configuration parameters for the given encoding approach.

        Parameters:
        - encoding_approach (str): The encoding approach to be used.
        - encoding_config (dict): The configuration parameters for the encoding approach.
        - kernel_size (int): The size of the kernel.
        - verbose (bool): Whether to print additional information.

        Raises:
        - Exception: If the integrity check fails.

        Returns:
        - None
        """

        if encoding_approach != constants.CircuitEncoding.PASS:
            n_qubits = encoding_config["n_qubits"]
        # for rotational, n_qubits must be kernel_size^2, while for ugate, n_qubits must be 3 and kernel_size must be 3
        if encoding_approach == constants.CircuitEncoding.ROTATIONAL:
            if n_qubits != kernel_size**2:
                raise Exception(f"n_qubits must be equal to kernel_size^2 = {kernel_size**2} for rotational encoding.")

        default_config = default_encoding_config(encoding_approach)
        # check if all the keys in the default config are present in the encoding config
        for key in default_config:
            if key not in encoding_config:
                raise Exception(f"Key '{key}' missing in encoding config for encoding approach '{encoding_approach}'.")
        
        #questo forse non serve
        for key in encoding_config:
            if key not in default_config:
                raise Exception(f"Key '{key}' not recognized in encoding config for encoding approach '{encoding_approach}'.")
        
        if verbose:
            #print the encoding approach, all the parameters inside the dictionary encoding_config, and the kernel_size
            print(f"Encoding approach: {encoding_approach}")
            print(f"Encoding config: {encoding_config}")
            print(f"Kernel size: {kernel_size}")
            print("Config integrity check passed.")

        return 

def check_existing_encoding(encoding_approach):
    #check if encoding approach is valid
    if encoding_approach not in constants.CircuitEncoding:
        raise Exception(f"Encoding approach '{encoding_approach}' not recognized.")

def quantize(value, levels):
    """
    Assuming the input value is in the range [0, 1], quantize it into the specified number of levels.
    """

    #check if the value is in the range [0, 1]
    if value < 0 or value > 1:
        raise Exception(f"Value {value} is not in the range [0, 1].")

    return min(1, math.floor(value*levels) / (levels-1))


def quantize_patch(patch, levels):
    if levels == None:
        return patch

    kernel_size = len(patch)
    for ii in range(kernel_size**2):
        row = ii // kernel_size
        col = ii % kernel_size
        patch[row][col] = quantize(patch[row][col], levels)
    return patch


def get_qc_characteristics(qc):
    depth = qc.depth()
    num_qubits = qc.num_qubits
    ops = {}
    for op in qc.data:
        gate_name = op.operation.name
        if gate_name not in ops:
            ops[gate_name] = 1
        else:
            ops[gate_name] += 1
    num_multi_qubit_gates = qc.num_nonlocal_gates()
    return {"depth": depth, "num_qubits": num_qubits, "ops": ops, "num_multi_qubit_gates": num_multi_qubit_gates}

def print_qc_characteristics(qc):
    characteristics = get_qc_characteristics(qc)
    print("Quantum circuit characteristics:")
    print(f"Depth: {characteristics['depth']}")
    print(f"Number of qubits: {characteristics['num_qubits']}")
    print("Operations:")
    for op in characteristics['ops']:
        print(f"{op}: {characteristics['ops'][op]}")
    print(f"Number of multi-qubit gates: {characteristics['num_multi_qubit_gates']}")
    return






