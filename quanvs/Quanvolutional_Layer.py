import torch
import time

import math
import constants

from qiskit.circuit import QuantumCircuit
from qiskit_ibm_runtime import SamplerV2
from qiskit_aer import AerSimulator

from quanvs.quanv_util import quantize_patch
import quanvs.quanv_util as utils
import quanvs.circuit_builder as circuit_builder

import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from constants import QuantumDevice

#from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

class QuanvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, 
                 quantization = 2, 
                 encoding_approach = None,
                 encoding_config = None,
                 simulator = QuantumDevice.NOISELESS,
                 verbose = False):
        super(QuanvolutionalLayer, self).__init__()

        if simulator == QuantumDevice.NOISELESS:
            self.simulator = AerSimulator(method='statevector')
            self.need_to_transpile = False
        else:
            raise Exception(f"Simulator '{simulator}' not recognized.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        assert padding % 0.5 == 0, "Padding must be a multiple of 0.5."

        self.int_padding = int(padding)
        self.partial_padding = self.padding - self.int_padding


        self.verbose = verbose

        self.look_up = {}
        self.quantization = quantization
        #initialize counters for duplicated patches as a dictionary.
        self.counters = {"images": 0, "total_patches": 0, "unique_patches": 0}

        utils.check_existing_encoding(encoding_approach)
        if encoding_config is None:
            encoding_config = utils.default_encoding_config(encoding_approach)
        utils.check_config_integrity(encoding_approach, encoding_config, kernel_size, verbose)

        self.encoding_approach = encoding_approach
        self.encoding_config = encoding_config

        start_time = time.time()
        self.circuits = self.generate_layer()

        end_time = time.time()
        if verbose:
            print(f"Time required to generate circuit layer: {end_time - start_time:.4f} seconds")


    def forward(self, x):

        # if x is not a tensor, convert it to a tensor
        if not isinstance(x, torch.Tensor):
            print("Alert: Input is not a tensor. Converting to tensor.")
            x = torch.tensor(x)

        self.device = x.device

        if len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 3:
            x = x.unsqueeze(0)


        # Calculate output dimensions
        batch_size, _, height, width = x.shape 
        padded_height = int(height + 2 * self.padding)
        padded_width = int(width + 2 * self.padding)
        out_height = (padded_height - self.kernel_size) // self.stride + 1
        out_width = (padded_width - self.kernel_size) // self.stride + 1

        # Create output tensor
        output = torch.zeros((batch_size, self.out_channels, out_height, out_width)).to(self.device)

        # Perform convolution
        start_time = time.time()
        for i in range(batch_size):
            image_start_time = time.time()
            self.counters["images"] += 1

            padded_x = torch.nn.functional.pad(x[i], (self.int_padding, self.int_padding, self.int_padding, self.int_padding))
            if self.partial_padding > 0:
                padded_x = torch.nn.functional.pad(padded_x, (1, 0, 1, 0))

            for j in range(self.out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # Apply padding to input tensor
                        patch = padded_x[:, h_start:h_end, w_start:w_end].cpu().numpy()

                        patch = patch[0]
                        patch = quantize_patch(patch, self.quantization)
                        value =  self.look_up.get((j, tuple(patch.reshape(-1))))
                        if j == 0:
                            self.counters["total_patches"] += 1
                        if value == None:
                            value =  self.to_quanvolute_patch(self.circuits[j], patch) 
                            self.look_up[(j, tuple(patch.reshape(-1)))] = value
                            if j == 0:
                                self.counters["unique_patches"] += 1
                            
                        output[i, j, h, w] = value

            image_end_time = time.time()
            image_time = image_end_time - image_start_time
            elapsed_time = image_end_time - start_time
            average_time_per_image = elapsed_time / (i + 1)
            estimated_remaining_time = average_time_per_image * (batch_size - i - 1)
            
            if self.verbose == True and i%5==4:
                print(f"Time for image {i+1}: {image_time:.2f} seconds")
                print(f"Estimated remaining time: {estimated_remaining_time/60:.2f} minutes\n")

                #print all the info in self.counters
                print(self.counters)


        return output
    
    def single_forward(self, x, circuit_index):
        batch_size, _, height, width = x.shape 
        padded_height = height + 2 * self.padding
        padded_width = width + 2 * self.padding
        out_height = (padded_height - self.kernel_size) // self.stride + 1
        out_width = (padded_width - self.kernel_size) // self.stride + 1
        output = torch.zeros((batch_size, 1, out_height, out_width)).to(self.device)

        for i in range(batch_size):
            for j in [circuit_index]:
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # Apply padding to input tensor
                        padded_x = torch.nn.functional.pad(x[i], (self.padding, self.padding, self.padding, self.padding))
                        patch = padded_x[:, h_start:h_end, w_start:w_end].cpu().numpy()

                        patch = patch[0]
                        patch = quantize_patch(patch, self.quantization)
                        value =  self.look_up.get((j, tuple(patch.reshape(-1))))
                        if value == None:
                            value =  self.to_quanvolute_patch(self.circuits[j], patch) 
                            self.look_up[(j, tuple(patch.reshape(-1)))] = value             
                        output[i, 0, h, w] = value
        return output

    def generate_filter(self):

        """

        """

        if self.encoding_approach == constants.CircuitEncoding.INTEGRATED:
            return circuit_builder.generate_circuit_integrated(self.encoding_config, self.kernel_size)

        elif self.encoding_approach == constants.CircuitEncoding.ROTATIONAL:
            return circuit_builder.generate_circuit_rotational(self.encoding_config)
        
        elif self.encoding_approach == constants.CircuitEncoding.PASS:
            return 0
        else:
            raise Exception(f"Encoding approach '{self.encoding_approach}' not recognized.")

    def generate_layer(self):
        layer = []
        for _ in range(self.out_channels):
            filter = self.generate_filter()
            layer.append(filter)
        return layer

    def print_counters(self):
        print(self.counters)

    def to_quanvolute_patch(self, circuit, patch):

        if self.encoding_approach == constants.CircuitEncoding.INTEGRATED or self.encoding_approach == constants.CircuitEncoding.ROTATIONAL:
            self.activation = self.encoding_config["activation"]
            n = self.kernel_size
            for i in range(n*n):
                row = i // n
                col = i % n
                circuit = circuit.assign_parameters({f"p{i}": patch[row][col]})
                circuit.measure_all()

            sampler = SamplerV2(mode=self.simulator)
            sampler.options.default_shots = self.encoding_config["n_shots"]
            result = sampler.run([circuit]).result()

            counts = result[0].data.meas.get_counts()

            sum_1s = sum(key.count('1') * count for key, count in counts.items()) / self.encoding_config["n_shots"] / (n*n)

            return sum_1s
        
        elif self.encoding_approach == constants.CircuitEncoding.PASS:
            return 0
        else:
            raise Exception(f"Encoding approach '{self.encoding_approach}' not recognized.")
        

    
    
  