# Integrated Encoding to Enhance Quantum Convolutional Neural Networks



**Authors:**  
- Daniele Lizzio Bosco $^{\star, \bullet, \diamond}$
- Beatrice Portelli  $^{\bullet, \diamond}$
- Giuseppe Serra  $^{\bullet}$


$\star$ corresponding author [lizziobosco.daniele@spes.uniud.it]

$\bullet$ Department of Mathematics, Computer Science and Physics, University of Udine, Udine, Italy

$\diamond$ Department of Biology, University of Naples Federico II, Napoli, Italy

**Link to the paper:**  
[arXiv link]

## Table of Contents
1. [Background](#background)
2. [Reproducing the Experiments](#reproducing-the-experiments)
3. [Code Highlights](#code-highlights)
4. [Results](#results)


## Background
[A section providing context about the research. Discuss the problem being addressed, the motivation for the work, and any relevant prior work or key concepts.]

TO DO

## Reproducing the Experiments
1. **Requirements:**  
    This project is based on Python ```3.10.12```.
      <details>
    <summary>Requirements list</summary>
    
    ```yaml
    matplotlib==3.9.2
    numpy==2.1.0
    pandas==2.2.2
    Pillow==10.4.0
    PyYAML==6.0.1
    PyYAML==6.0.2
    qiskit==1.1.0
    qiskit_aer==0.14.1
    qiskit_aer_gpu==0.14.1
    qiskit_ibm_runtime==0.24.0
    scipy==1.14.1
    skimage==0.0
    torch==2.3.0
    torchvision==0.18.0
    tqdm==4.66.4

    ```

    </details>
2. **Setup Instructions:**  
    To set up the environment and install necessary dependencies:

   ```sh
   git clone https://github.com/Dan-LB/integrated_encoding_for_QuanvNN.git
   cd integrated_encoding_for_QuanvNN 
   pip install -r requirements.txt
   ```

3. **Main results:**
    The following steps can be performed to obtain the main results - classification accuracy of the proposed integrated encoding, compared to rotational encoding and classical CNN, corresponding to Table 3 and Table 4. 

    By default, the scripts perform the experiments on the MiraBest dataset. To use the LArTPC dataset instead, switch ```SELECTED_TASK = "MiraBest"``` to ```SELECTED_TASK = "LArTPC"``` in the considered files.

    

    1. Execute ```main.py```. This script checks all the models in the ```configs``` folder and runs the experiment for 10 different seeds.
    2. Execute ```process_results.py```.
    3. The results will be stored in the folder ```results```.

4. **Other results:**
    - Fig. 3 can be obtained from the notebook ```Compute_quantization_error.ipynb```.
    - Fig. 4 can be obtained from the notebook ```Compute_quantization_reduction.ipynb```.
    - The procedure to produce Fig. 7 is described in the notebook ```Compute_expressibility.ipynb```.

## Code Highlights

* #### ```Quanvolutional_Layer.py```
    This file contains the class ```QuanvolutionalLayer``` as a Torch.nn module. 
    Once initialized, the layer generates the circuits to be used as filters.
* #### ```circuit_builder.py```
    Contains the functions to construct the circuits, with both the standard approach and the proposed one.
* #### ```model_builder.py```
    Contains the function to construct a Quanvolutional model by constructing first a classical CNN, and then stacking a Quanvolutional layer on it with the function ```quanv_model = stack_quanv_on_top(quanv_layer, classical_model)```. 
* #### ```main.py```
    This script is used to perform all the experiments with 10 different seeds for each model present in the ```config``` folder.
* #### ```configs```
    The ```configs``` folder contains all the configuration used in this work.

    Example of $\text{QNN-Int-Simple}$:
    <details>
    <summary><i>QNN-Int-Simple-k3.yaml</i></summary>
    
    ```yaml

    encoding: INTEGRATED
    model:
    conv1:
        in_channels: 1
        kernel_size: 3
        out_channels: 16
        padding: 0
    dropout_conv_rate: 0.2
    dropout_fc_rate: 0.2
    fc1:
        out_features: 32
    fc2:
        out_features: 2
    input_shape:
    - 1
    - 30
    - 30
    quanv:
    L: 18
    activation: Full
    kernel_size: 3
    n_qubits: 4
    n_shots: 1000
    ```

    </details>
    


## Results
[Tables + some numbers and some comments?]


