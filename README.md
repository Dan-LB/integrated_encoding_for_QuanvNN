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

## Reproducing the Experiments
1. **Requirements:**  
    WIP
   [List the dependencies, libraries, and hardware/software requirements.]

2. **Setup Instructions:**  
    WIP
   [Step-by-step guide on how to set up the environment and install necessary dependencies.]

   Example:
   ```sh
   git clone https://github.com/username/repo.git
   cd repo
   pip install -r requirements.txt

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

* #### Quanvolutional_Layer
    This file contains the class ```QuanvolutionalLayer``` as a Torch.nn module. 


## Results
[Tables + some numbers and some comments?]


