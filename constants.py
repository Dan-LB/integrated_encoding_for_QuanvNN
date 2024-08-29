from enum import Enum

class Dataset(Enum):
    LArTPC_e = "e-"
    LArTPC_gamma = "gamma"
    LArTPC_K = "K+"
    LArTPC_mu = "mu+"
    LArTPC_p = "p+"
    LArTPC_pi = "pi+"
    LArTPC_pi0 = "pi0"

class CircuitEncoding(Enum):
    ROTATIONAL= "Rotational"
    INTEGRATED = "Integrated"
    PASS = "Pass"

class QuanvActivation(Enum):
    FULL = "Full"
    HALF = "Half"
    SHIFTED = "Shifted"
    RANDOM = "Random"
    FIXED = "Fixed"


class Task(Enum):
    MIRABEST = "MiraBest"
    LArTPC = "LArTPC"

class QuantumDevice(Enum):
    NOISELESS = "Noiseless"

    


    

