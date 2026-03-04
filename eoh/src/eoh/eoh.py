
import random
import numpy as np

from .utils import createFolders
from .methods import methods
from .problems import problems

# main class for AEL
class EVOL:

    # initilization
    def __init__(self, paras, prob=None, **kwargs):

        print("----------------------------------------- ")
        print("---              Start EoH            ---")
        print("-----------------------------------------")
        # Create folder #
        createFolders.create_folders(paras.exp_output_path)
        print("- output folder created -")

        self.paras = paras

        print("-  parameters loaded -")

        self.prob = prob

        # Set deterministic seeds for reproducibility across runs.
        self.random_seed = getattr(paras, "exp_random_seed", 2024)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        
    # run methods
    def run(self):

        problemGenerator = problems.Probs(self.paras)

        problem = problemGenerator.get_problem()

        methodGenerator = methods.Methods(self.paras,problem)

        method = methodGenerator.get_method()

        method.run()

        print("> End of Evolution! ")
        print("----------------------------------------- ")
        print("---     EoH successfully finished !   ---")
        print("-----------------------------------------")
