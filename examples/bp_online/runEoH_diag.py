from eoh import eoh
from eoh.utils.getParas import Paras

# Parameter initilization #
paras = Paras()

# Set parameters #
paras.set_paras(method = "eoh_diag",    # ['ael','eoh','eoh_diag']
                problem = "bp_online", #['tsp_construct','bp_online']
                llm_api_endpoint = "http://127.0.0.1:4000",
                llm_api_key = "local",
                llm_model = "ollama/qwen2.5-coder:14b-instruct",
                ec_operators = ['e1','m1','m2'],
                ec_operator_weights = [1,1,1],
                ec_pop_size = 6, # number of samples in each population
                ec_n_pop = 6,  # number of populations
                exp_n_proc = 1,  # multi-core parallel
                eva_numba_decorator = False,
                exp_debug_mode = False)

# initilization
evolution = eoh.EVOL(paras)

# run
evolution.run()
