from eoh import eoh
from eoh.utils.getParas import Paras

# Parameter initilization #
paras = Paras()

# Set parameters #
paras.set_paras(method = "eoh_diag",    # ['ael','eoh','eoh_diag']
                problem = "bp_online", #['tsp_construct','bp_online']
                llm_api_endpoint = "vllm-nodeport.vllm-ns.svc.cluster.local:8000",
                llm_api_key = "EMPTY",  # works if your vLLM has no auth; otherwise put real key
                llm_model = "Qwen3.5-122B-A10B-FP8",
                ec_operators = ['e1','m1','m2'],
                ec_operator_weights = [1,1,1],
                ec_pop_size = 6, # number of samples in each population
                ec_n_pop = 6,  # number of populations
                exp_n_proc = 2,  # multi-core parallel
                eva_numba_decorator = False,
                exp_debug_mode = False)

# Override bp_online default evaluation timeout (seconds).
paras.eva_timeout = 90
print("paras.eva_timeout =", paras.eva_timeout)

# initilization
evolution = eoh.EVOL(paras)

# run
evolution.run()
