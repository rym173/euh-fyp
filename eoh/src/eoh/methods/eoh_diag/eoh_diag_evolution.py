import re
from ...llm.interface_LLM import InterfaceLLM


class EvolutionDiag:
    def __init__(self, api_endpoint, api_key, model_LLM, llm_use_local, llm_local_url, debug_mode, prompts, **kwargs):
        self.prompt_task = prompts.get_task()
        self.prompt_func_name = prompts.get_func_name()
        self.prompt_func_inputs = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf = prompts.get_inout_inf()
        self.prompt_other_inf = prompts.get_other_inf()

        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"

        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode

        self.interface_llm = InterfaceLLM(
            self.api_endpoint,
            self.api_key,
            self.model_LLM,
            llm_use_local,
            llm_local_url,
            self.debug_mode,
        )

    def _build_prompt(self, header, extra):
        return (
            self.prompt_task
            + "\n"
            + header
            + "\n"
            + "First, describe your new algorithm and main steps in one sentence. "
            + "The description must be inside a brace. Next, implement it in Python as a function named "
            + self.prompt_func_name
            + ". This function should accept "
            + str(len(self.prompt_func_inputs))
            + " input(s): "
            + self.joined_inputs
            + ". The function should return "
            + str(len(self.prompt_func_outputs))
            + " output(s): "
            + self.joined_outputs
            + ". "
            + self.prompt_inout_inf
            + " "
            + self.prompt_other_inf
            + "\n"
            + "Do not give additional explanations."
        )

    def get_prompt_i1(self):
        return self._build_prompt("", "")

    def get_prompt_e1(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv += (
                "No."
                + str(i + 1)
                + " algorithm and the corresponding code are: \n"
                + indivs[i]["algorithm"]
                + "\n"
                + indivs[i]["code"]
                + "\n"
            )
        header = (
            "I have "
            + str(len(indivs))
            + " existing algorithms with their codes as follows: \n"
            + prompt_indiv
            + "Please help me create a new algorithm that has a totally different form from the given ones."
        )
        return self._build_prompt(header, "")

    def get_prompt_e2(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv += (
                "No."
                + str(i + 1)
                + " algorithm and the corresponding code are: \n"
                + indivs[i]["algorithm"]
                + "\n"
                + indivs[i]["code"]
                + "\n"
            )
        header = (
            "I have "
            + str(len(indivs))
            + " existing algorithms with their codes as follows: \n"
            + prompt_indiv
            + "Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them. "
            + "Firstly, identify the common backbone idea in the provided algorithms. Secondly, based on the backbone idea describe your new algorithm in one sentence."
        )
        return self._build_prompt(header, "")

    def get_prompt_m1(self, indiv1):
        header = (
            "I have one algorithm with its code as follows. "
            + "Algorithm description: "
            + indiv1["algorithm"]
            + "\nCode:\n"
            + indiv1["code"]
            + "\n"
            + "Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided."
        )
        return self._build_prompt(header, "")

    def get_prompt_m2(self, indiv1):
        header = (
            "I have one algorithm with its code as follows. "
            + "Algorithm description: "
            + indiv1["algorithm"]
            + "\nCode:\n"
            + indiv1["code"]
            + "\n"
            + "Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided."
        )
        return self._build_prompt(header, "")

    def _extract_algorithm(self, text):
        alg = re.findall(r"\{(.*)\}", text, re.DOTALL)
        if alg:
            return alg[0].strip()
        if "python" in text:
            alg = re.findall(r"^.*?(?=python)", text, re.DOTALL)
        elif "import" in text:
            alg = re.findall(r"^.*?(?=import)", text, re.DOTALL)
        else:
            alg = re.findall(r"^.*?(?=def)", text, re.DOTALL)
        return alg[0].strip() if alg else ""

    def _extract_code(self, text):
        fence = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
        if fence:
            return fence[0].strip()
        code = re.findall(r"import.*return", text, re.DOTALL)
        if not code:
            code = re.findall(r"def.*return", text, re.DOTALL)
        return code[0].strip() if code else ""

    def _get_alg(self, prompt_content):
        response = self.interface_llm.get_response(prompt_content)
        algorithm = self._extract_algorithm(response)
        code = self._extract_code(response)

        n_retry = 1
        while (not algorithm or not code):
            if self.debug_mode:
                print("Error: algorithm or code not identified, retrying ...")
            response = self.interface_llm.get_response(prompt_content)
            algorithm = self._extract_algorithm(response)
            code = self._extract_code(response)
            if n_retry > 3:
                break
            n_retry += 1

        if not code:
            raise ValueError("Failed to extract code from LLM response.")

        if "import numpy as np" not in code:
            code = "import numpy as np\n" + code

        if "def score" not in code:
            m = re.search(r"def\s+(\w+)\s*\(", code)
            if m:
                code = re.sub(r"def\s+\w+\s*\(", "def score(", code, count=1)

        return [code, algorithm]

    def i1(self):
        prompt_content = self.get_prompt_i1()
        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()
        return self._get_alg(prompt_content)

    def e1(self, parents):
        prompt_content = self.get_prompt_e1(parents)
        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()
        return self._get_alg(prompt_content)

    def e2(self, parents):
        prompt_content = self.get_prompt_e2(parents)
        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()
        return self._get_alg(prompt_content)

    def m1(self, parent):
        prompt_content = self.get_prompt_m1(parent)
        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()
        return self._get_alg(prompt_content)

    def m2(self, parent):
        prompt_content = self.get_prompt_m2(parent)
        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()
        return self._get_alg(prompt_content)
