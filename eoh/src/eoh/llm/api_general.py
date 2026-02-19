import http.client
import json
from urllib.parse import urlparse


class InterfaceAPI:
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.n_trial = 5

    def get_response(self, prompt_content):
        payload_explanation = json.dumps(
            {
                "model": self.model_LLM,
                "messages": [
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_content}
                ],
            }
        )

        headers = {
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json",
            "x-api2d-no-cache": 1,
        }
        
        response = None
        n_trial = 1
        while True:
            n_trial += 1
            if n_trial > self.n_trial:
                return response
            try:
                endpoint = self.api_endpoint
                if endpoint.startswith("http://") or endpoint.startswith("https://"):
                    parsed = urlparse(endpoint)
                    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
                    conn = conn_cls(parsed.netloc)
                    base_path = parsed.path.rstrip("/")
                    path = f"{base_path}/v1/chat/completions" if base_path else "/v1/chat/completions"
                else:
                    conn = http.client.HTTPSConnection(endpoint)
                    path = "/v1/chat/completions"
                conn.request("POST", path, payload_explanation, headers)
                res = conn.getresponse()
                data = res.read()
                json_data = json.loads(data)
                response = json_data["choices"][0]["message"]["content"]
                break
            except:
                if self.debug_mode:
                    print("Error in API. Restarting the process...")
                continue
            

        return response
