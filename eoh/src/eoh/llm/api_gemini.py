import http.client
import json


class InterfaceGemini:
    def __init__(self, api_key, model_LLM, debug_mode):
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.n_trial = 5

    def get_response(self, prompt_content):
        payload = json.dumps(
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt_content}],
                    }
                ]
            }
        )

        headers = {
            "Content-Type": "application/json",
        }

        response = None
        n_trial = 1
        while True:
            n_trial += 1
            if n_trial > self.n_trial:
                return response
            try:
                conn = http.client.HTTPSConnection("generativelanguage.googleapis.com")
                path = f"/v1beta/models/{self.model_LLM}:generateContent?key={self.api_key}"
                conn.request("POST", path, payload, headers)
                res = conn.getresponse()
                data = res.read()
                if res.status != 200:
                    try:
                        print(f"Gemini API error {res.status}: {data.decode('utf-8')}")
                    except Exception:
                        print(f"Gemini API error {res.status}: <non-text body>")
                    continue
                json_data = json.loads(data)
                response = json_data["candidates"][0]["content"]["parts"][0]["text"]
                break
            except Exception as exc:
                print(f"Error in Gemini API. Restarting the process... {exc}")
                continue

        return response
