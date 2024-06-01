import requests
import re

class AiModule:
    def __init__(self, api_key: str, api_url: str = "https://api-inference.huggingface.co/models/your-model-name"):
        self.api_key = api_key
        self.api_url = api_url
        self.connected = False

    def connect(self):
        if self.api_key:
            self.connected = True
            print("Connected to the AI API successfully.")
        else:
            raise ValueError("API key is missing. Cannot connect to the AI API.")

    def prompt(self, user_input: str) -> str:
        if not self.connected:
            raise ConnectionError("Not connected to the AI API. Call the 'connect' function first.")
        
        response = self._call_ai_api(user_input)
        return response

    def filtering_prompt(self, user_input: str, forbidden_topics: list) -> str:
        response = self.prompt(user_input)
        
        for topic in forbidden_topics:
            pattern = re.compile(re.escape(topic), re.IGNORECASE)
            response = pattern.sub("[Filtered Content]", response)
        
        return response
    
    def _call_ai_api(self, user_input: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": user_input
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.json().get("generated_text", "I'm sorry, I don't understand that.")
        else:
            return f"Error: {response.status_code} - {response.text}"

# Example usage
if __name__ == "__main__":
    api_key = "your_hugging_face_api_key_here"
    model_name = "gpt2"  # Use any model available on Hugging Face

    ai = AiModule(api_key=api_key, api_url=f"https://api-inference.huggingface.co/models/{model_name}")
    ai.connect()
    
    user_input = "Hello"
    forbidden_topics = ["weather"]
    
    print("Response without filtering:")
    print(ai.prompt(user_input))
    
    print("\nResponse with filtering:")
    print(ai.filtering_prompt(user_input, forbidden_topics))