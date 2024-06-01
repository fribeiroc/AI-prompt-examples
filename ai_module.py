from huggingface_hub import InferenceClient

class AiModule:
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.client = None

    def connect(self):
        """Makes connection to with Inference API"""
        if self.api_key:
            self.client = InferenceClient(token=self.api_key, model=self.model_name)
            print("Connected to the AI API successfully.")
        else:
            raise ValueError("API key is missing. Cannot connect to the AI API.")

    def prompt(self, user_input: str) -> str:
        """Inputs a prompt for the model and returns a response"""
        if self.client is None:
            raise ConnectionError("Not connected to the AI API. Call the 'connect' function first.")
        
        response = self.client.text_generation(user_input)
        return response if response else "No response from the API."

    def filtering_prompt(self, user_input: str, forbidden_topics: list) -> str:
        """Contacts the model to generate another response without mentioning the forbidden topics"""
        response = self.prompt(user_input)
        
        for topic in forbidden_topics:
            if topic in response:
                response = self.client.text_generation(f"Change the following text avoiding the topics: {topic}\n {response}")
        
        return response

if __name__ == "__main__":
    api_key = "User API Key" # User API Key from Hugging Face
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"  # Use any model available on Hugging Face

    ai = AiModule(api_key=api_key, model_name=model_name)
    ai.connect()
    
    user_input = "Tell me about the weather today"
    forbidden_topics = ["weather", "temperature"]
    
    print("Response without filtering:")
    print(ai.prompt(user_input))
    
    print("\nResponse with filtering:")
    print(ai.filtering_prompt(user_input, forbidden_topics))