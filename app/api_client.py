from openai import OpenAI
from app import config


class ChatClient:
    def __init__(self):
        self.client = OpenAI(
            base_url=config.BASE_URL,
            api_key=config.API_KEY,
        )
        self.messages = []

    def send_message(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=self.messages,
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE,
        )

        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})
        return assistant_message

    def clear_history(self):
        self.messages.clear()
