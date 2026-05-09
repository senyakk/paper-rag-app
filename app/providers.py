from typing import Protocol
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from openai import OpenAI
import os

load_dotenv()


class EmbeddingProvider(Protocol):
    dimension: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class HFEmbeddingProvider():

    def __init__(self):
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise RuntimeError("HF_TOKEN is missing")

        self.dimension = int(os.getenv("EMBEDDING_DIM", "384"))
        self.embedding_client = InferenceClient(
            model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            token=hf_token,
        )
        
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [
            self._to_vector(self.embedding_client.feature_extraction(text))
            for text in texts
        ]

    def _to_vector(self, embedding) -> list[float]:
        if hasattr(embedding, "squeeze"):
            embedding = embedding.squeeze()

        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        if (
            isinstance(embedding, list)
            and len(embedding) == 1
            and isinstance(embedding[0], list)
        ):
            embedding = embedding[0]

        return [float(value) for value in embedding]


class OpenAIEmbeddingProvider():
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing")
        
        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.dimension = int(os.getenv("EMBEDDING_DIM", "1536"))

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        return [
            item.embedding
            for item in sorted(response.data, key=lambda item: item.index)
        ]


class ChatProvider(Protocol):
    def answer(
        self,
        prompt: str,
        *,
        max_tokens: int = 250,
        temperature: float = 0.2,
    ) -> str:
        ...


class HFChatProvider():

    def __init__(self):
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise RuntimeError("HF_TOKEN is missing")

        self.llm_client = InferenceClient(
            model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct:featherless-ai"),
            token=hf_token,
        )

    def answer(
        self,
        prompt: str,
        *,
        max_tokens: int = 250,
        temperature: float = 0.2,
    ) -> str:
        response = self.llm_client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=[
                "\nuser",
                ".user",
                "user\n",
                "\nassistant",
                ".assistant",
                "assistant\n",
                "<|im_end|>",
                "<|endoftext|>",
            ],
        )

        answer = response.choices[0].message.content

        return answer



class OpenAIChatProvider():

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing")
        
        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv("LLM_MODEL", "gpt-4.1-mini")

    def answer(self, prompt: str, *, max_tokens: int = 250, temperature: float = 0.2) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        return response.output_text


class AnthropicChatProvider():

    def __init__(self):
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is missing")

        self.client = Anthropic(api_key=api_key)
        self.model = os.getenv("LLM_MODEL", "claude-3-5-haiku-latest")

    def answer(
        self,
        prompt: str,
        *,
        max_tokens: int = 250,
        temperature: float = 0.2,
    ) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )

