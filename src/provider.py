from typing import Dict, Any

from openai import OpenAI

DEFAULT_PARAMS = {
    "temperature": 0.6,
    "top_p": 0.9,
    "max_tokens": 1536,
}


class LLMProvider:
    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        system_prompt: str = "",
        merge_system: bool = False,
        params: Dict[str, Any] = DEFAULT_PARAMS,
        **kwargs: Any
    ) -> None:
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.api = OpenAI(base_url=base_url, api_key=api_key)
        self.params = params
        self.merge_system = merge_system
        for k, v in DEFAULT_PARAMS.items():
            if k not in self.params:
                self.params[k] = v

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "system_prompt": self.system_prompt,
            "params": self.params,
        }
