import copy
import json
from typing import Any, Dict, List, Optional, cast

from jinja2 import Template
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from src.data import ChatMessages
from src.provider import LLMProvider


def encode_prompt(template_path: str, **kwargs: Any) -> str:
    with open(template_path, encoding="utf-8") as f:
        template = Template(f.read())
    return template.render(**kwargs).strip()


def parse_output(output: str) -> Dict[str, Any]:
    start_index = output.find("{")
    end_index = output.rfind("}")
    text = output[start_index : end_index + 1]
    text = text.strip()
    record: Dict[str, Any] = json.loads(text)
    for k in record:
        assert isinstance(k, str)
    return record


def generate(messages: ChatMessages, provider: LLMProvider, **kwargs: Any) -> str:
    params = copy.deepcopy(provider.params)
    for k, v in kwargs.items():
        params[k] = v

    messages_copy = copy.deepcopy(messages)

    # If we have additional system prompt in provider, add it to messages
    if provider.system_prompt != "" and messages_copy[0]["role"] == "system":
        messages_copy[0]["content"] = provider.system_prompt + "\n\n" + messages_copy[0]["content"]

    if provider.merge_system and messages_copy[0]["role"] == "system":
        system_content = messages_copy[0]["content"]
        user_content = messages_copy[1]["content"]
        messages_copy = messages_copy[1:]
        messages_copy[0]["content"] = f"{system_content}\n\nUser: {user_content}"

    casted_messages = [cast(ChatCompletionMessageParam, message) for message in messages_copy]
    chat_completion = provider.api.chat.completions.create(
        model=provider.model_name, messages=casted_messages, **params
    )
    return str(chat_completion.choices[0].message.content).strip()
