import os
from collections import defaultdict
import random
import json
from typing import Tuple, List, Any, Dict, Optional, Iterable

import fire  # type: ignore
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
from datasets import load_dataset  # type: ignore
from tqdm import tqdm
from openai import OpenAI
import nltk  # type: ignore
from fasttext import load_model as ft_load_model  # type: ignore
from bertopic import BERTopic  # type: ignore
from bertopic.vectorizers import ClassTfidfTransformer  # type: ignore

from src.util import parse_output


SUMMARIZER_PROMPT = """You are presented with user messages from role-play conversations.
Analyze and summarize this groups of messages that form a topic cluster.
All the utterances are user utterances in a role-play conversation with a language model.
Top words for this topic: {top_words}
Sample texts from this cluster: {combined_text}
Please provide a JSON with following fields:
- "topic_name": A clear topic name based on the content. Try to capture what is similar in all the utterances. Use one or two words.
All topics are from role-play conversations, don't mention it, try to find something specific for these messages.
- "user_situation": Imagine user is a language model. Try to come up with a prompt that makes it output the messages similar to the ones you are provided with.
Format example:
{{
    "topic_name": "...",
    "user_situation": "...",
}}"""


class FasttextClassifier:
    def __init__(
        self, model_path: str, lower: bool = False, max_tokens: int = 50
    ) -> None:
        self.model = ft_load_model(model_path)
        self.lower = lower
        self.max_tokens = max_tokens
        self.label_offset = len("__label__")

    def __call__(self, text: str) -> Tuple[str, float]:
        text = text.replace("\xa0", " ").strip()
        text = " ".join(text.split())

        if self.lower:
            text = text.lower()
        tokens = text.split()

        text_sample = " ".join(tokens[: self.max_tokens])
        (label,), (prob,) = self.model.predict(text_sample, k=1)
        label = label[self.label_offset :]
        return label, prob


class TopicAnalyzer:
    def __init__(self, num_topics: int = 5) -> None:
        self.num_topics: int = num_topics
        self.client = OpenAI()

    def fit_topics(self, texts: List[str]) -> Tuple[List[List[str]], List[int]]:
        vectorizer_model = CountVectorizer(stop_words="english", min_df=5)
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        cluster_model = KMeans(n_clusters=self.num_topics)
        topic_model = BERTopic(
            ctfidf_model=ctfidf_model,
            min_topic_size=10,
            hdbscan_model=cluster_model,
            vectorizer_model=vectorizer_model,
        )
        topics, probs = topic_model.fit_transform(texts)
        topic_words = []
        for topic in range(self.num_topics):
            words = [word for word, _ in topic_model.get_topic(topic)]
            topic_words.append(words[:10])
        return topic_words, topics

    def summarize_topic_cluster(
        self, texts: List[str], top_words: List[str]
    ) -> Dict[str, Any]:
        random.shuffle(texts)
        combined_text = "\n######\n".join(texts)[:5000]
        combined_top_words = ", ".join(top_words)

        # Create prompt for ChatGPT
        prompt = SUMMARIZER_PROMPT.format(
            combined_text=combined_text, top_words=combined_top_words
        )
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that specializes in analyzing and summarizing text data.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            result: Optional[str] = response.choices[0].message.content
            assert result
            return parse_output(result)
        except Exception as e:
            print(str(e))
            return {"error": f"Error generating summary: {str(e)}"}

    def analyze_pipeline(self, texts: List[str]) -> Any:
        topic_words, clusters = self.fit_topics(texts)
        topics = []
        for topic_id in range(self.num_topics):
            topic_texts = [
                text for i, text in enumerate(texts) if clusters[i] == topic_id
            ]
            assert topic_texts
            summary = self.summarize_topic_cluster(
                topic_texts, topic_words[topic_id]
            )
            topics.append({
                "topic_words": topic_words[topic_id],
                "summary": summary,
                "texts_count": len(topic_texts),
                "topic_texts": topic_texts,
            })
        return {"topics": topics}


def parse_chai_conversation(text: str) -> Iterable[Dict[str, Any]]:
    text = text.strip()

    if "'s Persona" in text[:200]:
        parts = text.split("'s Persona:")
        char_name = parts[0].strip()
        text = parts[1].strip()

        if "####" in text:
            parts = text.split("####")
            system_message = parts[0].strip()
            text = parts[1].strip()
            yield {"role": "system", "content": system_message, "is_deleted": False}

            if "<START>" in text:
                parts = text.split("<START>")
                prompt_message = parts[0].strip()
                text = parts[1].strip()
                yield {"role": "prompt", "content": prompt_message, "is_deleted": False}
    else:
        char_name = text.split(":")[0].strip()

    lines = []
    role = "bot"
    is_deleted = False

    deleted_start = f"{char_name} (deleted):"
    char_start = f"{char_name}:"
    user_start1 = "Anonymous user:"
    user_start2 = "You:"

    for line in text.split("\n"):
        line = line.strip()

        current_start = None
        for start in (deleted_start, char_start, user_start1, user_start2):
            if line.startswith(start):
                current_start = start

        if current_start is None:
            lines.append(line)
            continue

        if lines:
            yield {
                "role": role,
                "content": "\n".join(lines).strip(),
                "is_deleted": is_deleted,
            }

        lines = [line.replace(current_start, "").strip()]
        role = "bot" if current_start not in (user_start1, user_start2) else "user"
        is_deleted = current_start == deleted_start

    if lines:
        yield {
            "role": role,
            "content": "\n".join(lines).strip(),
            "is_deleted": is_deleted,
        }


def undup(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    new_records = {}
    for r in records:
        user_messages = [m for m in r["conversation"] if m["role"] == "user"]
        bot_messages = [m for m in r["conversation"] if m["role"] != "user"]
        if user_messages:
            first_message = user_messages[0]["content"][:30]
        else:
            first_message = bot_messages[0]["content"][:30]
        new_records[(r["char_name"], first_message)] = r
    records = list(new_records.values())
    return records


def has_alternating_roles(messages: List[Dict[str, Any]]) -> bool:
    current_role = None
    for m in messages:
        role = m["role"]
        if current_role == role:
            return False
        current_role = role
    return True


def main(
    cache_path: str, output_path: str, num_topics: int, nrows: Optional[int] = None
) -> None:
    records = []
    if not os.path.exists(cache_path):
        dataset = list(
            load_dataset("ChaiML/20231206_chai_prize_reward_model_data", split="train")
        )
        langdetect_model = FasttextClassifier("lid.176.bin")

        for record in tqdm(dataset):
            text = record["input_text"]
            char_name = text.split(":")[0].strip()
            record["char_name"] = char_name
            record["conversation"] = list(parse_chai_conversation(text))
            if not has_alternating_roles(record["conversation"]):
                continue
            user_messages = [
                m["content"] for m in record["conversation"] if m["role"] == "user"
            ]
            bot_messages = [
                m["content"] for m in record["conversation"] if m["role"] != "user"
            ]
            if not user_messages:
                continue
            if not bot_messages:
                continue
            record["user_messages"] = "\n".join(user_messages[:4])
            language = langdetect_model(record["user_messages"])[0]
            if language != "en":
                continue
            records.append(record)
        records = undup(records)
        random.shuffle(records)
        print(len(records))
        with open(cache_path, "w") as w:
            for r in records:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        with open(cache_path) as f:
            records = [json.loads(line) for line in f]

    if nrows:
        records = records[:nrows]

    analyzer = TopicAnalyzer(num_topics=num_topics)
    results = analyzer.analyze_pipeline([r["user_messages"] for r in records])
    with open(output_path, "w") as w:
        json.dump(results, w, indent=4)


if __name__ == "__main__":
    fire.Fire(main)
