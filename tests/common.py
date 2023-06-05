from openai import Completion


def make_completion(text: str) -> Completion:
    return {"content": text}
