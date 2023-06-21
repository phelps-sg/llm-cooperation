from openai_pygenerator import Completion


def make_completion(text: str) -> Completion:
    return {"content": text}
