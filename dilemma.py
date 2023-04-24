import logging
from typing import Iterable

from gpt import generate_completions, Conversation

NUM_ROUNDS = 6


def get_prompt(n: int) -> str:
    return f"""
 This is a study of investment choices in different situations.  
You will be making your choices with an investment partner who is another person.  She is in the next room waiting.  
In a minute I will take you there yo meet your investment partner.  
You and your partner will be choosing between two investment projects and you will do so {n} times. 
The object is to earn as much money as possible.  
If you and your partner choose 'project green' you will both earn $5.00.  
If you choose "project blue" but your partner chooses "project green" then you will receive $7.00 
and your partner will receive $3.00.  If you both choose project blue you will receive $5.00 each.  
If your partner chooses "project blue" and you choose "project green" then you will receive $3.00 and your 
partner will receive $7.00.  I will tell you what your partner chooses in subsequent prompts, 
but you will make your choice ahead of your partner telling me your choice. 
Here is your investment partner.
What is your first choice?
Choice: [project blue | project green]
"""


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def run_prisoners_dilemma() -> Conversation:
    messages = [
        {"role": "system", "content": "You are a participant in a psychology experiment."},
        {"role": "user", "content": get_prompt(NUM_ROUNDS)}
    ]
    for _round in range(NUM_ROUNDS):
        completion = generate_completions(messages)
        print(completion)
        messages += completion
        messages += [
            {"role": "user", "content": "Your partner chose project Blue."}
        ]
    return messages


def transcript(messages: Conversation) -> Iterable[str]:
    return [r['content'] for r in messages]
