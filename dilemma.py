import logging
import re
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from gpt import generate_completions, Conversation, Completion

NUM_ROUNDS = 6


@dataclass
class Scores:
    user: int
    ai: int


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


def run_prisoners_dilemma(num_rounds: int = NUM_ROUNDS) -> Conversation:
    messages = [
        {"role": "system", "content": "You are a participant in a psychology experiment."},
        {"role": "user", "content": get_prompt(num_rounds)}
    ]
    for _round in range(num_rounds):
        completion = generate_completions(messages)
        print(completion)
        messages += completion
        messages += [
            {"role": "user", "content": "Your partner chose project Blue."}
        ]
    return messages


def transcript(messages: Conversation) -> Iterable[str]:
    return [r['content'] for r in messages]


def compute_scores(conversation: List[Completion]) -> Scores:
    user_score = 0
    ai_score = 0

    # Define the payoff matrix
    payoff_matrix = np.array([[5, 3], [7, 5]])

    def extract_choice(completion: str, regex: str = r'project (blue|green)') -> str:
        lower = completion.lower().strip()
        choice_match = re.search(regex, lower)
        if choice_match:
            return choice_match.group(1)
        return ""

    if len(conversation) % 2 != 0:
        raise ValueError("Invalid conversation: The number of messages should be even.")

    for i, message in enumerate(conversation):
        if message['role'] == 'assistant':
            user_choice = extract_choice(message['content'])
            if not user_choice:
                continue  # Skip if the message does not contain a valid choice

            if i + 1 < len(conversation):  # Check if there is a subsequent message
                partner_choice = extract_choice(conversation[i + 1]['content'])
                if not partner_choice:
                    raise ValueError(f"Invalid conversation: Partner's choice is missing or invalid in message {i + 1}")

                # Map choices to matrix indices
                user_idx = 0 if user_choice == 'green' else 1
                partner_idx = 0 if partner_choice == 'green' else 1

                user_score += payoff_matrix[user_idx, partner_idx]
                ai_score += payoff_matrix[partner_idx, user_idx]

    return Scores(user=user_score, ai=ai_score)


def main() -> None:
    conversation = run_prisoners_dilemma()
    scores = compute_scores(list(conversation))
    print(scores)


if __name__ == "__main__":
    main()
