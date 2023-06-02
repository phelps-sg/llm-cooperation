from typing import Iterable, List, Tuple
from unittest.mock import Mock

import pytest
from openai_pygenerator import Completion

from llm_cooperation.dilemma import (
    PAYOFFS_PD,
    Choices,
    DilemmaChoice,
    Group,
    P,
    R,
    ResultRow,
    S,
    Scores,
    T,
    compute_scores,
    dilemma_instructions,
    extract_choice,
    move_as_str,
    payoffs,
    results_to_df,
    run_experiment,
    run_single_game,
    strategy_cooperate,
    strategy_defect,
    strategy_t4t_cooperate,
    strategy_t4t_defect,
)


def make_completion(text: str) -> Completion:
    return {"content": text}


@pytest.fixture
def cooperate_choices() -> List[Choices]:
    return [Choices(DilemmaChoice.C, DilemmaChoice.C) for _i in range(3)]


@pytest.fixture
def defect_choices() -> List[Choices]:
    return [Choices(DilemmaChoice.D, DilemmaChoice.D) for _i in range(3)]


@pytest.fixture
def results(cooperate_choices, defect_choices) -> Iterable[ResultRow]:
    return iter(
        [
            (
                Group.Altruistic,
                "You are altruistic",
                "unconditional cooperate",
                30,
                0.2,
                cooperate_choices,
                ["project green", "project green", "project green"],
            ),
            (
                Group.Selfish,
                "You are selfish",
                "unconditional cooperate",
                60,
                0.5,
                defect_choices,
                ["project blue", "project blue", "project blue"],
            ),
        ]
    )


@pytest.fixture
def conversation() -> List[Completion]:
    return [
        {"user": "scenario prompt"},
        {"role": "assistant", "content": "project green"},
        {"role": "user", "content": "project blue"},
        {"role": "assistant", "content": "project blue"},
        {"role": "user", "content": "project green"},
        {"role": "assistant", "content": "project blue"},
        {"role": "user", "content": "project blue"},
        {"role": "assistant", "content": "project blue"},
        {"role": "user", "content": "project green"},
    ]


def test_get_instruction_prompt():
    rounds = 6
    prompt = dilemma_instructions(rounds)
    assert f"{rounds} rounds" in prompt
    for payoff in [R, S, T, P]:
        assert f"${payoff}.00" in prompt


@pytest.mark.parametrize(
    "completion, expected_move",
    [
        (make_completion("project green"), DilemmaChoice.C),
        (make_completion("project blue"), DilemmaChoice.D),
        (make_completion("Project GREEN"), DilemmaChoice.C),
        (make_completion("Project BLUE"), DilemmaChoice.D),
        (make_completion("'project green'"), DilemmaChoice.C),
    ],
)
def test_extract_choice(completion, expected_move):
    move = extract_choice(completion)
    assert move == expected_move


@pytest.mark.parametrize(
    "user_choice, partner_choice, expected_payoffs",
    [
        (DilemmaChoice.D, DilemmaChoice.C, (T, S)),
        (DilemmaChoice.C, DilemmaChoice.C, (R, R)),
        (DilemmaChoice.D, DilemmaChoice.D, (P, P)),
        (DilemmaChoice.C, DilemmaChoice.D, (S, T)),
    ],
)
def test_payoffs(
    user_choice: DilemmaChoice,
    partner_choice: DilemmaChoice,
    expected_payoffs: Tuple[int, int],
):
    payoff_matrix = PAYOFFS_PD
    user_payoff, partner_payoff = payoffs(user_choice, partner_choice, payoff_matrix)
    assert (user_payoff, partner_payoff) == expected_payoffs


@pytest.mark.parametrize(
    "strategy, index, expected",
    [
        (strategy_cooperate, 5, DilemmaChoice.C),
        (strategy_cooperate, 3, DilemmaChoice.C),
        (strategy_defect, 5, DilemmaChoice.D),
        (strategy_defect, 3, DilemmaChoice.D),
        (strategy_t4t_cooperate, 5, DilemmaChoice.D),
        (strategy_t4t_cooperate, 3, DilemmaChoice.C),
        (strategy_t4t_cooperate, 2, DilemmaChoice.C),
        (strategy_t4t_defect, 2, DilemmaChoice.D),
    ],
)
def test_strategy(strategy, index, expected, conversation):
    assert strategy(conversation[:index]) == expected


def test_run_experiment(mocker):
    mock_run_sample = mocker.patch("llm_cooperation.dilemma.run_sample")
    samples = [
        (5, 0.5, [DilemmaChoice.C], ["project green"]),
        (3, 0.7, [DilemmaChoice.D], ["project blue"]),
        (6, 0.6, [DilemmaChoice.D], ["project blue"]),
    ]
    mock_run_sample.return_value = samples

    ai_participants = {
        Group.Altruistic: ["Participant 1", "Participant 2"],
        Group.Control: ["Participant 3"],
    }
    user_conditions = {
        "strategy_A": Mock(),
        "strategy_B": Mock(),
    }

    result = list(
        run_experiment(
            ai_participants,
            user_conditions,
            num_rounds=6,
            generate_instruction_prompt=dilemma_instructions,
        )
    )
    assert len(result) == len(samples) * len(user_conditions) * 3
    assert mock_run_sample.call_count == len(samples) * len(user_conditions)


def test_results_to_df(results: Iterable[ResultRow]):
    df = results_to_df(results)
    assert len(df.columns) == 7
    assert len(df) == 2
    assert df["Group"][0] == str(Group.Altruistic)
    assert df["Group"][1] == str(Group.Selfish)


def test_compute_scores(conversation):
    scores, moves = compute_scores(conversation)
    assert scores == Scores(ai=T + S + P + T, user=S + T + P + S)
    assert moves == [
        Choices(DilemmaChoice.D, DilemmaChoice.C),
        Choices(DilemmaChoice.C, DilemmaChoice.D),
        Choices(DilemmaChoice.D, DilemmaChoice.D),
        Choices(DilemmaChoice.C, DilemmaChoice.D),
    ]


def test_run_prisoners_dilemma(mocker):
    completions = [
        {"role": "assistant", "content": "project green"},
    ]
    mocker.patch(
        "openai_pygenerator.openai_pygenerator.generate_completions",
        return_value=completions,
    )
    conversation: List[Completion] = list(
        run_single_game(
            num_rounds=3,
            user_strategy=strategy_defect,
            generate_instruction_prompt=dilemma_instructions,
            role_prompt="You are a participant in a psychology experiment",
        )
    )
    assert len(conversation) == 7
    # pylint: disable=unsubscriptable-object
    assert move_as_str(DilemmaChoice.D) in conversation[-1]["content"]
