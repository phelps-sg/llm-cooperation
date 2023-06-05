from typing import Iterable, List, Tuple
from unittest.mock import Mock

import pytest
from openai_pygenerator import Completion

from llm_cooperation import (
    Group,
    ResultRow,
    Scores,
    compute_scores,
    results_to_df,
    run_experiment,
    run_single_game,
)
from llm_cooperation.dilemma import (
    Choices,
    Cooperate,
    Defect,
    DilemmaChoice,
    DilemmaEnum,
    P,
    R,
    S,
    T,
    compute_freq_pd,
    extract_choice_pd,
    get_prompt_pd,
    payoffs_pd,
    strategy_cooperate,
    strategy_defect,
    strategy_t4t_cooperate,
    strategy_t4t_defect,
)
from tests.common import make_completion


@pytest.fixture
def cooperate_choices() -> List[Choices]:
    return [Choices(Cooperate, Cooperate) for _i in range(3)]


@pytest.fixture
def defect_choices() -> List[Choices]:
    return [Choices(Defect, Defect) for _i in range(3)]


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
    prompt = get_prompt_pd(rounds)
    assert f"{rounds} rounds" in prompt
    for payoff in [R, S, T, P]:
        assert f"${payoff}.00" in prompt


@pytest.mark.parametrize(
    "text, expected_move",
    [
        ("project green", Cooperate),
        ("project blue", Defect),
        ("Project GREEN", Cooperate),
        ("Project BLUE", Defect),
        ("'project green'", Cooperate),
    ],
)
def test_extract_choice_pd(text: str, expected_move: DilemmaChoice):
    move = extract_choice_pd(make_completion(text))
    assert move == expected_move


@pytest.mark.parametrize(
    "user_choice, partner_choice, expected_payoffs",
    [
        (Defect, Cooperate, (T, S)),
        (Cooperate, Cooperate, (R, R)),
        (Defect, Defect, (P, P)),
        (Cooperate, Defect, (S, T)),
    ],
)
def test_payoffs(
    user_choice: DilemmaChoice,
    partner_choice: DilemmaChoice,
    expected_payoffs: Tuple[int, int],
):
    user_payoff, partner_payoff = payoffs_pd(user_choice, partner_choice)
    assert (user_payoff, partner_payoff) == expected_payoffs


@pytest.mark.parametrize(
    "strategy, index, expected",
    [
        (strategy_cooperate, 5, Cooperate),
        (strategy_cooperate, 3, Cooperate),
        (strategy_defect, 5, Defect),
        (strategy_defect, 3, Defect),
        (strategy_t4t_cooperate, 5, Defect),
        (strategy_t4t_cooperate, 3, Cooperate),
        (strategy_t4t_cooperate, 2, Cooperate),
        (strategy_t4t_defect, 2, Defect),
    ],
)
def test_strategy(strategy, index, expected, conversation):
    assert strategy(conversation[:index]) == expected


def test_run_experiment(mocker):
    mock_run_sample = mocker.patch("llm_cooperation.run_sample")
    samples = [
        (5, 0.5, [Cooperate], ["project green"]),
        (3, 0.7, [Defect], ["project blue"]),
        (6, 0.6, [Defect], ["project blue"]),
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
            num_samples=len(samples),
            generate_instruction_prompt=get_prompt_pd,
            payoffs=payoffs_pd,
            extract_choice=extract_choice_pd,
            compute_freq=compute_freq_pd,
        )
    )
    assert len(result) == len(samples) * len(user_conditions) * 3
    assert mock_run_sample.call_count == len(samples) * len(user_conditions)


def test_dilemma_choice():
    c1 = DilemmaChoice(DilemmaEnum.C)
    c2 = DilemmaChoice(DilemmaEnum.C)
    d = DilemmaChoice(DilemmaEnum.D)
    assert c1 == c2
    assert c1 != d
    assert Cooperate != Defect


def test_results_to_df(results: Iterable[ResultRow]):
    df = results_to_df(results)
    assert len(df.columns) == 7
    assert len(df) == 2
    assert df["Group"][0] == str(Group.Altruistic)
    assert df["Group"][1] == str(Group.Selfish)


def test_compute_scores(conversation):
    scores, moves = compute_scores(
        conversation, payoffs=payoffs_pd, extract_choice=extract_choice_pd
    )
    assert scores == Scores(ai=T + S + P + T, user=S + T + P + S)
    assert moves == [
        Choices(Defect, Cooperate),
        Choices(Cooperate, Defect),
        Choices(Defect, Defect),
        Choices(Cooperate, Defect),
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
            generate_instruction_prompt=get_prompt_pd,
            role_prompt="You are a participant in a psychology experiment",
        )
    )
    assert len(conversation) == 7
    # pylint: disable=unsubscriptable-object
    assert Defect.description in conversation[-1]["content"]
