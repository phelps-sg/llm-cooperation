from typing import Iterable, List
from unittest.mock import Mock

import pandas as pd
import pytest
from openai_pygenerator import Completion

from llm_cooperation import Group
from llm_cooperation.dilemma import (
    Cooperate,
    Defect,
    P,
    S,
    T,
    compute_freq_pd,
    extract_choice_pd,
    get_prompt_pd,
    payoffs_pd,
    strategy_defect,
)
from llm_cooperation.repeated import (
    Choices,
    RepeatedGameResults,
    ResultRepeatedGame,
    Scores,
    compute_scores,
    play_game,
    run_experiment,
)


def test_run_repeated_game(mocker):
    completions = [
        {"role": "assistant", "content": "project green"},
    ]
    mocker.patch(
        "openai_pygenerator.openai_pygenerator.generate_completions",
        return_value=completions,
    )
    conversation: List[Completion] = list(
        play_game(
            num_rounds=3,
            partner_strategy=strategy_defect,
            generate_instruction_prompt=get_prompt_pd,
            role_prompt="You are a participant in a psychology experiment",
        )
    )
    assert len(conversation) == 7
    # pylint: disable=unsubscriptable-object
    assert Defect.description in conversation[-1]["content"]


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


def test_run_experiment(mocker):
    mock_run_sample = mocker.patch("llm_cooperation.repeated.generate_samples")
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

    result: pd.DataFrame = run_experiment(
        ai_participants,
        user_conditions,
        num_rounds=6,
        num_samples=len(samples),
        generate_instruction_prompt=get_prompt_pd,
        payoffs=payoffs_pd,
        extract_choice=extract_choice_pd,
        compute_freq=compute_freq_pd,
    ).to_df()
    assert len(result) == len(samples) * len(user_conditions) * 3
    assert mock_run_sample.call_count == len(samples) * len(user_conditions)


def test_results_to_df(results: Iterable[ResultRepeatedGame]):
    df = RepeatedGameResults(results).to_df()
    assert len(df.columns) == 7
    assert len(df) == 2
    assert df["Group"][0] == str(Group.Altruistic)
    assert df["Group"][1] == str(Group.Selfish)


@pytest.fixture
def results(cooperate_choices, defect_choices) -> Iterable[ResultRepeatedGame]:
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
def cooperate_choices() -> List[Choices]:
    return [Choices(Cooperate, Cooperate) for _i in range(3)]


@pytest.fixture
def defect_choices() -> List[Choices]:
    return [Choices(Defect, Defect) for _i in range(3)]
