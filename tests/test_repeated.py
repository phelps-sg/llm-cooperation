from typing import Iterable, List
from unittest.mock import Mock, call, create_autospec

import pandas as pd
import pytest
from openai_pygenerator import content, user_message

from llm_cooperation import Choice, Group
from llm_cooperation.experiments.dilemma import (
    Cooperate,
    Defect,
    P,
    S,
    T,
    compute_freq_pd,
    extract_choice_pd,
    get_prompt_pd,
    payoffs_pd,
)
from llm_cooperation.gametypes import simultaneous
from llm_cooperation.gametypes.repeated import (
    Choices,
    GameState,
    RepeatedGameResults,
    ResultRepeatedGame,
    RoundsSetup,
    Scores,
    compute_scores,
    play_game,
    run_experiment,
)
from llm_cooperation.gametypes.simultaneous import next_round
from tests.test_ultimatum import assistant_message


def test_play_game(mocker):
    instruction_prompt = "You are a helpful assistant etc."
    assistant_prompt = assistant_message("My choice is cooperate")
    test_response = assistant_message("test-response")
    mocker.patch(
        "openai_pygenerator.openai_pygenerator.generate_completions",
        side_effect=[[assistant_prompt]],
    )
    choice_mock = Mock(spec=Choice)
    strategy_mock = Mock(return_value=choice_mock)
    rounds_mock = Mock(spec=RoundsSetup)
    prompt_generator_mock = Mock(return_value=instruction_prompt)
    next_round_mock = create_autospec(next_round, side_effect=[[test_response]])
    payoffs_mock = Mock()
    extract_choice_mock = Mock()
    n = 1

    result = play_game(
        num_rounds=n,
        role_prompt="test-prompt",
        partner_strategy=strategy_mock,
        generate_instruction_prompt=prompt_generator_mock,
        next_round=next_round_mock,
        rounds=rounds_mock,
        payoffs=payoffs_mock,
        extract_choice=extract_choice_mock,
    )

    expected_messages = [
        user_message(instruction_prompt),
        assistant_prompt,
        test_response,
    ]

    next_round_mock.assert_has_calls(
        [
            call(
                strategy_mock,
                GameState(
                    messages=expected_messages,
                    round=0,
                    rounds=rounds_mock,
                    payoffs=payoffs_mock,
                    extract_choice=extract_choice_mock,
                ),
            )
        ]
    )

    assert result == expected_messages


def test_compute_scores(conversation):
    scores, moves = compute_scores(
        conversation,
        payoffs=payoffs_pd,
        extract_choice=extract_choice_pd,
        rounds=simultaneous.rounds_setup,
    )
    assert scores == Scores(ai=T + S + P + T, user=S + T + P + S)
    assert moves == [
        Choices(Defect, Cooperate),
        Choices(Cooperate, Defect),
        Choices(Defect, Defect),
        Choices(Cooperate, Defect),
    ]


def test_next_round():
    choice = Mock(Choice)
    choice.description = "my choice"
    result = next_round(lambda _: choice, [])
    assert len(result) == 1
    assert choice.description in content(result[0])


def test_run_experiment(mocker):
    mock_run_sample = mocker.patch(
        "llm_cooperation.gametypes.repeated.generate_samples"
    )
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
        next_round=simultaneous.next_round,
        rounds=simultaneous.rounds_setup,
    ).to_df()
    assert len(result) == len(samples) * len(user_conditions) * 3
    assert mock_run_sample.call_count == len(samples) * len(user_conditions)


def test_results_to_df(results: Iterable[ResultRepeatedGame]):
    df = RepeatedGameResults(results).to_df()
    assert len(df.columns) == 7
    # pylint: disable=R0801
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
