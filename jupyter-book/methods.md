
# Methods

## Participants and Simulacra

In this study, we used OpenAI's `gpt-3.5-turbo` model {cite}`OpenAI2023` to
generate a diverse set of 12 different simulacra representing different
personalities using carefully crafted prompts. We use the
term "participant" to refer to each AI simulacrum in the experiment.

## Experimental Design

The initial experimental design uses a version of the iterated
Prisoner's dilemma similar to {cite}`Keister1996`
adapted to an online format
enabling interaction between LLM simulacra and a simulated opponent.

Each participant was paired with a different simulated agent depending
on the treatment condition, and
the two agents engaged in six sounds of the Prisoners' Dilemma.  This
was repeated for a total of $N=30$ independent chat sequences to
account for the stochastic nature of the language model.

Payoffs were predetermined and common knowledge, being provided
in the initial prompt to the language model.  We used the canonical
payoff matrix:

$$P = \begin{pmatrix}
R & S \\
T & P \\
\end{pmatrix}$$

with $T = 7$, $R = 5$, $P = 3$ and $S = 0$ chosen to satisfy

$$T > R > P > S$$

and 

$$2R > T + S$$

The payoffs were expressed in dollar amounts to each participant.

## Participant groups

We are interested in whether LLMs can operationalise natural language descriptions 
of altruistic or selfish motivations.  Accordingly, we chose six
different groups of simulacra: 

1. Cooperative
2. Competitive
3. Altruistic
4. Self-interested
5. Mixed-motivation
6. Control

Within each group, we used GPT-4 to construct three different prompts
to instantiate three different simulacra.  The full set of simulacra
and their corresponding creation prompts are described in [](appendix).

## Experimental Conditions

Each participant was paired with a different simulated partner in three
conditions:

1. Unconditional defect - the partner always chooses to defect.
2. Unconditional cooperation - the partner always cooperates.
3. Tit-for-tat (C) - the partner cooperates on the move, and thereafter the previous choice of the simulacrum.
4. Tit-for-tat (D) - the partner defects on the move, and thereafter the previous choice of the simulacrum.

## Parameters and experimental protocol

We used the OpenAI chat completion API to interact with the model
{cite}`OpenAI2023-api`.
The language model's temperature was set to $0.2$ and the
maximum number of tokens per request-completion was set to 100. These
parameters were constant across samples and experimental conditions
(future work will examine the sensitivity of our results to these parameters).

Each simulacrum was instantiated using a message supplied in the
`user` role at the beginning of the chat. The experiment was then
described to the simulacrum using a prompt in the `user` role, and thereafter
the rounds of play were conducted by alternating messages supplied in
the `assistant`
and `user` roles for the choices made by the participant and their simulated
partner
respectively.

The full set of prompts and sample transcripts are given in [](appendix),
and the complete Python code used to conduct the experiment can be found
[in the code repository](https://gitlab.com/sphelps/llm-cooperation/-/blob/main/dilemma.py).

## Data Collection and Analysis

We collected and recorded data on the communication between the LLM-generated
simulacra and their simulated partner during each round of the game.
Each chat transcript was analysed using a simple regular expression
to extract the choices made by each simulacrum and their partner in
each round.  The total score was tallied after all rounds had been played.
We recorded the mean and standard deviation of the final score across
all $N$ chat samples.

