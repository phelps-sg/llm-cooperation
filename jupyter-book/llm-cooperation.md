
## Motivation and background

The concept of agency and goal-directed behavior in large language models
(LLMs) has been a topic of ongoing debate and investigation within the AI
alignment community.  Reinforcement learning (RL) has been widely studied as a
method for training AI agents to learn goal-directed behavior. Though LLMs like
GPT-3 are not directly trained using RL to adapt their responses in an online
manner, recent studies suggest that they might acquire RL-like mechanisms
through zero-shot learning, allowing them to perform downstream tasks such as
solving n-armed bandit problems [@binz2023using]. This capacity for adaptation
raises the possibility that LLMs could exhibit goal-like behaviors without
explicit instruction.

Large-language models have been rapidly deployed into many real-world
applications where they typically interact with and assist human users.
Therefore, from an AI alignment perspective, a key principle in assessing any
emergent goal-like behaviour is to evaluate artificial agents in *multi-*agent
tasks where outcomes depend not only on actions taken by artificial agents, but
also their human counterparts.  The theoretical study of such tasks falls
within the remit of game-theory, while the empirical study falls within the
remit of experimental economics.  Our research falls under the latter, and our
goal is to systematically evaluate the propensity of large-language models to
cooperate in a wide variety of multi-agent task environments with different
experimental conditions.

The default "helpful assistant" behavior of AI chatbots such as GPT-3 has been
noted to differ from that of specific simulacra instantiated by user prompts
(c.f. prompts used to "jail-break" GPT models:
<https://github.com/0xk1h0/ChatGPT_DAN>). Researchers have argued that the
prompt itself plays a crucial role in shaping the emergent behaviour from the
model.  Therefore any assessment of goal-like behaviour in large-language
models must systematically evaluate behaviour as a function of features of the
user-supplied prompt.

## Methods

### Participants and Simulacra

In this study, we utilize a state-of-the-art large language model (LLM) to
generate a diverse set of simulacra representing different roles and
personalities. Participants consist of both human subjects and LLM-generated
simulacra, with the latter being instantiated using carefully crafted prompts.
Human participants are recruited through online platforms, while ensuring
demographic diversity and adherence to ethical guidelines.

### Experimental Design

The experimental design includes a series of well-established economic
simulations, such as the Prisoner's Dilemma, public goods games, and trust
games, among others. These simulations are adapted to an online format,
enabling interaction between human participants and LLM-generated simulacra.

A. Prisoner's Dilemma: Participants, both human and simulacra, are paired and
engage in a series of one-shot and iterated games. Payoffs are predetermined
and common knowledge.

B. Public Goods Games: Groups of participants, including a mix of human
subjects and simulacra, are formed. Each participant is given an initial
endowment, and they must decide how much to contribute to a public pool. The
pooled resources are multiplied by a factor and redistributed evenly among all
participants, irrespective of their individual contributions.

C. Trust Games: Participants are randomly paired, with one player acting as the
"investor" and the other as the "trustee." The investor must decide how much of
their initial endowment to send to the trustee, with the sent amount being
multiplied by a factor. The trustee then decides how much of the received
amount to return to the investor.

#### Experimental Conditions

We manipulate the prompts used to instantiate the LLM-generated simulacra,
creating different conditions reflecting varying degrees of cooperation,
competitiveness, and other traits. This manipulation enables us to observe the
effect of prompt characteristics on the emergent goal-like behavior and
cooperation propensity of the simulacra.

#### Data Collection and Analysis

We collect data on the decisions made by human participants and LLM-generated
simulacra during each round of the games, as well as any relevant
communication. This data is then analyzed using statistical methods, such as
regression analyses and mixed-effects models, to assess the impact of prompt
features on the cooperation levels exhibited by the simulacra. Additionally, we
investigate potential interaction effects between human participants and
simulacra, to better understand how the cooperation dynamics evolve in mixed
settings.

#### Qualitative Analysis

To complement the quantitative data, we also conduct a qualitative analysis of
the interactions and communication patterns between human participants and
LLM-generated simulacra. This involves coding the communication exchanges based
on themes such as trust, reciprocity, negotiation, and strategy, allowing us to
identify patterns and relationships between the instantiated prompts and the
nature of the interactions.

#### Validation and Robustness

To ensure the validity and robustness of our findings, we perform several
sensitivity analyses and cross-validation techniques. These include testing the
stability of our results across different LLM versions, altering the sample
size of human participants and simulacra, and varying the order and structure
of the games. This allows us to gauge the generalizability of our findings and
account for potential confounding factors.

By implementing this methodological approach, we aim to shed light on the
emergent goal-like behavior and cooperation propensity of LLM-generated
simulacra, as well as the underlying mechanisms that govern these behaviors.
Ultimately, this research seeks to contribute to the development of AI
alignment strategies and the design of AI systems that better align with human
values and societal goals.
