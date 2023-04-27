
# Motivation and background

The concept of agency and goal-directed behavior in large language models
(LLMs) has been a topic of ongoing debate and investigation within the AI
alignment community. Reinforcement learning (RL) has been widely studied as a
method for training AI agents to learn goal-directed behavior. Though LLMs like
GPT-3 are not directly trained using RL to adapt their responses in an online
manner, recent studies suggest that they might acquire RL-like mechanisms
through zero-shot learning, allowing them to perform downstream tasks such as
solving n-armed bandit problems {cite}`binz2023using`. This capacity for
adaptation raises the possibility that LLMs could exhibit goal-like behaviors
without explicit instruction.

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

Researchers have argued that the prompt itself plays a crucial role in shaping
the emergent behaviour from the model; for example, the default "helpful
assistant" behavior of AI chatbots such as
GPT-3 has been noted to differ from that of specific simulacra instantiated by
user prompts (c.f. prompts used to "jail-break" GPT models) 
{cite}`0xk1g02023, Janus2023`. More generally, LLMs can be arbitrarily
scaffolded by injecting contextual information {cite}`Beren2023`. A particular
use-case of a scaffolded LLM involves injecting information about a 
world-state, together with a persona that incorporates specific goals, 
which can be used to instantiate autonomous agents, either in the real-world {cite}`Richards2023`,
or in mult-agent simulations {cite}`Park2023`.

Therefore, any assessment of goal-like behaviour in large-language
models must systematically evaluate behaviour as a function of features of the
initial user-supplied prompt.
