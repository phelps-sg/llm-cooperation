
# Motivation and background

The concept of agency and goal-directed behavior in large language models
(LLMs) has been a topic of ongoing debate and investigation within the AI
alignment community. While there are a diverse set of opinions on the
subject, a challenge for researchers is that the internal processing of large
language models is largely opaque, and in the case of recent models
such as GPT-4 the training procedures themselves are also subject to a 
degree of secrecy. Therefore, objective assessment of the capabilities 
of large-language models cannot be conducted through inductive reasoning 
starting from first principles, but instead is a matter of empirical investigation, 
with experiments being the ultimate arbiter of what they can or can't do, 
e.g. {cite}`Google2023`.

Reinforcement learning (RL) has been widely studied as a method for training 
AI agents to learn goal-directed behavior. Though LLMs like
GPT-3 are not directly trained using RL to adapt their responses in an online
manner, recent studies suggest that they might acquire RL-like mechanisms
through zero-shot learning, allowing them to perform downstream tasks such as
solving n-armed bandit problems {cite}`binz2023using`. This capacity for
adaptation raises the possibility that LLMs could exhibit goal-like behaviors
without explicit instruction.

Large-language models have been rapidly deployed into many real-world
applications where they typically interact with and assist human users.
Therefore, a key principle in assessing any
emergent goal-like behaviour for these use-cases is to evaluate artificial agents in multi-agent
tasks where outcomes depend not only on actions taken by artificial agents, but
also their human counterparts. When evaluating incentives, the aspect of the outcome that is of interest is the
expected utility obtained by each party, and the theoretical study of such tasks falls
within the remit of game-theory, while the empirical study falls within the
remit of experimental economics. Our research falls under the latter, and our
goal is to systematically evaluate the propensity of large-language models to
cooperate in a wide variety of multi-agent task environments with different
experimental conditions.

Many scenarios discussed in the AI alignment debate focus on competitive zero-sum 
interactions.  For example, a common analogy is competition for resources between
different species occupying a particular niche; for example, {cite}`Tegmark2023` argues

> We humans drove the West African Black Rhino extinct not because we were rhino-haters, but because we were smarter than them and had different goals for how to use their habitats and horns. In the same way, superintelligence with almost any open-ended goal would want to preserve itself and amass resources to accomplish that goal better.

In an AI safety context, the intuition behind such arguments is that AI systems have been to shown to
outsmart humans in zero-sum games such as Chess and Go, and therefore if AI systems
find themselves in situations in which they are competing with humans, the AI "species"
will clearly out-compete inferior humans.

However, many interactions in both natural and artificial settings are characterized
by non-zero-sum payoff structures {cite}`Phelps2015`.  A famous example that was
used to analyse existential risk of nuclear conflict during the cold war is the 
Prisoner's Dilemma {cite}`Axelrod1997`.  

In addition to the Prisoner's Dilemma, it is important to consider related non-zero-sum games such as the Hawk-Dove
game, also known as the Chicken game. This game was introduced by Maynard Smith in his paper "The Logic of Animal
Conflict" as
a way to analyze the outcomes of competition for resources among animals. The Hawk-Dove, also known as "Chicken", game
demonstrates that, in certain payoff structures, limited conflict can be an evolutionary equilibrium when interactions
are repeated within a
large population {cite}`smith1973`. This game has been also applied to an analysis of existential risk in nuclear
conflict; {cite}`dixit2019we` argues that the Cuban missile crisis can analysed as a high-stakes dynamic chicken game in
which neither the USSR nor the USA wanted to "blink" first.

Interestingly, in a one-shot version of the game between human players, behaving
irrationally by limiting one's options
can be a superior strategy. For example, in a game of Chicken where two opposing drivers are on a collision course and
neither driver wants to be seen as the "chicken" by swerving, if we limit our choices by removing the steering wheel,
and make this common knowledge, then the opposing driver's best response is to swerve. Similar arguments were used
during the cold war to remove rational deliberation from the decision whether to retaliate in the event
of a preemptive strike by the enemy by "taking the human out of the loop" and putting systems on automated
hair-trigger alert. Thus, in contrast to chess or Go, in non-zero-sum interactions, the most ruthless agents, or those
with superior cognitive
capacity, do not necessarily prevail.

Moreover, in both one-shot and iterated Prisoner's Dilemma games with the number of rounds being common
knowledge, the rational strategy is to defect, but experiments have shown that real people tend to cooperate,
albeit conditionally, using strategies such as Tit-for-Tat. The fact that real people cooperate in these scenarios,
despite the seemingly rational strategy to defect, highlights
the importance of social norms in shaping human behavior. Norms can facilitate cooperative outcomes by providing a
shared understanding of acceptable behavior and allowing for the enforcement of rules through social sanctions.

In the context of AI alignment and non-zero-sum games, this underscores the importance of considering not only the
cognitive capacity of AI agents but also their understanding and adherence to social norms. The ability of AI systems to
adapt their behavior based on natural language prompts and to engage in reciprocal cooperation, e.g. Tit-for-Tat, is
crucial for creating AI agents that can better align with human values in complex, non-zero-sum settings.

By investigating the behavior of AI-generated agents in the iterated Prisoner's Dilemma and other social dilemmas
such as the ultimatum game we can contribute to a more comprehensive understanding of AI alignment in various
interaction scenarios. This knowledge can, in turn, inform the development of AI systems that are better equipped to
navigate the complexities of human cooperation and competition, while adhering to social norms and human values.

Researchers have argued that the prompt itself plays a crucial role in shaping
the emergent behaviour from the model; for example, the default "helpful
assistant" behavior of AI chatbots such as
GPT-3 has been noted to differ from that of specific simulacra instantiated by
user prompts (c.f. prompts used to "jail-break" GPT models) 
{cite}`0xk1g02023, Janus2023`. More generally, LLMs can be arbitrarily
scaffolded by injecting contextual information {cite}`Beren2023`. A particular
use-case of a scaffolded LLM involves injecting information about a 
world-state, together with a persona that incorporates specific goals, 
which can be used to instantiate autonomous agents, 
either in the real-world {cite}`Richards2023`, or in mult-agent 
simulations {cite}`Park2023`.

From an AI alignment perspective, the fact that large language models can easily
be scaffolded to deploy autonomous goal-oriented agents into production at very
little cost highlights the need to systematically evaluate the conditions in which LLM-instantiated
agents have a propensity or otherwise to cooperate in social dilemmas.

Given that the nature of an LLM agent depends on the persona and context
introduced in the initial 
prompt, a key question is to what extent the level of cooperation elicited
from the AI depends on features of the prompt.  In particular, we are
interested in whether large language models are capable of translating concepts
such as altruism and selfishness, as expressed in natural language, into
corresponding action policies in social dilemmas. This question
is important, as the ability
to operationalise these concepts in a variety of contexts would demonstrate 
the LLMs are capable of understanding and acting on cooperative norms 
that underpin human social behavior.  This is the key research question
investigated in our paper.
