# Conclusion and future work

As can be seen from  <a href="Results.html#figure-1-cooperation-frequency-by-group">Figure 1</a>, without
having to resort to statistical tests, our results provide
clear support for {ref}`hypotheses` 1 through 3, demonstrating that simulacra instantiated with
cooperative, competitive, altruistic, and self-interested prompts exhibit distinct levels of cooperation in the iterated
Prisoner's Dilemma. This indicates that LLMs can operationalize natural language descriptions of cooperative and
competitive behavior to some extent. However, the remaining hypotheses were not supported, suggesting a more complex
relationship between prompt content and emergent behavior in LLM-generated agents.

Interestingly, the simulacrum from the control group instantiated with "you are a participant in a psychology
experiment" (see <a href="Results.html#table-2-cooperation-frequency-by-participant-and-condition">Table 2</a>)
exhibited behavior
more closely aligned with how real people tend to act in iterated Prisoner's Dilemmas, suggesting that GPT-3.5 possesses
some knowledge about human behavior in such contexts. However, when combined with the other results, it appears that the
LLM struggles to generalize this behavior in a nuanced way beyond a superficial ability to cooperate more or less
depending on whether the role description is altruistic or selfish.
The unexpected pattern of increased cooperation with defectors and decreased cooperation with cooperators challenges our initial hypotheses and highlights a potential limitation in the LLM’s ability to translate altruism or selfishness into strategies based on conditioned reciprocity. This result suggests that while the agents are sensitive to the general cooperative or competitive nature of the prompts, their capacity to effectively adapt their behavior to their partner’s actions might be more limited.

Another potential limitation of the study is that the LLM has been exposed to
a vast literature on the iterated Prisoner's Dilemma in its training data,
and it is unclear how would it perform in more ecologically valid task environments
that it has no prior exposure to.  This limitation could be addressed by
inventing new social dilemma games with corresponding task descriptions
which are not vignettes from the existing literature.

Recognizing these limitations, we call upon the research community to further investigate the factors contributing to
the emergent behavior of LLM-generated agents in social dilemmas, both within and beyond the Prisoner's Dilemma. This
broader research program could involve exploring the potential for more refined or complex prompts to elicit a wider
range of cooperative behaviors in various experimental economics scenarios, such as the ultimatum game, the dictator
game, and the public goods game, among others. Examining the role of model architecture and training parameters in
shaping agent behaviors, as well as analyzing the impact of various partner strategies on agent behavior in these
different contexts, could shed light on the model's adaptability and alignment with human values.

In future studies, it would be valuable to examine other parameter settings, such as temperature, to explore their
effects on the emergent behavior of LLM-generated agents. Additionally, as more advanced LLMs like GPT-4 become
available, it would be interesting to investigate whether they exhibit similar limitations or are capable of more
nuanced cooperative behaviors in a wider array of social dilemmas. 

By addressing these questions, we hope to
collectively build a deeper understanding of AI alignment in the context of complex, non-zero-sum interactions across
various experimental economics settings, ultimately fostering the development of AI systems that better adhere to human
values and social norms.