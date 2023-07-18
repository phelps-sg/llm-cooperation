# llm-cooperation

[![GitHub Workflow Status](https://github.com/phelps-sg/llm-cooperation/actions/workflows/tests.yaml/badge.svg)](https://github.com/phelps-sg/llm-cooperation/actions/workflows/tests.yaml)

This repo contains code, explanations and results of experiments to ascertain the propensity of large-language models
to cooperate in social dilemmas.  The experiments are described in:

S. Phelps and Y. I. Russell, *Investigating Emergent Goal-Like Behaviour in Large Language Models Using Experimental
Economics*, working paper, May 2023, [arXiv:2305.07970](https://arxiv.org/abs/2305.07970).

## Getting started

1. Install [mambaforge](https://github.com/conda-forge/miniforge#mambaforge).
2. In a shell:
~~~bash
export OPENAI_API_KEY='<my key>'
make install
make run
~~~

## Configuration

To run specific experiments and parameter combinations follow instructions below.

1. In a shell:

~~~bash
mkdir ~/.llm-cooperation
cat > ~/.llm-cooperation/llm_config.py << EOF

grid = {
        "temperature": [0.1, 0.6],
        "model": ["gpt-3.5-turbo", "gpt-4"],
        "max_tokens": [300]
}

sample_size = 3

experiments = ["dictator", "dilemma"]
EOF
~~~

2. Edit `$HOME/.llm-cooperation/llm_config.py` with required values.

3. In a shell:
~~~bash
export OPENAI_API_KEY='<key>'
make run
~~~


## Contributing

If you have a new experiment then please submit a pull request.
All code should have corresponding tests and all experiments
should be replicable.

