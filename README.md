# Continuous Mountain Car in 3D
* The extension of OpenAI Gym [Continuous Mountain Car](https://gym.openai.com/envs/MountainCarContinuous-v0/) to 3D.
* Two extensions are available:
  1. The Y-dimension has another sinusoidal curve, similar to the 3D extension 
  of [Taylor, Matthew E., Gregory Kuhlmann, and Peter Stone. "Autonomous transfer for reinforcement learning." AAMAS (1). 2008.](https://www.cs.utexas.edu/~ai-lab/pubs/AAMAS08-taylor.pdf)
  2. The Y-dimension has no curve. 
* The rendering is still in 2D. Currently, there are no plans to 
extend the rendering to 3D.

* If you use our code, please cite our [paper](https://arxiv.org/abs/2305.05666): 

```bib
@article{panangaden2023policy,
  title={Policy Gradient Methods in the Presence of Symmetries and State Abstractions},
  author={Panangaden, Prakash and Rezaei-Shoshtari, Sahand and Zhao, Rosie and Meger, David and Precup, Doina},
  journal={arXiv preprint arXiv:2305.05666},
  year={2023}
}
```

## Setup
* Simply clone the repo and install it with `pip install -e .`. The environment is registered in the OpenAI Gym Registry. 
* A demo script is available [here](scripts/test_env.py).
