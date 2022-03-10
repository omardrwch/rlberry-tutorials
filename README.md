# rlberry-tutorials

Tutorials about how to use the [rlberry](https://github.com/rlberry-py/rlberry) library.

Check out our [introductory slides](https://docs.google.com/presentation/d/1TrepWrm2TMgSBkCS5OMFZvcoMcJiqBZ7AxwlP7DJ9hc/edit?usp=sharing)!

# Setup

1. Create a virtual environment and clone this repository

```bash
conda create -n rlberry-tutorial python=3.8
conda activate rlberry-tutorial
git clone https://github.com/omardrwch/rlberry-tutorials.git
cd rlberry-tutorials
```

2. Install rlberry

```bash
pip install git+https://github.com/rlberry-py/rlberry.git@main#egg=rlberry[torch_agents]
pip install tensorboard
pip install ipython
```

