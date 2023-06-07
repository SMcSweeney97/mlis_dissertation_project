# Stat Mech Gym
This repository contains gym implementations of models used in non-equilibrium statistical mechanics and studies of dynamics large deviations.

## Overview

See `src` (pending module name)

This package contains three parts, the environments (src.envs), environment wrappers (src.wrappers) and utilities (src.utils)

## Requirements
On linux (tested on Ubuntu 18.04.4 LTS)

Using conda:

```bash
conda create --name test_env
conda activate test_env
conda install pip
pip install -r requirements.txt
```

To remove the environment:
```bash
conda deactivate
conda env remove -n test_env
```