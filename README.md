# Description

This is the source code developed for the first assignment of the course: "Machine Learning for the Built Environment",
at Delft University of Technology, The Netherlands. This program solves the programming-related, linear regression
exercises proposed in the description of the assignment. In brief, these exercises revolve around the modeling of the
trajectory of a quadrocopter in 3D space.

# Requirements

This program relies on Matplotlib and Numpy, both of which can be installed as such:

```batch
pip install -r requirements.txt
```

Note that the program uses a custom plotting style, and thus the installed Matplotlib version ***must*** be compatible
with
c.3.7.1. In case encounter any graphics-related problems while executing the program, please comment out all code
interacting with the matplotlib.rcParams dictionary and add restore the internal plotting configuration by including the
following command before the various function definitions:

```python
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
```

# Usage

The program can be executed as such:

```batch
python main.py
```

# Authors

Group 9

- Dennis Lagendijk
- Dimitrios Mantas
- Maria Luisa Tarozzo Kawasaki
