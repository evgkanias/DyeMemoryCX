# Dye Memory Central Complex Project

## Environment

In order to use this codem the required packages are listed below:
* Python >= 3.11
* NumPy >= 1.26.3
* SciPy >= 1.12
* Pandas >= 2.1.4
* Matplotlib >= 3.8.2
* Loguru >= 0.7.2
* PyYAML >= 6.0.1

## Installation

```commandline
pip install -r requirements.txt
```

```commandline
pip install .
```

## Run the experiments

To plat the curves shown in Fig. 4B, navigate to the main directory of the project and run:
```commandline
python . single_curve
```
This will plot the linear and exponential curves of the figure. If you want to save them without showing them, then run:
```commandline
python . single_curve --save data/plots -t SVG --show False
```
This command will save the plot in the ```data/plots``` directory, in SVG format. To create all the plots in
Supplementary Fig. S1, run:
```commandline
python . fit_curves --optimise True
```
The ```--optimise True``` is optional, and it runs the optimisation. If not used, the optimisation won't run, and the
pre-optimised parameters will be used for the plotting. To run all the simulations, type:
```commandline
python . run_simulation -c configs/dye-s2a.yaml
python . run_simulation -c configs/original.yaml
```
These commands will run the simulations for the original and the S2 sample after annealing. The different configuration
files in the [configs](configs) directory represent the set of parameters for the model and experimental set up. You can
create your own configuration files and test them, but they have to follow the same structure.
You can also show the results of the simulation by adding at the end of the command:
```-r last``` (path and responses of the last simulation run),
or ```-a last``` (summarised results over all the simulations run using the last configuration file).
To just plot the previously run simulations, run for example
```commandline
python . show_results -r data/stats/Original-010
python . show_results -a last --save data/plots -t PNG --show False
```
You can also run
```commandline
python . --help
```
to see a list of your options and explanations.

## Report an issue

If you have any issues installing or using the package, you can report it
[here](https://github.com/InsectRobotics/DyeMemoryCX/issues).

## Author

The code is written by [Evripidis Gkanias](https://evgkanias.github.io/).

## Copyright

Copyright &copy; 2024, Insect robotics Group, Institute of Perception,
Action and Behaviour, School of Informatics, the University of Edinburgh.