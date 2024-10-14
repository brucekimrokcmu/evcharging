# EV charging robot project 

This project implements a Contact Particle Filter for robotic applications using MuJoCo simulation.

## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have a Linux machine (tested on Ubuntu 20.04)
* You have Python 3.10 or later installed
* You have pip installed for managing Python packages

## Installation

1. Clone the repository:
```
git clone https://github.com/brucekimrokcmu/evcharging.git
cd evcharging
```

2. It's recommended to use a conda environment.
```
conda activate <YOUR CONDA ENV>
```

3. Install the required Python packages:
```
pip install -r requirements.txt
```

4. Project Structure
```
.
├── config.json
├── contact_particle_filter.py
├── controller.py
├── residual_observer.py
├── requirements.txt
├── test_contact_particle_filter_e2e.py
├── test_contact_particle_filter_unit.py
├── test_PID_control.py
├── test_residual_observer.py
├── test_trajectory_generator.py
├── trajectory_generator.py
└── visualization.py
```

## Running the Tests

1. To run the end-to-end test for the Contact Particle Filter:
```
python test_contact_particle_filter_e2e.py
```

2. To test the PID control:
```
python test_contact_particle_filter_unit.py
```

3. To test the residual observer:
```
python test_residual_observer.py
```

4. To test the trajectory generator:
```
python test_trajectory_generator.py

```
## Configuration

The `config.json` file contains various parameters for the simulation and algorithms. You can modify this file to adjust the behavior of the system.

## Visualization

The `visualization.py` file contains functions for visualizing the results of the simulation and the Contact Particle Filter. These visualizations are typically called from the test scripts.

## Contributing

If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## Contact

If you have any questions or feedback, please contact the repository owner.