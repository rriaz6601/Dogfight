# Deep RL for autonomous air combat

Codebase for high fidelity environment for reinforcement learning training.

**Paper:** [Curriculum Based Deep RL for Autonomous Air Combat](docs/paper.pdf)

**Talk:** I am making a video soon

**Videos:** [Video of agent playing against a deterministic opponent](https://www.youtube.com/watch?v=_d810Ynsc5g)

## Recent releases

##### v0.1.0

First version, this is a research code and will need some cleaning to be used better.
However still a good starting point for someone starting to work in multiagent aircraft
control.

## Installation

This project uses [poetry dependency management](https://python-poetry.org/). Once you
have properly configured poetry just do

`poetry install`

Then you can activate the virtual environment using:

`poetry shell`

### Environment support

The main focus of this work is the aircraft environment.

Visualisation is also an important part of simulation. For this purpose please copy the
provided plugins in the folder to X-Plane plugin folder, then various scripts use it as
visualiser. [X-Plane](https://www.x-plane.com/) is required to visualise. This project
wanted a computationally efficient high-fidelity simulator, so chose
[JSBSim](https://github.com/JSBSim-Team/jsbsim) but that meant that now we needed another
system to visualise. X-Plane is a good choice for that.

## Running experiments

Please look at the scripts folder, it has various scripts for training online and
offline RL.

| Script Name            | Description                                               |
| ---------------------- | --------------------------------------------------------- |
| train\_gym\_env        | Train in an online RL setting asynchronously              |
| debug\_dogfight        | Run online training in synchorous mode with visualisation |
| enjoy\_gym\_env        | Visualising the online agent                              |
| human\_actor           | collect data using human                                  |
| dataset\_preprocessing | read csv data and then clean it to store as MDP dataset   |
| train_d3rl             | Train in an offline setting                               |
| enjoy_d3rl             | visualise the agent trained offline                       |



### Monitoring training sessions

This project uses Tensorboard summaries. Run Tensorboard to monitor your experiment:

`tensorboard --logdir=train_dir --port=6006`

#### WandB support

This project also supports experiment monitoring with Weights and Biases. In order to
setup WandB locally run `wandb login` in the terminal
(https://docs.wandb.ai/quickstart#1.-set-up-wandb)

## Citation

If my paper gets accepted I will add the citation here.

For questions, issues, inquiries please email usamar240@gmail.com. Github issues and
pull requests are welcome.
