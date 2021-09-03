# Deepracer Gym Environment

AWS Deepracer is a service hosted with AWS Robomaker platform. To make it easy for partcipants, we are releasing a gym environment for Deepracer. The simulator runs by starting a Docker container that runs the simulator and using a network connection with ZeroMQ server to provide a Gym interface.

Run these to quickly get started.

```bash
# Install docker if needed
sudo snap install docker

# Install the Deepracer Gym Environment
pip install -e ./deepracer-gym

# Start the Deepracer docker container
source deepracer-gym/start_deepracer_docker.sh
# This might take a while to download and start

# Wait until the terminal says "===Waiting for gym client==="

# Open a new terminal
# Run a random actions agent with Deepracer Gym
python deepracer-gym/random_actions_example.py

# Stop the docker container once done
source deepracer-gym/stop_deepracer_docker.sh
```

# Resources required

We recommend using 3 CPUs and 6 GB or RAM for the simulator.

If you want to reduce these, modify the flags `--cpus="3"` and `--memory="6g"` in `start_deepracer_docker.sh`.

