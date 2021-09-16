![](https://i.imgur.com/BznomH1.png)
<h1 align="center"><a href="https://www.aicrowd.com/challenges/neurips-2021-aws-deepracer-ai-driving-olympics-challenge">NeurIPS 2021 - AWS Deepracer Challenge</a> - Starter Kit</h1>

This is the starter kit for the [AWS Deepracer Challenge](https://www.aicrowd.com/challenges/neurips-2021-aws-deepracer-ai-driving-olympics-challenge), a part of AI Driving Olympics at NeurIPS 2021, hosted on [AIcrowd](https://www.aicrowd.com).

In this competition, you will train a reinforcement learning agent (i.e. an autonomous car), to run on the deepracer simulator. This model will then be tested on a **real world track** with a miniature AWS Deepracer car.
Your goal is to train a model that can complete a lap as fast as possible without going off track, while avoiding crashing into the objects placed on the track.  

Clone the repository to compete now!

This repository contains:

- **Deepracer Gym Environment** which makes it easy to use the deepracer simulator.
- **Documentation** on how to submit your models to the leaderboard.
- Information on **best practices** to have hassle free submissions.
- **Starter code** for you to get started!

[IMPORTANT - Accept the rules before you submit](https://www.aicrowd.com/challenges/neurips-2021-aws-deepracer-ai-driving-olympics-challenge/challenge_rules)

# Table of contents

- [📚 Competition procedure](#-competition-procedure)
- [💪 Getting started](#-getting-started)
- [🏎 Deepracer Gym Environment](#-deepracer-gym-environment)
- [🛠 Preparing your submission](#-preparing-your-submission)
  * [Write your agents](#write-your-agents)
- [📨 Submission](#-submission)
  * [Repository Structure](#repository-structure)
  * [Runtime configuration](#runtime-configuration)
  * [🚀 Submitting to AIcrowd](#-submitting-to-aicrowd)
    + [`aicrowd.json`](#aicrowdjson)
    + [Configuring the submission repository](#configuring-the-submission-repository)
    + [Pushing the code to AIcrowd](#pushing-the-code-to-aicrowd)
- [📝 Submission checklist](#-submission-checklist)
- [📎 Important links](#-important-links)
- [✨ Contributors](#-contributors)

# 📚 Competition procedure

The AWS Deepracer Challenge is an opportunity for participants to test their agents for **simulation to real world transfer**, testing it on a **real world track** with a miniature AWS Deepracer car.
Your goal is to train a model that can complete a lap as fast as possible without going off track, while avoiding crashing into the objects placed on the track.  

In this challenge, you will train your models locally and then upload them to AIcrowd (via git) to be evaluated.

**The following is a high level description of how this process works.**

![Submission Flow](https://gitlab.aicrowd.com/deepracer/neurips-2021-aws-deepracer-starter-kit/-/raw/master/docs/submission_flow.gif)

1. **Sign up** to join the competition [on the AIcrowd website](https://www.aicrowd.com/challenges/neurips-2021-aws-deepracer-ai-driving-olympics-challenge).
2. **Clone** this repo and start developing your solution.
3. **Design and build** your agents that can compete in Deepracer environment and implement an agent class as described in [writing your agents](#write-your-agents) section.
4. [**Submit**](#-submission) your agents to [AIcrowd Gitlab](https://gitlab.aicrowd.com) for evaluation. [[Refer this for detailed instructions]](#-submission).

# 💪 Getting started

> We recommend using `python 3.6` or higher. If you are using Miniconda/Anaconda, you can install it using `conda install python=3.6`

Clone the starter kit repository and install the dependencies.

```bash
git clone http://gitlab.aicrowd.com/deepracer/neurips-2021-aws-deepracer-starter-kit.git

cd neurips-2021-aws-deepracer-starter-kit

#Optional: Install Deepracer Gym Environment
pip install -e ./deepracer-gym
```

# 🏎 Deepracer Gym Environment

Originally, AWS Deepracer is a service hosted with AWS Robomaker platform. To make it easy for partcipants, we are releasing a gym environment for Deepracer. The simulator runs by starting a Docker container that runs the simulator and using a network connection with ZeroMQ server to provide a Gym interface. 

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

For more instructions look at [`deepracer-gym/README.md`](`deepracer-gym/README.md`)

# 🛠 Preparing your submission

## Write your agents

Your agents need to implement a subclass of [`DeepracerAgent`](agents/deepracer_base_agent.py#L1) class from [`agents/deepracer_base_agent.py`](agents/deepracer_base_agent.py). You can check the code in [`agents`](agents) directory for examples.

**Note:** If your agent doesn't inherit the `DeepracerAgent` class, the evaluation will fail.

Once your agent class is ready, you can specify the class to use as the player agent in your [`submission_config.py`](submission_config.py). The starter kit comes with a random agent submission. The [`submission_config.py`](submission_config.py) in the starter kit points to this class. You should update it to use your class.

# 📨 Submission

## Repository Structure

**File/Directory** | **Description**
--- | ---
[`agents`](agents) | Directory containing different scripted bots, baseline agent and bots performing random actions. We recommend that you add your agents to this directory.
[`submission_config.py`](config.py) | File containing the configuration options for local evaluation. We will use the same player agent you specify here during the evaluation.
[`utils/submit.sh`](utils/submit.sh) | Helper script to submit your repository to [AIcrowd GitLab](https://gitlab.aicrowd.com).
[`Dockerfile`](Dockerfile) | (Optional) Docker config for your submission. Refer [runtime configuration](#runtime-configuration) for more information.
[`requirements.txt`](requirements.txt) | File containing the list of python packages you want to install for the submission to run. Refer [runtime configuration](#runtime-configuration) for more information.
[`apt.txt`](apt.txt) | File containing the list of packages you want to install for submission to run. Refer [runtime configuration](#runtime-configuration) for more information.

## Runtime configuration

You can specify the list of python packages needed for your code to run in your [`requirements.txt`](requirements.txt) file. We will install the packages using `pip install` command.

You can also specify the OS packages needed using [`apt.txt`](apt.txt) file. We install these packages using `apt-get install` command.

For more information on how you can configure the evaluation runtime, please refer [`RUNTIME.md`](docs/RUNTIME.md).

## 🚀 Submitting to AIcrowd

### **Add your SSH key** to AIcrowd GitLab

You can add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/profile/keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).


### `aicrowd.json`

Your repository should have an `aicrowd.json` file with following fields:

```json
{
    "challenge_id" : "neurips-2021-aws-deepracer-ai-driving-olympics-challenge",
    "authors" : ["Your Name"],
    "description" : "Brief description for your submission"
}
```

This file is used to identify your submission as a part of the AWS Deepracer Challenge. You must use the `challenge_id` as specified above.

### Configuring the submission repository

```bash
git remote add aicrowd git@gitlab.aicrowd.com:<username>/neurips-2021-aws-deepracer-starter-kit.git
```
**Note**: The above step needs to be done only once. This configuration will be saved in your repository for future use.

### Pushing the code to AIcrowd for evaluation

```bash
./utils/submit.sh "some description"
```

If you want to submit without the helper script, please refer [`SUBMISSION.md`](docs/SUBMISSION.md).


# 📝 Submission checklist

- [x] **Accept the challenge rules**. You can do this by going to the [challenge overview page](https://www.aicrowd.com/challenges/neurips-2021-aws-deepracer-ai-driving-olympics-challenge) and clicking the "Participate" button. You only need to do this once.
- [x] **Add your agent code** that implements the `DeepracerAgent` class from `evaluator/base_agent`. 
- [x] **Add your model checkpoints** (if any) to the repo. The `utils/submit.sh` will automatically detect large files and add them to git LFS. If you are using the script, please refer to [this post explaining how to add your models](https://discourse.aicrowd.com/t/how-to-upload-large-files-size-to-your-submission/2304).
- [x] **Update runtime configuration** using `requirements.txt`, `apt.txt` and/or `Dockerfile` as necessary. Please make sure that you specified the same package versions that you use locally on your machine.

# 📎 Important links

- 💪 Challenge information
   * [Challenge page](https://www.aicrowd.com/challenges/neurips-2021-aws-deepracer-ai-driving-olympics-challenge)
   * [Leaderboard](https://www.aicrowd.com/challenges/neurips-2021-aws-deepracer-ai-driving-olympics-challenge/leaderboards)
 - 🗣 Community
    * [Challenge discussion forum](https://www.aicrowd.com/challenges/neurips-2021-aws-deepracer-ai-driving-olympics-challenge/discussion)
- 🎮 Deepracer resources
    * [Deepracer documentation](https://docs.aws.amazon.com/deepracer/latest/developerguide/what-is-deepracer.html)
    * [Deepracer homepage](https://aws.amazon.com/deepracer/)
    * [ICRA Paper](https://ieeexplore.ieee.org/document/9197465)
    

# ✨ Contributors

- [Dipam Chakraborty](https://www.aicrowd.com/participants/dipam)
- [Siddhartha Laghuvarapu](https://www.aicrowd.com/participants/siddhartha)
- [Jyotish Poonganam](https://www.aicrowd.com/participants/jyotish)
- Sahika Genc


**Best of Luck** 🎉 
