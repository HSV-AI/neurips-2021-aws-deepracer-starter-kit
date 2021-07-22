![AIcrowd-Logo](https://raw.githubusercontent.com/AIcrowd/AIcrowd/master/app/assets/images/misc/aicrowd-horizontal.png)

<h1 align="center"><a href="https://www.aicrowd.com/challenges/neurips-2021-aws-deepracer-ai-driving-olympics-challenge">NeurIPS 2021- AWS Deepracer AI Driving Olympics Challenge</a> - Starter Kit</h1>



[![Discord](https://img.shields.io/discord/565639094860775436.svg)](https://discord.com)


This repository is the AWS Deepracer AI Driving Olympics Challenge **Submission template and Starter kit**! 

AWS DeepRacer is an AWS Machine Learning service for exploring reinforcement learning that is focused on autonomous racing.
In this competition, you will train a reinforcement learning agent (i.e. an autonomous car), that learns to drive by interacting with its environment, e.g., the track, by taking an action in a given state to maximize the expected reward. 
Your goal is to train a model that can complete a lap as fast as possible without going off track, while avoiding crashing into the objects placed on the track.  

Clone the repository to compete now!

**This repository contains**:
*  **Documentation** on how to submit your agent to the leaderboard
*  **The procedure** for best practices and information on how we evaluate your agent, etc.
*  **Starter code** for you to get started!
*  **Baseline**: Baseline Models

[IMPORTANT - Accept the rules before you submit](https://www.aicrowd.com/challenges/neurips-2021-aws-deepracer-ai-driving-olympics-challenge/challenge_rules)


# Table of Contents

- [📚 Competition Procedure](#competition-procedure)
- [💪 Setup](#how-to-access-and-use-dataset)
- [🛠 Specify software runtime / dependencies?](#how-do-i-specify-my-software-runtime-dependencies-)
- [🚀 Making a submission](#how-to-make-submission)
- [🤔 Other concepts and FAQs](#other-concepts)
- [📎 Important links](#-important-links)


## 📚  Competition Procedure


**The following is a high level description of how this round works**

![](https://i.imgur.com/xzQkwKV.jpg)

1. **Sign up** to join the competition [on the AIcrowd website].(https://www.aicrowd.com/challenges/neurips-2021-aws-deepracer-ai-driving-olympics-challenge)
2. **Clone** this repo and start developing your solution.
3. **Train** your models and writer code in `run.py`.
4. [**Submit**](#how-to-submit-a-model) your trained models to [AIcrowd Gitlab](https://gitlab.aicrowd.com) for evaluation [(full instructions below)](#how-to-submit-a-model). The automated evaluation setup will evaluate the submissions against the test dataset to compute and report the metrics on the leaderboard of the competition.


## 💪 Setup


2.  **Clone the repository**

    ```
    git clone git@gitlab.aicrowd.com:neurips-2021-aws-deepracer-ai-driving-olympics-challenge/neurips-2021-aws-deepracer-ai-driving-olympics-challenge-starter-kit.git
    ```

3. **Install** competition specific dependencies!
    ```
    cd neurips-2021-aws-deepracer-ai-driving-olympics-challenge-starter-kit
    pip3 install -r requirements.txt
    ```

4. Try out the baseline model available in `run.py`.





## 🚀 Making a submission

### Repository structure

Please follow the example structure as it is in the starter kit for the code structure.
The different files and directories have following meaning:

```
.
├── aicrowd.json           # Submission meta information - like your username
├── apt.txt                # Packages to be installed inside docker image
├── data                   # Your local dataset copy - you don't need to upload it (read DATASET.md)
├── requirements.txt       # Python packages to be installed
├── run.py                # IMPORTANT: Your testing/inference phase code, must be derived from AirbornePredictor (example in test.py)
```

### Specify runtime/dependencies

We accept submissions with custom runtime, so you don't need to worry about which libraries or framework to pick from.

The configuration files typically include `requirements.txt` (pypi packages), `environment.yml` (conda environment), `apt.txt` (apt packages) or even your own `Dockerfile`.

You can check detailed information about the same in the 👉 [RUNTIME.md](/docs/RUNTIME.md) file.

### Submitting to aicrowd

- **Add your SSH key** to AIcrowd GitLab

You can add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/profile/keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).

- Your repository should have an aicrowd.json file with following fields:

```json
{
  "challenge_id": "evaluations-api-deepracer",
  "grader_id": "evaluations-api-deepracer",
  "authors": ["aicrowd-bot"],
  "tags": "change-me",
  "description": "Random agent for AWS Deep Racer",
}
```


This JSON is used to map your submission to the challenge - so please remember to use the correct `challenge_id` as specified above.

- Follow the instructions in [SUBMISSION.md](/docs/SUBMISSION.md) to get your submission evaluated.


# 🤔 Other Concepts

## Time constraints

You need to make sure that your model finishes evaluation in 1500 seconds, otherwise your evaluation will be marked failed.

## Local evaluation

You can also test end to end evaluation on your own systems, by executing `run.py`.

## Hardware used for evaluations

We use g4dn instances to run your evaluations.



# 📎 Important links


💪 &nbsp;Challenge Page: https://www.aicrowd.com/challenges/neurips-2021-aws-deepracer-ai-driving-olympics-challenge

🗣️ &nbsp;Discussion Forum: https://www.aicrowd.com/challengesneurips-2021-aws-deepracer-ai-driving-olympics-challengee/discussion

🏆 &nbsp;Leaderboard: https://www.aicrowd.com/challenges/neurips-2021-aws-deepracer-ai-driving-olympics-challenge/leaderboards
