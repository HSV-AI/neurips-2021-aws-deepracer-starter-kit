![AIcrowd-Logo](https://raw.githubusercontent.com/AIcrowd/AIcrowd/master/app/assets/images/misc/aicrowd-horizontal.png)

# NeurIPS 2021- AWS Deepracer AI Driving Olympics Challenge - Starter Kit

👉 [Challenge page](https://www.aicrowd.com/challenges/neurips-2021-aws-deepracer-ai-driving-olympics-challenge)

[![Discord](https://img.shields.io/discord/565639094860775436.svg)](https://discord.com)


This repository is the main AWS Deepracer AI Driving Olympics Challenge **Submission template and Starter kit**! 

The AI Driving Olympics (AI-DO) is a series of embodied intelligence competitions in the field of autonomous vehicles.
The overall objective of the AI-DO is to provide accessible mechanisms for benchmarking progress in autonomy applied to the task of autonomous driving

Clone the repository to compete now!

**This repository contains**:
*  **Documentation** on how to submit your agent to the leaderboard
*  **The procedure** for best practices and information on how we evaluate your agent, etc.
*  **Starter code** for you to get started!
*  **Baseline**: Baseline Models


# Table of Contents

1. [Competition Procedure](#competition-procedure)
2. [How to access and use dataset](#how-to-access-and-use-dataset)
3. [How to start participating](#how-to-start-participating)
4. [How do I specify my software runtime / dependencies?](#how-do-i-specify-my-software-runtime-dependencies-)
5. [What should my code structure be like ?](#what-should-my-code-structure-be-like-)
6. [How to make submission](#how-to-make-submission)
7. [:star: SiamMOT baseline](#submit-siammot-baseline)
8. [Other concepts and FAQs](#other-concepts)
9. [Important links](#-important-links)


#  Competition Procedure


**The following is a high level description of how this round works**

![](https://i.imgur.com/xzQkwKV.jpg)

1. **Sign up** to join the competition [on the AIcrowd website].(https://www.aicrowd.com/challenges/airborne-object-tracking-challenge)
2. **Clone** this repo and start developing your solution.
3. **Train** your models and writer code in `run.py`.
4. [**Submit**](#how-to-submit-a-model) your trained models to [AIcrowd Gitlab](https://gitlab.aicrowd.com) for evaluation [(full instructions below)](#how-to-submit-a-model). The automated evaluation setup will evaluate the submissions against the test dataset to compute and report the metrics on the leaderboard of the competition.

# How to setup the environment



# How to start participating

## Setup

1. **Add your SSH key** to AIcrowd GitLab

You can add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/profile/keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).

2.  **Clone the repository**

    ```
    git clone git@gitlab.aicrowd.com:amazon-prime-air/airborne-detection-starter-kit.git
    ```

3. **Install** competition specific dependencies!
    ```
    cd airborne-detection-starter-kit
    pip3 install -r requirements.txt
    ```

4. Try out the baseline model available in `run.py`.


## How do I specify my software runtime / dependencies ?

We accept submissions with custom runtime, so you don't need to worry about which libraries or framework to pick from.

The configuration files typically include `requirements.txt` (pypi packages), `environment.yml` (conda environment), `apt.txt` (apt packages) or even your own `Dockerfile`.

You can check detailed information about the same in the 👉 [RUNTIME.md](/docs/RUNTIME.md) file.

## What should my code structure be like ?

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

Finally, **you must specify an AIcrowd submission JSON in `aicrowd.json` to be scored!** 

The `aicrowd.json` of each submission should contain the following content:

```json
{
  "challenge_id": "evaluations-api-airborne",
  "grader_id": "evaluations-api-airborne",
  "authors": ["aicrowd-bot"],
  "tags": "change-me",
  "description": "Random prediction model for Airborne challenge",
}
```

This JSON is used to map your submission to the challenge - so please remember to use the correct `challenge_id` as specified above.

Please specify if your code will use a GPU or not for the evaluation of your model. If you specify `true` for the GPU, GPU will be provided and used for the evaluation.

## How to make submission

👉 [SUBMISSION.md](/docs/SUBMISSION.md)

**Best of Luck** :tada: :tada:


# Other Concepts

## Time constraints

You need to make sure that your model finishes evaluation in 1500 seconds, otherwise your evaluation will be marked failed.

## Local evaluation

You can also test end to end evaluation on your own systems.

## Hardware used for evaluations

We use g4dn instances to run your evaluations.



# 📎 Important links


💪 &nbsp;Challenge Page: https://www.aicrowd.com/challenges/neurips-2021-aws-deepracer-ai-driving-olympics-challenge

🗣️ &nbsp;Discussion Forum: https://www.aicrowd.com/challengesneurips-2021-aws-deepracer-ai-driving-olympics-challengee/discussion

🏆 &nbsp;Leaderboard: https://www.aicrowd.com/challenges/airborne-object-tracking-challengeneurips-2021-aws-deepracer-ai-driving-olympics-challenge/leaderboards
