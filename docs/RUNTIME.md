# Adding your runtime

## **Installation setup steps**

### **How to specify your installation setup for the submission**
The entrypoint to the installation is the Dockerfile
The Dockerfile provided will have commands to install apt packages from apt.txt and pip packages from requirements.txt
You are strongly advised to specify the version of the library that you use to use for your submission

Examples

For requirements.txt

```torch==1.8.1```

You can add git repositories to requirements.txt as well

```git+https://github.com/tqdm/tqdm.git@devel#egg=tqdm```

For apt.txt

```firefox=45.0.2+build1-0ubuntu1```

## (Optional) Check your installation setup on your own machine

### Setup the environment 
Install docker on your machine and run

```bash
docker build -t your-custom-tag . 
```

If you get installation errors during the above step, your submission is likely to fail, please review the errors and fix the installation

### Run the submission locally

Run the docker container This will create an environment that emulates how your submission environment will be.

```bash
docker run -it your-custom-tag /bin/bash
```

```bash
python rollout.py 
```

If you get runtime errors during the above step, your submission is likely to fail. Please review the errors.
A common error is not specifying one or multiple required libraries

## Installation FAQ
- How to install with an `environment.yml`?

  - Add `environment.yml` to the base of your repo. You also need to add commands to add `environment.yml` to the `Dockerfile`. Afterwards, You’re encouraged to follow the above steps Check your installation setup on your own machine to check everything is properly installed.

- How do I install with a setup.py
  - You need to add the command to run it in the ```Dockerfile``` - ```RUN pip install .```

- What’s the package versions I have installed on my machine?

  - You can find the versions of the python package installations you currently have using pip freeze

- What’s aicrowd_gym
  - AIcrowd gym is a library AIcrowd uses to limit the functionality on the submission environment to prevent tampering with the RL Library. For all intents and purposes, aicrowd_gym should work the same as gym for you.
- How do I use a custom script to install?
  - You’ll need to call the custom script in the ```Dockerfile```, example you can add this line ```RUN ./custom_script.sh```