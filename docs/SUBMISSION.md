# Making submission

This file will help you in making your first submission.


## Submission Entrypoint (where you write your code!)

The evaluator will import the `submission_agent` from `submission_config.py` for generating actions in the deepracer environment. 

## IMPORTANT: Saving Models before submission!

Before you submit make sure that you have saved your models, which are needed by your inference code.
In case your files are larger in size you can use `git-lfs` to upload them. More details [here](https://discourse.aicrowd.com/t/how-to-upload-large-files-size-to-your-submission/2304).

## How to submit a trained model!

To make a submission, you will have to create a **private** repository on [https://gitlab.aicrowd.com/](https://gitlab.aicrowd.com/).

You will have to add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/profile/keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).

Then you can create a submission by making a _tag push_ to your repository on [https://gitlab.aicrowd.com/](https://gitlab.aicrowd.com/).
**Any tag push (where the tag name begins with "submission-") to your private repository is considered as a submission**  
Then you can add the correct git remote, and finally submit by doing :

```
cd neurips-2021-aws-deepracer-starter-kit
# Add AIcrowd git remote endpoint
git remote add aicrowd git@gitlab.aicrowd.com:<username>/neurips-2021-aws-deepracer-starter-kit.git
git push aicrowd master
```

```
# Create a tag for your submission and push
git tag -am "submission-v0.1" submission-v0.1
git push aicrowd master
git push aicrowd submission-v0.1

# Note : If the contents of your repository (latest commit hash) does not change,
# then pushing a new tag will **not** trigger a new evaluation.
```

You now should be able to see the details of your submission at :
[gitlab.aicrowd.com/<YOUR_AICROWD_USER_NAME>/neurips-2021-aws-deepracer-starter-kit/issues](https://gitlab.aicrowd.com//<YOUR_AICROWD_USER_NAME>/neurips-2021-aws-deepracer-starter-kit/issues)

**NOTE**: Remember to update your username instead of `<YOUR_AICROWD_USER_NAME>` above.

### Other helpful files

👉 [RUNTIME.md](/docs/RUNTIME.md)
