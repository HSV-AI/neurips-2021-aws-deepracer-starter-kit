#!/bin/bash

set -e

REPO_ROOT_DIR=$(git rev-parse --show-toplevel)
SCRIPTS_DIR="${REPO_ROOT_DIR}/utils"

source "${SCRIPTS_DIR}/logging.sh"


print_usage() {
cat << USAGE
Usage: ./utils/submit.sh "impala-ppo-v0.1"
USAGE
}


bad_remote_message() {
  log_normal "AIcrowd remote not found"
  log_error "Please run \`git remote add aicrowd git@gitlab.aicrowd.com:<username>/<repo>.git\` and rerun this command"
  exit 1
}

get_submission_remote() {
  bad_remotes=(
    git@gitlab.aicrowd.com:deepracer/neurips-2021-aws-deepracer-starter-kit.git
    http://gitlab.aicrowd.com/deepracer/neurips-2021-aws-deepracer-starter-kit.git
  )
  submission_remote=""

  for remote in $(git remote); do
    remote=$(git remote get-url $remote)
    if [[ ! "$remote" =~ "$bad_remotes" ]] && echo $remote | grep "gitlab.aicrowd.com" > /dev/null; then
      submission_remote=$remote
    fi
  done

  if [[ "$submission_remote" == "" ]]; then
    bad_remote_message
  fi

  echo $submission_remote
}

check_remote() {
  log_info Checking git remote settings...
  log_normal Using $(get_submission_remote) as the submission repository
}


setup_lfs() {
  git lfs install
  HTTPS_REMOTE=$(git remote -v | grep gitlab.aicrowd.com | head -1 | awk '{print $2}' | sed 's|git@gitlab.aicrowd.com:|https://gitlab.aicrowd.com|g')
  git config lfs.$HTTPS_REMOTE/info/lfs.locksverify false
  find . -type f -size +5M -exec git lfs track {} &> /dev/null \;
  git add .gitattributes
}


setup_commits() {
  REMOTE=$(get_submission_remote)
  TAG=$(echo "$@" | sed 's/ /-/g')
  git add --all
  git commit -m "Changes for submission-$TAG" || true  # don't exit when no new commits are there
  git tag -am "submission-$TAG" "submission-$TAG" || (log_error "There is another submission with the same description. Please give a different description." && exit 1)
  git push -f $REMOTE master
  git push -f $REMOTE "submission-$TAG"
}


submit() {
  check_remote
  setup_lfs "$@"
  setup_commits "$@"
}
  


if [[ $# -lt 1 ]]; then
  print_usage
  exit 1
fi

submit "$@"
