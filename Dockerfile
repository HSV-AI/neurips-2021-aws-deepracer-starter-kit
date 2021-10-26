FROM python:3.8
# Create user home directory
ENV USER_NAME aicrowd
ENV HOME_DIR /home/${USER_NAME}

# Replace HOST_UID/HOST_GUID with your user / group id
ENV HOST_UID 1001
ENV HOST_GID 1001

# Use bash as default shell, rather than sh
ENV SHELL /bin/bash

# Set up user
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${HOST_UID} \
    ${USER_NAME}

# Copy submission files to docker image
USER ${USER}
WORKDIR ${HOME_DIR}
COPY --chown=1001:1001 . ${HOME_DIR}

RUN apt-get update && apt-get install <apt.txt
RUN pip install -r requirements.txt