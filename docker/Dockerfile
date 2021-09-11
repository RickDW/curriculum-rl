# syntax=docker/dockerfile:1

# For more information, please refer to https://aka.ms/vscode-docker-python
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Fix error caused by libgthread-2.0 not being available
# and install additional software
RUN apt-get update && apt-get install -y libglib2.0-0 git xvfb python3-opengl

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
# USER appuser

# Install pip requirements
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
# TODO: don't pip install as root

# set up the xvfb entrypoint
# COPY docker/entrypoint.sh /usr/local/bin
# RUN chmod 777 /usr/local/bin/entrypoint.sh
# ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

COPY docker/xvfb_python.sh /usr/local/bin/xvfb_python.sh

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python"]
