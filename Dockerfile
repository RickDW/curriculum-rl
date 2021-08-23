# For more information, please refer to https://aka.ms/vscode-docker-python
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Fix error caused by libgthread-2.0 not being available
RUN apt-get update && apt-get install -y libglib2.0-0

# Install git
RUN apt-get install -y git

WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
# USER appuser

# Install pip requirements
COPY requirements.txt .
RUN pip install -r requirements.txt && pip cache purge
# Purging the cache has a minor effect on the image size, but every little bit helps

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python"]
