FROM python:3.8-slim

# Set ARG and ENV for non-interactive installations and Python unbuffered mode
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install essential system packages and Python development tools
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        python3-dev \
        software-properties-common \
        pkg-config \
        libhdf5-dev \
        libhdf5-serial-dev \
        build-essential \
        git \
        ffmpeg \
        libglib2.0-0 \
        cmake && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /code

# Copy requirements and install them
COPY ./requirements.txt /code/requirements.txt

# Create a non-root user and switch to it
RUN useradd -m -u 1000 user
USER user

# Set environment variables for the user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=$HOME/app \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_THEME=huggingface

# Check disk space before installing dependencies
RUN df -h

# Upgrade pip and install dependencies with binary wheels
RUN python3.8 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3.8 -m pip install --no-cache-dir -r /code/requirements.txt && \
    python3.8 -m pip install --no-cache-dir uvicorn

# Set the application directory for the user
WORKDIR $HOME/app

# Copy the application code to the container
COPY --chown=user . $HOME/app

# Expose the Uvicorn FastAPI port
EXPOSE 80

# Run the application
CMD ["python3.8", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80", "--reload", "--log-level", "info"]
