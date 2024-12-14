FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set ARG and ENV for non-interactive installations and Python unbuffered mode
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install essential system packages
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python3.8 \
    python3.8-venv \
    python3-pip \
    git \
    ffmpeg \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

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
    PYTHONUNBUFFERED=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_THEME=huggingface

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir --upgrade -r /code/requirements.txt

# Set the application directory for the user
WORKDIR $HOME/app

# Copy the application code to the container
COPY --chown=user . $HOME/app

# Expose the Uvicorn FastAPI port
EXPOSE 7860

# Run the application using the same command as local setup
CMD ["python3.8", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--reload"]
