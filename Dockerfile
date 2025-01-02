FROM python:3.8-slim

# Set ARG and ENV for non-interactive installations and Python unbuffered mode
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    APP_HOME=/code

# Create and set permissions for the application directory
RUN mkdir -p $APP_HOME && \
    chown -R 1000:1000 $APP_HOME

# Install essential system packages and Python development tools
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        python3-venv \
        python3-dev \
        software-properties-common \
        pkg-config \
        libhdf5-dev \
        libhdf5-serial-dev \
        build-essential \
        git \
        ffmpeg \
        libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR $APP_HOME

# Clone the repository
RUN git clone https://github.com/kadabengarann/face-verify-api.git $APP_HOME

# Create a non-root user and set ownership
RUN useradd -m -u 1000 user && \
    chown -R user:user $APP_HOME

# Switch to non-root user
USER user

# Set environment variables for the user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=$APP_HOME \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_THEME=huggingface

# Install Python dependencies
RUN python3.8 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3.8 -m pip install --no-cache-dir -r requirements.txt && \
    python3.8 -m pip install --no-cache-dir uvicorn

# Expose the Uvicorn FastAPI port
EXPOSE 7860

# Start the application
CMD cd $APP_HOME && \
    git pull origin main && \
    python3.8 -m uvicorn app:app --host 0.0.0.0 --port 7860 --reload --log-level info