FROM ubuntu:20.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-casacore \
    casacore-tools \
    && rm -rf /var/lib/apt/lists/*
    

# Set the working directory
WORKDIR /corrupt_visibilities_app

# Install Python dependencies
COPY requirements.txt /corrupt_visibilities_app/requirements.txt
RUN pip3 install --no-cache-dir -r /corrupt_visibilities_app/requirements.txt

# Copy the app script
COPY corrupt.py /corrupt_visibilities_app/corrupt.py

# Entry point
ENTRYPOINT ["python3", "/corrupt_visibilities_app/corrupt.py"]
