# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by OpenCV and other libraries
# This is the crucial step that fixes the build error
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Upgrade pip and install any needed wheel support
RUN pip install --upgrade pip

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Tell Docker that the container will listen on port 10000
# Render's default port for web services
EXPOSE 10000

# Command to run the application using a production-ready server
# This command is what starts your Flask app on Render
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "app:app"]
