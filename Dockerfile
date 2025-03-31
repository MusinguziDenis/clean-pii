# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the necessary directories and files into the container
COPY appv2.py .
COPY clean/ clean/
COPY inference/ inference/
COPY phi_models/ phi_models/

# Expose the port the app runs on
EXPOSE 8080

# Define the command to run the application
CMD ["python", "appv2.py"]