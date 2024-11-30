# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install ffprobe
RUN apt update && apt install -y ffmpeg

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port the application will run on
EXPOSE 5000

# Run the application when the container launches
CMD ["flask", "run", "--host=0.0.0.0"]