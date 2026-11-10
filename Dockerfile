# Base image — slim Python 3.11
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create results directory in case it doesn't exist
RUN mkdir -p results

# Default command: run the full 60-trial experiment
CMD ["python", "run_experiments_adaptive.py"]
