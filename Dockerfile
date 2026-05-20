# Use an official lightweight Python image
FROM python:3.10-slim

# Set system environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements first to leverage Docker cache layers
COPY requirements.txt .

# Install Python dependencies directly (FAISS 1.12.0 doesn't need C++ compiler on wheels)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py"]