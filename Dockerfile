# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install NLTK data (if needed)
RUN python -m nltk.downloader punkt

# Expose the port Milvus is running on (adjust as per your setup)
EXPOSE 19530

# Run the specified script when the container launches
CMD [ "python", "./main.py" ]


