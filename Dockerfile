# Use the official Ultralytics image (pre-installed with OpenCV & ML libs)
FROM ultralytics/ultralytics:latest-python

# Set working directory
WORKDIR /app

# We SKIP 'apt-get update' because this image already has everything!

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create the results directory
RUN mkdir -p static/results

EXPOSE 8001

# Run the application
CMD ["python", "main.py"]