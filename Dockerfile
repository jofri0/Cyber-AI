# Use official Python image
FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
