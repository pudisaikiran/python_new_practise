# Stage 1: Build the React frontend
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Install dependencies first (for better caching)
COPY frontend/package*.json ./
RUN npm install

# Copy frontend source code and build it
COPY frontend/ ./
RUN npm run build


# Stage 2: Build the FastAPI backend and serve everything
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend source code
COPY . .

# Copy the compiled React files from the builder stage
# This ensures the image always has a fresh build, regardless of the host machine
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Expose the port Uvicorn runs on
EXPOSE 8002

# Run the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
