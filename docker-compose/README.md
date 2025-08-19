# MLflow Docker Compose Setup

This directory contains a Docker Compose configuration for running MLflow with PostgreSQL and MinIO. This guide will walk you through deploying MLflow locally using Docker Compose. We will use the docker-compose.yml file. This is the simplest way to run MLflow to give it a try.

## Getting Started

1. Copy the example environment file and adjust values as needed:

   ```bash
   cp .env.dev.example .env
   ```

   The environment file defines credentials, ports, and the MLflow version.

2. Start the services:

   ```bash
   docker compose up -d
   ```

   Docker Compose will read variables from the `.env` file in this directory.

3. Access MLflow at [http://localhost:5000](http://localhost:5000) or the port specified in your `.env` file.

4. To stop the stack:

   ```bash
   docker compose down
   ```