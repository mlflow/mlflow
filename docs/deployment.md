# Genesis-Flow Deployment Guide

This comprehensive guide covers deploying Genesis-Flow in production environments with enterprise-grade security, scalability, and reliability.

## Table of Contents

1. [Pre-Deployment Planning](#pre-deployment-planning)
2. [Infrastructure Requirements](#infrastructure-requirements)
3. [Security Configuration](#security-configuration)
4. [Storage Backend Setup](#storage-backend-setup)
5. [Application Deployment](#application-deployment)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Maintenance and Operations](#maintenance-and-operations)
8. [Troubleshooting](#troubleshooting)

## Pre-Deployment Planning

### Deployment Architecture Decision Matrix

| Factor | File Store | MongoDB + Azure | PostgreSQL + S3 |
|--------|------------|-----------------|-----------------|
| **Scale** | Small (<1000 runs) | Large (>10K runs) | Enterprise (>100K runs) |
| **Users** | 1-5 users | 10-100 users | 100+ users |
| **Availability** | Single point | High availability | High availability |
| **Security** | Basic | Enterprise | Enterprise |
| **Cost** | Low | Medium | High |
| **Complexity** | Low | Medium | High |

### Capacity Planning

#### Storage Requirements

```bash
# Estimate storage needs
# Metadata: ~1KB per run + parameters/metrics
# Artifacts: Highly variable (models, logs, datasets)

# Example calculation for 10,000 runs
RUNS=10000
METADATA_SIZE=$((RUNS * 1))  # 10MB metadata
ARTIFACT_SIZE=$((RUNS * 100))  # 1GB artifacts (100MB avg per run)
TOTAL_SIZE=$((METADATA_SIZE + ARTIFACT_SIZE))  # ~1GB total

echo "Estimated storage needed: ${TOTAL_SIZE}MB"
```

#### Compute Requirements

| Component | CPU | Memory | Storage | Notes |
|-----------|-----|--------|---------|-------|
| **Genesis-Flow Server** | 2-4 cores | 4-8GB | 10GB | Scales with concurrent users |
| **MongoDB** | 4-8 cores | 8-16GB | 100GB+ | Storage grows with metadata |
| **Load Balancer** | 1-2 cores | 2-4GB | 10GB | For high availability |

## Infrastructure Requirements

### Minimum System Requirements

```yaml
# Production minimum requirements
Server Specifications:
  CPU: 4 cores (2.4GHz+)
  Memory: 8GB RAM
  Storage: 100GB SSD
  Network: 1Gbps
  OS: Ubuntu 20.04+ / RHEL 8+ / Windows Server 2019+

Database (MongoDB):
  CPU: 4 cores
  Memory: 16GB RAM
  Storage: 500GB SSD (with growth capacity)
  IOPS: 3000+ provisioned

Artifact Storage (Azure Blob):
  Capacity: 1TB+ (scalable)
  Tier: Hot/Cool based on access patterns
  Replication: LRS/GRS based on requirements
```

### Network Requirements

```bash
# Required network ports
Genesis-Flow Server: 5000 (HTTP/HTTPS)
MongoDB: 27017 (MongoDB Protocol)
Azure Blob: 443 (HTTPS)

# Firewall rules
iptables -A INPUT -p tcp --dport 5000 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 27017 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 443 -j ACCEPT
```

## Security Configuration

### SSL/TLS Configuration

#### MongoDB SSL Setup

```bash
# Generate MongoDB SSL certificates
mkdir -p /etc/mongodb/ssl

# Create CA certificate
openssl genrsa -out /etc/mongodb/ssl/ca-key.pem 2048
openssl req -new -x509 -days 3650 -key /etc/mongodb/ssl/ca-key.pem \
    -out /etc/mongodb/ssl/ca-cert.pem

# Create server certificate
openssl genrsa -out /etc/mongodb/ssl/server-key.pem 2048
openssl req -new -key /etc/mongodb/ssl/server-key.pem \
    -out /etc/mongodb/ssl/server.csr

openssl x509 -req -in /etc/mongodb/ssl/server.csr \
    -CA /etc/mongodb/ssl/ca-cert.pem \
    -CAkey /etc/mongodb/ssl/ca-key.pem \
    -CAcreateserial -out /etc/mongodb/ssl/server-cert.pem -days 365

# Combine server certificate and key
cat /etc/mongodb/ssl/server-cert.pem /etc/mongodb/ssl/server-key.pem > \
    /etc/mongodb/ssl/server.pem
```

#### MongoDB Configuration with SSL

```yaml
# /etc/mongod.conf
net:
  port: 27017
  ssl:
    mode: requireSSL
    PEMKeyFile: /etc/mongodb/ssl/server.pem
    CAFile: /etc/mongodb/ssl/ca-cert.pem

security:
  authorization: enabled

storage:
  dbPath: /var/lib/mongodb
  journal:
    enabled: true

systemLog:
  destination: file
  logAppend: true
  path: /var/log/mongodb/mongod.log
```

### Authentication Setup

#### MongoDB User Creation

```javascript
// Connect to MongoDB as admin
use admin

// Create admin user
db.createUser({
  user: "admin",
  pwd: "secure_admin_password",
  roles: [
    { role: "userAdminAnyDatabase", db: "admin" },
    { role: "readWriteAnyDatabase", db: "admin" },
    { role: "dbAdminAnyDatabase", db: "admin" }
  ]
})

// Create Genesis-Flow application user
use mlflow_db
db.createUser({
  user: "mlflow_user",
  pwd: "secure_mlflow_password",
  roles: [
    { role: "readWrite", db: "mlflow_db" },
    { role: "dbAdmin", db: "mlflow_db" }
  ]
})
```

#### Azure Storage Authentication

```bash
# Using Azure CLI
az login
az account set --subscription "your-subscription-id"

# Create storage account
az storage account create \
    --name genesisflowstorage \
    --resource-group genesis-flow-rg \
    --location eastus \
    --sku Standard_LRS \
    --encryption-services blob

# Create container
az storage container create \
    --name artifacts \
    --account-name genesisflowstorage \
    --public-access off
```

## Storage Backend Setup

### MongoDB Deployment

#### Production MongoDB Configuration

```yaml
# docker-compose.yml for MongoDB
version: '3.8'
services:
  mongodb:
    image: mongo:6.0
    container_name: genesis-flow-mongodb
    restart: always
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: secure_password
      MONGO_INITDB_DATABASE: mlflow_db
    volumes:
      - mongodb_data:/data/db
      - ./mongodb.conf:/etc/mongod.conf:ro
      - ./ssl:/etc/mongodb/ssl:ro
    command: ["mongod", "--config", "/etc/mongod.conf"]
    networks:
      - genesis-flow-network

  mongodb-backup:
    image: mongo:6.0
    container_name: mongodb-backup
    restart: "no"
    environment:
      MONGO_URI: "mongodb://admin:secure_password@mongodb:27017"
    volumes:
      - ./backups:/backups
    command: >
      bash -c "
        mongodump --uri=$$MONGO_URI --out=/backups/$(date +%Y%m%d_%H%M%S)
      "
    depends_on:
      - mongodb
    networks:
      - genesis-flow-network

volumes:
  mongodb_data:

networks:
  genesis-flow-network:
    driver: bridge
```

#### MongoDB Indexing for Performance

```javascript
// Connect to your Genesis-Flow database
use mlflow_db

// Create indexes for better performance
db.experiments.createIndex({ "name": 1 }, { unique: true })
db.experiments.createIndex({ "lifecycle_stage": 1 })
db.experiments.createIndex({ "creation_time": 1 })

db.runs.createIndex({ "experiment_id": 1 })
db.runs.createIndex({ "run_id": 1 }, { unique: true })
db.runs.createIndex({ "status": 1 })
db.runs.createIndex({ "start_time": 1 })
db.runs.createIndex({ "end_time": 1 })
db.runs.createIndex({ "user_id": 1 })

// Compound indexes for common queries
db.runs.createIndex({ "experiment_id": 1, "status": 1 })
db.runs.createIndex({ "experiment_id": 1, "start_time": -1 })

db.metrics.createIndex({ "run_id": 1, "key": 1 })
db.metrics.createIndex({ "run_id": 1, "key": 1, "step": 1 })

db.params.createIndex({ "run_id": 1, "key": 1 })
db.tags.createIndex({ "run_id": 1, "key": 1 })
```

### Azure Blob Storage Configuration

```bash
# Set up Azure Blob Storage connection
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=genesisflowstorage;AccountKey=your_key;EndpointSuffix=core.windows.net"

# Alternative: Use SAS token
export AZURE_STORAGE_SAS_TOKEN="?sv=2021-06-08&ss=b&srt=sco&sp=rwdlacupx&se=2024-12-31T23:59:59Z&st=2023-01-01T00:00:00Z&spr=https&sig=signature"

# Test connection
az storage container list --connection-string "$AZURE_STORAGE_CONNECTION_STRING"
```

## Application Deployment

### Docker Deployment

#### Production Dockerfile

```dockerfile
# Multi-stage build for optimized image
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml poetry.lock ./

# Install Poetry and dependencies
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only=main --no-dev

# Production stage
FROM python:3.11-slim as production

# Create non-root user
RUN groupadd -r mlflow && useradd -r -g mlflow mlflow

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .

# Set ownership and permissions
RUN chown -R mlflow:mlflow /app
USER mlflow

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

EXPOSE 5000

# Use environment variables for configuration
ENV MLFLOW_TRACKING_URI=""
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=""
ENV MLFLOW_SERVER_HOST="0.0.0.0"
ENV MLFLOW_SERVER_PORT="5000"

CMD ["mlflow", "server", \
     "--backend-store-uri", "${MLFLOW_TRACKING_URI}", \
     "--default-artifact-root", "${MLFLOW_DEFAULT_ARTIFACT_ROOT}", \
     "--host", "${MLFLOW_SERVER_HOST}", \
     "--port", "${MLFLOW_SERVER_PORT}", \
     "--workers", "4"]
```

#### Docker Compose for Production

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  genesis-flow:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    container_name: genesis-flow-server
    restart: always
    ports:
      - "5000:5000"
    environment:
      MLFLOW_TRACKING_URI: "mongodb://mlflow_user:secure_password@mongodb:27017/mlflow_db"
      MLFLOW_DEFAULT_ARTIFACT_ROOT: "azure://artifacts/"
      AZURE_STORAGE_CONNECTION_STRING: "${AZURE_STORAGE_CONNECTION_STRING}"
      MLFLOW_ENABLE_SECURE_MODEL_LOADING: "true"
      MLFLOW_STRICT_INPUT_VALIDATION: "true"
    volumes:
      - ./logs:/app/logs
      - ./config:/app/config:ro
    depends_on:
      - mongodb
    networks:
      - genesis-flow-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  mongodb:
    image: mongo:6.0
    container_name: genesis-flow-mongodb
    restart: always
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: "${MONGO_ROOT_PASSWORD}"
      MONGO_INITDB_DATABASE: mlflow_db
    volumes:
      - mongodb_data:/data/db
      - ./mongodb/mongod.conf:/etc/mongod.conf:ro
      - ./mongodb/ssl:/etc/mongodb/ssl:ro
    command: ["mongod", "--config", "/etc/mongod.conf"]
    networks:
      - genesis-flow-network
    healthcheck:
      test: ["CMD", "mongo", "--eval", "db.adminCommand('ping')"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    container_name: genesis-flow-nginx
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - genesis-flow
    networks:
      - genesis-flow-network

volumes:
  mongodb_data:
    driver: local

networks:
  genesis-flow-network:
    driver: bridge
```

### Kubernetes Deployment

#### Kubernetes Manifests

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: genesis-flow
---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: genesis-flow-config
  namespace: genesis-flow
data:
  MLFLOW_SERVER_HOST: "0.0.0.0"
  MLFLOW_SERVER_PORT: "5000"
  MLFLOW_ENABLE_SECURE_MODEL_LOADING: "true"
  MLFLOW_STRICT_INPUT_VALIDATION: "true"
---
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: genesis-flow-secrets
  namespace: genesis-flow
type: Opaque
data:
  # Base64 encoded values
  mongodb-uri: bW9uZ29kYjovL21sZmxvd191c2VyOnNlY3VyZV9wYXNzd29yZEBtb25nb2RiOjI3MDE3L21sZmxvd19kYg==
  azure-connection-string: RGVmYXVsdEVuZHBvaW50c1Byb3RvY29sPWh0dHBzO0FjY291bnROYW1lPWdlbmVzaXNmbG93c3RvcmFnZTtBY2NvdW50S2V5PXlvdXJfa2V5
---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genesis-flow
  namespace: genesis-flow
  labels:
    app: genesis-flow
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 1
  selector:
    matchLabels:
      app: genesis-flow
  template:
    metadata:
      labels:
        app: genesis-flow
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 2000
      containers:
      - name: genesis-flow
        image: genesis-flow:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 5000
          name: http
        env:
        - name: MLFLOW_TRACKING_URI
          valueFrom:
            secretKeyRef:
              name: genesis-flow-secrets
              key: mongodb-uri
        - name: MLFLOW_DEFAULT_ARTIFACT_ROOT
          value: "azure://artifacts/"
        - name: AZURE_STORAGE_CONNECTION_STRING
          valueFrom:
            secretKeyRef:
              name: genesis-flow-secrets
              key: azure-connection-string
        envFrom:
        - configMapRef:
            name: genesis-flow-config
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
          readOnly: true
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: config-volume
        configMap:
          name: genesis-flow-config
      - name: logs-volume
        emptyDir: {}
---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: genesis-flow-service
  namespace: genesis-flow
  labels:
    app: genesis-flow
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 5000
    protocol: TCP
    name: http
  selector:
    app: genesis-flow
---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: genesis-flow-ingress
  namespace: genesis-flow
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
spec:
  tls:
  - hosts:
    - genesis-flow.yourdomain.com
    secretName: genesis-flow-tls
  rules:
  - host: genesis-flow.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: genesis-flow-service
            port:
              number: 80
```

#### MongoDB StatefulSet

```yaml
# mongodb-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mongodb
  namespace: genesis-flow
spec:
  serviceName: mongodb-service
  replicas: 3
  selector:
    matchLabels:
      app: mongodb
  template:
    metadata:
      labels:
        app: mongodb
    spec:
      containers:
      - name: mongodb
        image: mongo:6.0
        ports:
        - containerPort: 27017
        env:
        - name: MONGO_INITDB_ROOT_USERNAME
          value: admin
        - name: MONGO_INITDB_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mongodb-secret
              key: root-password
        volumeMounts:
        - name: mongodb-data
          mountPath: /data/db
        - name: mongodb-config
          mountPath: /etc/mongod.conf
          subPath: mongod.conf
        resources:
          requests:
            cpu: 1000m
            memory: 2Gi
          limits:
            cpu: 2000m
            memory: 4Gi
      volumes:
      - name: mongodb-config
        configMap:
          name: mongodb-config
  volumeClaimTemplates:
  - metadata:
      name: mongodb-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
```

### Load Balancer Configuration

#### NGINX Configuration

```nginx
# nginx.conf
upstream genesis_flow_backend {
    least_conn;
    server genesis-flow-1:5000 max_fails=3 fail_timeout=30s;
    server genesis-flow-2:5000 max_fails=3 fail_timeout=30s;
    server genesis-flow-3:5000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name genesis-flow.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name genesis-flow.yourdomain.com;

    # SSL configuration
    ssl_certificate /etc/nginx/ssl/genesis-flow.crt;
    ssl_certificate_key /etc/nginx/ssl/genesis-flow.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";

    # Increase upload limits for artifacts
    client_max_body_size 100M;
    client_body_timeout 60s;
    client_header_timeout 60s;

    location / {
        proxy_pass http://genesis_flow_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Health check
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503;
    }

    location /health {
        access_log off;
        proxy_pass http://genesis_flow_backend;
        proxy_set_header Host $host;
    }
}
```

## Monitoring and Observability

### Application Monitoring

#### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "genesis_flow_rules.yml"

scrape_configs:
  - job_name: 'genesis-flow'
    static_configs:
      - targets: ['genesis-flow:5000']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'mongodb'
    static_configs:
      - targets: ['mongodb:9216']
    scrape_interval: 30s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

#### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Genesis-Flow Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Active Experiments",
        "type": "stat",
        "targets": [
          {
            "expr": "genesis_flow_experiments_active",
            "legendFormat": "Active Experiments"
          }
        ]
      }
    ]
  }
}
```

### Log Management

#### Structured Logging Configuration

```python
# logging_config.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'experiment_id'):
            log_entry['experiment_id'] = record.experiment_id
        if hasattr(record, 'run_id'):
            log_entry['run_id'] = record.run_id
            
        return json.dumps(log_entry)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/logs/genesis-flow.log')
    ]
)

logger = logging.getLogger(__name__)
for handler in logger.handlers:
    handler.setFormatter(JSONFormatter())
```

## Maintenance and Operations

### Backup Strategy

#### MongoDB Backup Script

```bash
#!/bin/bash
# backup_mongodb.sh

set -e

# Configuration
MONGO_URI="mongodb://admin:password@mongodb:27017"
BACKUP_DIR="/backups/mongodb"
RETENTION_DAYS=30
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "${BACKUP_DIR}"

# Perform backup
echo "Starting MongoDB backup at $(date)"
mongodump --uri="${MONGO_URI}" --out="${BACKUP_DIR}/${DATE}"

# Compress backup
tar -czf "${BACKUP_DIR}/${DATE}.tar.gz" -C "${BACKUP_DIR}" "${DATE}"
rm -rf "${BACKUP_DIR}/${DATE}"

# Upload to cloud storage
az storage blob upload \
    --file "${BACKUP_DIR}/${DATE}.tar.gz" \
    --container-name backups \
    --name "mongodb/${DATE}.tar.gz" \
    --connection-string "$AZURE_STORAGE_CONNECTION_STRING"

# Clean up old backups
find "${BACKUP_DIR}" -name "*.tar.gz" -mtime +${RETENTION_DAYS} -delete

echo "MongoDB backup completed at $(date)"
```

#### Automated Backup with Cron

```bash
# Add to crontab
0 2 * * * /opt/scripts/backup_mongodb.sh >> /var/log/backup.log 2>&1
0 3 * * 0 /opt/scripts/backup_azure_artifacts.sh >> /var/log/backup.log 2>&1
```

### Update and Rollback Procedures

#### Rolling Update Script

```bash
#!/bin/bash
# rolling_update.sh

set -e

NEW_VERSION=$1
if [ -z "$NEW_VERSION" ]; then
    echo "Usage: $0 <new_version>"
    exit 1
fi

echo "Starting rolling update to version $NEW_VERSION"

# Update image tag
kubectl set image deployment/genesis-flow \
    genesis-flow=genesis-flow:$NEW_VERSION \
    -n genesis-flow

# Wait for rollout
kubectl rollout status deployment/genesis-flow -n genesis-flow

# Verify deployment
kubectl get pods -n genesis-flow -l app=genesis-flow

echo "Rolling update completed successfully"
```

#### Rollback Script

```bash
#!/bin/bash
# rollback.sh

echo "Starting rollback..."

# Rollback to previous version
kubectl rollout undo deployment/genesis-flow -n genesis-flow

# Wait for rollback
kubectl rollout status deployment/genesis-flow -n genesis-flow

# Verify rollback
kubectl get pods -n genesis-flow -l app=genesis-flow

echo "Rollback completed successfully"
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: MongoDB Connection Timeout

```bash
# Symptoms
# - Genesis-Flow fails to start
# - Connection timeout errors in logs

# Diagnosis
# Check MongoDB connectivity
mongosh --host mongodb:27017 --username admin --password

# Check network connectivity
telnet mongodb 27017

# Solutions
# 1. Verify MongoDB is running
kubectl get pods -n genesis-flow -l app=mongodb

# 2. Check MongoDB logs
kubectl logs -n genesis-flow mongodb-0

# 3. Verify network policies
kubectl get networkpolicies -n genesis-flow
```

#### Issue: High Memory Usage

```bash
# Symptoms
# - Genesis-Flow pods getting OOMKilled
# - Slow response times

# Diagnosis
# Check resource usage
kubectl top pods -n genesis-flow

# Check memory metrics
curl -s http://genesis-flow:5000/metrics | grep memory

# Solutions
# 1. Increase memory limits
kubectl patch deployment genesis-flow -n genesis-flow -p \
'{"spec":{"template":{"spec":{"containers":[{"name":"genesis-flow","resources":{"limits":{"memory":"8Gi"}}}]}}}}'

# 2. Enable memory profiling
export MLFLOW_ENABLE_MEMORY_PROFILING=true
```

#### Issue: Artifact Upload Failures

```bash
# Symptoms
# - Artifact logging fails
# - Azure Blob storage errors

# Diagnosis
# Test Azure connectivity
az storage container list --connection-string "$AZURE_STORAGE_CONNECTION_STRING"

# Check container permissions
az storage container show-permission \
    --name artifacts \
    --connection-string "$AZURE_STORAGE_CONNECTION_STRING"

# Solutions
# 1. Verify connection string
echo $AZURE_STORAGE_CONNECTION_STRING

# 2. Check container exists
az storage container create \
    --name artifacts \
    --connection-string "$AZURE_STORAGE_CONNECTION_STRING"

# 3. Test upload manually
echo "test" | az storage blob upload \
    --container-name artifacts \
    --name test.txt \
    --connection-string "$AZURE_STORAGE_CONNECTION_STRING"
```

### Performance Tuning

#### MongoDB Performance Optimization

```javascript
// Connect to MongoDB
use mlflow_db

// Check slow queries
db.runCommand({profile: 2, slowms: 100})

// View profiler data
db.system.profile.find().limit(5).sort({ts: -1}).pretty()

// Optimize indexes based on slow queries
db.runs.createIndex({"experiment_id": 1, "start_time": -1})
db.metrics.createIndex({"run_id": 1, "key": 1, "timestamp": -1})

// Check index usage
db.runs.find({"experiment_id": "123"}).explain("executionStats")
```

#### Application Performance Tuning

```python
# Genesis-Flow configuration optimizations
import os

# Database connection pool settings
os.environ['MLFLOW_MONGODB_MAX_POOL_SIZE'] = '100'
os.environ['MLFLOW_MONGODB_MIN_POOL_SIZE'] = '10'
os.environ['MLFLOW_MONGODB_MAX_IDLE_TIME'] = '30000'

# Cache settings
os.environ['MLFLOW_ENABLE_CACHING'] = 'true'
os.environ['MLFLOW_CACHE_TTL'] = '300'

# Worker settings
os.environ['MLFLOW_SERVER_WORKERS'] = '4'
os.environ['MLFLOW_WORKER_TIMEOUT'] = '300'
```

### Disaster Recovery

#### Backup Verification

```bash
#!/bin/bash
# verify_backup.sh

BACKUP_FILE=$1
TEMP_DIR="/tmp/backup_verify"

# Extract backup
mkdir -p $TEMP_DIR
tar -xzf $BACKUP_FILE -C $TEMP_DIR

# Start temporary MongoDB
docker run -d --name temp-mongo \
    -v $TEMP_DIR:/data/db \
    mongo:6.0

# Wait for MongoDB to start
sleep 10

# Verify data integrity
mongosh --host localhost:27017 --eval "
    use mlflow_db;
    print('Experiments:', db.experiments.countDocuments());
    print('Runs:', db.runs.countDocuments());
    print('Sample experiment:', JSON.stringify(db.experiments.findOne()));
"

# Cleanup
docker stop temp-mongo
docker rm temp-mongo
rm -rf $TEMP_DIR
```

#### Recovery Procedure

```bash
#!/bin/bash
# disaster_recovery.sh

BACKUP_DATE=$1
if [ -z "$BACKUP_DATE" ]; then
    echo "Usage: $0 <backup_date>"
    exit 1
fi

echo "Starting disaster recovery for backup date: $BACKUP_DATE"

# Download backup from cloud storage
az storage blob download \
    --container-name backups \
    --name "mongodb/${BACKUP_DATE}.tar.gz" \
    --file "/tmp/${BACKUP_DATE}.tar.gz" \
    --connection-string "$AZURE_STORAGE_CONNECTION_STRING"

# Stop application
kubectl scale deployment genesis-flow --replicas=0 -n genesis-flow

# Restore MongoDB
kubectl exec -it mongodb-0 -n genesis-flow -- \
    mongorestore --drop --uri="mongodb://admin:password@localhost:27017" \
    /tmp/restore

# Start application
kubectl scale deployment genesis-flow --replicas=3 -n genesis-flow

# Verify recovery
curl -f http://genesis-flow.yourdomain.com/health

echo "Disaster recovery completed successfully"
```

This comprehensive deployment guide provides enterprise-grade deployment patterns for Genesis-Flow with security, scalability, and operational best practices.