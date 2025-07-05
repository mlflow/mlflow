# Genesis Platform: Microservices Architecture Transformation

## Executive Summary

This document presents a comprehensive microservices architecture for the Genesis ML Platform, transforming the current monolithic `genesis-service-modelhub` into a scalable, maintainable ecosystem of specialized services. The proposed architecture emphasizes separation of concerns, independent scalability, and optional high-performance components using Go.

## Current State Analysis

### Monolithic Service Overview

The current `genesis-service-modelhub` is a large FastAPI monolith handling multiple domains:

```mermaid
graph TD
    subgraph "Current Monolithic Architecture"
        MS[genesis-service-modelhub<br/>FastAPI Monolith]
        
        subgraph "Domains in Single Service"
            EXP[Experiments & Runs]
            DS[Datasets]
            PR[Prompts]
            OBS[Observability]
            PL[Pipelines]
            INF[Inference/KServe]
            MOD[Models]
            SEC[Secrets/Variables]
            REG[Container Registry]
        end
        
        subgraph "Workers"
            OW[Observer Worker]
            EW[Evaluator Worker]
        end
    end
    
    MS --> MongoDB[(MongoDB)]
    MS --> MLF[MLflow Server]
    MS --> K8S[Kubernetes API]
    MS --> ARGO[Argo Workflows]
    MS --> KAFKA[Kafka]
    
    OW --> KAFKA
    EW --> KAFKA
```

### Current Pain Points

1. **Tight Coupling**: All domains share the same codebase and deployment
2. **Scaling Challenges**: Cannot scale individual components independently
3. **Deployment Risk**: Changes to one domain affect all others
4. **Resource Inefficiency**: Memory/CPU allocated for all features even if unused
5. **Development Bottlenecks**: Teams step on each other's toes
6. **Testing Complexity**: Must test entire monolith for small changes

## Proposed Microservices Architecture

### High-Level Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        UI[genesis-frontend]
        CLI[CLI Tools]
        SDK[Python/JS SDKs]
    end
    
    subgraph "API Gateway Layer"
        GW[genesis-gateway<br/>API Gateway<br/>Node.js/NestJS]
    end
    
    subgraph "Core Services Layer"
        ES[Experiment Service<br/>Python/FastAPI]
        MS[Model Registry Service<br/>Python/FastAPI]
        DS[Dataset Service<br/>Python/FastAPI]
        PS[Prompt Service<br/>Python/FastAPI]
        OBS[Observability Service<br/>Python/FastAPI]
        PLS[Pipeline Service<br/>Python/FastAPI]
        IS[Inference Service<br/>Python/FastAPI]
        SS[Secrets Service<br/>Go/Gin]
    end
    
    subgraph "High-Performance Services"
        MAS[Metrics Aggregator<br/>Go]
        EPS[Event Processor<br/>Go]
        QS[Query Service<br/>Go]
    end
    
    subgraph "Worker Layer"
        OW[Observer Workers<br/>Python]
        EW[Evaluator Workers<br/>Python]
        MW[Monitoring Workers<br/>Go]
    end
    
    subgraph "Data Layer"
        MDB[(MongoDB/<br/>Cosmos DB)]
        REDIS[(Redis)]
        TS[(TimescaleDB)]
        BLOB[Azure Blob]
    end
    
    subgraph "Infrastructure"
        KAFKA[Kafka]
        K8S[Kubernetes]
        ARGO[Argo]
    end
    
    UI --> GW
    CLI --> GW
    SDK --> GW
    
    GW --> ES
    GW --> MS
    GW --> DS
    GW --> PS
    GW --> OBS
    GW --> PLS
    GW --> IS
    GW --> SS
    
    ES --> MDB
    MS --> MDB
    DS --> MDB
    DS --> BLOB
    PS --> MDB
    OBS --> MDB
    OBS --> TS
    PLS --> ARGO
    IS --> K8S
    SS --> REDIS
    
    MAS --> TS
    EPS --> KAFKA
    QS --> MDB
    QS --> REDIS
    
    OW --> KAFKA
    EW --> KAFKA
    MW --> KAFKA
    
    style GW fill:#ff9999
    style SS fill:#99ff99
    style MAS fill:#99ff99
    style EPS fill:#99ff99
    style QS fill:#99ff99
    style MW fill:#99ff99
```

### Service Breakdown

#### 1. API Gateway (genesis-gateway)
**Current**: `genesis-bff-modelhub`  
**Technology**: Node.js with NestJS (existing)  
**Responsibilities**:
- Request routing and load balancing
- Authentication and authorization
- Rate limiting and throttling
- Request/response transformation
- API versioning
- Circuit breaking

```mermaid
graph LR
    subgraph "API Gateway Features"
        AUTH[Authentication]
        ROUTE[Routing]
        LIMIT[Rate Limiting]
        CACHE[Response Cache]
        TRANSFORM[Data Transform]
        LOG[Request Logging]
    end
```

#### 2. Experiment Service
**Extracted From**: Experiments and runs endpoints  
**Technology**: Python/FastAPI + Genesis-Flow  
**Responsibilities**:
- MLflow experiment management
- Run tracking and metadata
- Metrics and parameters storage
- Direct MongoDB integration via Genesis-Flow

```mermaid
graph TD
    subgraph "Experiment Service"
        API[FastAPI]
        GF[Genesis-Flow]
        
        API --> GF
        GF --> MDB[(MongoDB)]
        GF --> BLOB[Blob Storage]
    end
```

#### 3. Model Registry Service
**Extracted From**: Model management endpoints  
**Technology**: Python/FastAPI + Genesis-Flow  
**Responsibilities**:
- Model versioning and lifecycle
- Model metadata and tags
- Model serving preparation
- Integration with container registry

#### 4. Dataset Service
**Extracted From**: Dataset endpoints  
**Technology**: Python/FastAPI  
**Responsibilities**:
- Dataset versioning and storage
- Data quality monitoring
- Integration with Evidently
- Large file handling with streaming

#### 5. Prompt Engineering Service
**Extracted From**: Prompt endpoints  
**Technology**: Python/FastAPI  
**Responsibilities**:
- Prompt template management
- Version control for prompts
- A/B testing support
- Prompt evaluation metrics

#### 6. Observability Service
**Extracted From**: Observability endpoints  
**Technology**: Python/FastAPI  
**Responsibilities**:
- Trace collection and storage
- Metrics aggregation
- Dashboard APIs
- Cost tracking integration

```mermaid
graph TD
    subgraph "Observability Service Architecture"
        API[REST API]
        TS[Trace Service]
        MS[Metrics Service]
        DS[Dashboard Service]
        
        API --> TS
        API --> MS
        API --> DS
        
        TS --> MDB[(MongoDB)]
        MS --> TDB[(TimescaleDB)]
        DS --> CACHE[(Redis)]
    end
```

#### 7. Pipeline Service
**Extracted From**: Pipeline endpoints  
**Technology**: Python/FastAPI  
**Responsibilities**:
- Argo workflow management
- Pipeline templates
- Execution monitoring
- Pipeline versioning

#### 8. Inference Service
**Extracted From**: KServe endpoints  
**Technology**: Python/FastAPI  
**Responsibilities**:
- Model deployment to KServe
- Endpoint management
- Autoscaling configuration
- Traffic routing

#### 9. Secrets Service (Go)
**Extracted From**: Variables endpoints  
**Technology**: Go with Gin framework  
**Why Go**: High-performance, secure handling of sensitive data  
**Responsibilities**:
- Kubernetes secrets management
- Encryption/decryption
- Access control
- Audit logging

```go
// Example Go service structure
type SecretsService struct {
    k8sClient kubernetes.Interface
    cache     *redis.Client
    encryptor crypto.Encryptor
}

func (s *SecretsService) GetSecret(ctx context.Context, name string) (*Secret, error) {
    // Fast path: check cache
    if cached, err := s.cache.Get(ctx, name); err == nil {
        return s.decrypt(cached)
    }
    
    // Slow path: fetch from K8s
    secret, err := s.k8sClient.CoreV1().
        Secrets(namespace).
        Get(ctx, name, metav1.GetOptions{})
    
    // Cache for next time
    s.cache.Set(ctx, name, encrypted, ttl)
    
    return s.decrypt(secret)
}
```

### High-Performance Go Services

#### 1. Metrics Aggregator Service
**Purpose**: Real-time metrics aggregation and processing  
**Technology**: Go  
**Why Go**: High throughput, low latency, efficient memory usage

```mermaid
graph LR
    subgraph "Metrics Aggregator"
        IN[Kafka Consumer]
        AGG[Aggregation Engine]
        OUT[TimescaleDB Writer]
        
        IN -->|High Volume| AGG
        AGG -->|Batched| OUT
    end
```

**Key Features**:
- Process millions of metrics per second
- Real-time aggregation (sum, avg, percentiles)
- Efficient memory usage with streaming algorithms
- Batched writes to TimescaleDB

#### 2. Event Processor Service
**Purpose**: High-throughput event processing  
**Technology**: Go  
**Why Go**: Excellent concurrency primitives, low GC overhead

```go
// High-performance event processor
type EventProcessor struct {
    workers   int
    inputCh   chan Event
    outputCh  chan ProcessedEvent
    processor EventHandler
}

func (ep *EventProcessor) Start(ctx context.Context) {
    // Fan-out pattern for parallel processing
    for i := 0; i < ep.workers; i++ {
        go ep.worker(ctx)
    }
}

func (ep *EventProcessor) worker(ctx context.Context) {
    for {
        select {
        case event := <-ep.inputCh:
            processed := ep.processor.Process(event)
            ep.outputCh <- processed
        case <-ctx.Done():
            return
        }
    }
}
```

#### 3. Query Service
**Purpose**: Fast read queries with caching  
**Technology**: Go  
**Why Go**: Fast JSON serialization, efficient caching

**Features**:
- Multi-level caching (Redis + in-memory)
- Query optimization
- Connection pooling
- Response streaming for large datasets

### Worker Architecture

```mermaid
graph TD
    subgraph "Worker Pool Architecture"
        K[Kafka]
        
        subgraph "Python Workers"
            OW1[Observer Worker 1]
            OW2[Observer Worker 2]
            EW1[Evaluator Worker 1]
            EW2[Evaluator Worker 2]
        end
        
        subgraph "Go Workers"
            MW1[Monitoring Worker 1]
            MW2[Monitoring Worker 2]
            MW3[Monitoring Worker 3]
        end
        
        K --> OW1
        K --> OW2
        K --> EW1
        K --> EW2
        K --> MW1
        K --> MW2
        K --> MW3
        
        OW1 --> MDB[(MongoDB)]
        OW2 --> MDB
        EW1 --> MDB
        EW2 --> MDB
        MW1 --> TS[(TimescaleDB)]
        MW2 --> TS
        MW3 --> TS
    end
```

#### Worker Types

1. **Observer Workers** (Python)
   - Process observability traces
   - Use autonomize-observer SDK
   - Write to MongoDB

2. **Evaluator Workers** (Python)
   - Evaluate prompts and models
   - Complex ML computations
   - Integration with ML libraries

3. **Monitoring Workers** (Go)
   - High-frequency metrics collection
   - System resource monitoring
   - Real-time alerting

### Event-Driven Architecture

```mermaid
graph LR
    subgraph "Event Flow"
        P1[Experiment Service]
        P2[Model Service]
        P3[Dataset Service]
        
        K[Kafka]
        
        C1[Observer Worker]
        C2[Evaluator Worker]
        C3[Notification Service]
        
        P1 -->|experiment.created| K
        P2 -->|model.deployed| K
        P3 -->|dataset.updated| K
        
        K --> C1
        K --> C2
        K --> C3
    end
```

### Data Architecture

```mermaid
graph TD
    subgraph "Polyglot Persistence"
        subgraph "Document Store"
            MDB[(MongoDB/Cosmos DB)]
            DESC1[Experiments, Runs,<br/>Models, Prompts]
        end
        
        subgraph "Time Series"
            TS[(TimescaleDB)]
            DESC2[Metrics, Monitoring,<br/>Performance Data]
        end
        
        subgraph "Cache Layer"
            REDIS[(Redis)]
            DESC3[Sessions, Hot Data,<br/>Query Cache]
        end
        
        subgraph "Object Store"
            BLOB[Azure Blob]
            DESC4[Models, Datasets,<br/>Artifacts, Logs]
        end
    end
```

## Migration Strategy

### Phase 1: Strangler Fig Pattern
```mermaid
graph LR
    subgraph "Phase 1"
        MON[Monolith]
        GW[API Gateway]
        NS1[New Service 1]
        
        GW --> MON
        GW -.->|Selected Routes| NS1
    end
```

### Phase 2: Progressive Decomposition
```mermaid
graph LR
    subgraph "Phase 2"
        MON[Reduced Monolith]
        GW[API Gateway]
        NS1[Service 1]
        NS2[Service 2]
        NS3[Service 3]
        
        GW --> MON
        GW --> NS1
        GW --> NS2
        GW --> NS3
    end
```

### Phase 3: Complete Migration
```mermaid
graph LR
    subgraph "Phase 3"
        GW[API Gateway]
        NS1[Service 1]
        NS2[Service 2]
        NS3[Service 3]
        NS4[Service 4]
        NS5[Service 5]
        
        GW --> NS1
        GW --> NS2
        GW --> NS3
        GW --> NS4
        GW --> NS5
    end
```

### Migration Priority

1. **High Priority** (Phase 1)
   - Secrets Service (security critical)
   - Observability Service (already semi-independent)
   - Dataset Service (clear boundaries)

2. **Medium Priority** (Phase 2)
   - Model Registry Service
   - Prompt Engineering Service
   - Pipeline Service

3. **Lower Priority** (Phase 3)
   - Experiment Service
   - Inference Service

## Service Communication Patterns

### Synchronous Communication
```mermaid
sequenceDiagram
    participant Client
    participant Gateway
    participant Service
    participant Database
    
    Client->>Gateway: HTTP Request
    Gateway->>Service: Route Request
    Service->>Database: Query Data
    Database-->>Service: Return Data
    Service-->>Gateway: Response
    Gateway-->>Client: HTTP Response
```

### Asynchronous Communication
```mermaid
sequenceDiagram
    participant Service A
    participant Kafka
    participant Service B
    participant Service C
    
    Service A->>Kafka: Publish Event
    Kafka-->>Service B: Consume Event
    Kafka-->>Service C: Consume Event
    Note over Service B,Service C: Process in parallel
```

## Security Architecture

```mermaid
graph TD
    subgraph "Security Layers"
        subgraph "Edge Security"
            WAF[Web Application Firewall]
            DDoS[DDoS Protection]
        end
        
        subgraph "Gateway Security"
            AUTH[Authentication]
            AUTHZ[Authorization]
            LIMIT[Rate Limiting]
        end
        
        subgraph "Service Security"
            mTLS[Mutual TLS]
            RBAC[RBAC Policies]
            SECRETS[Secret Management]
        end
        
        subgraph "Data Security"
            ENC[Encryption at Rest]
            TLS[TLS in Transit]
            AUDIT[Audit Logging]
        end
    end
```

## Deployment Architecture

### Kubernetes Deployment
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: genesis-platform
---
# Example service deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: experiment-service
  namespace: genesis-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: experiment-service
  template:
    metadata:
      labels:
        app: experiment-service
    spec:
      containers:
      - name: experiment-service
        image: genesis/experiment-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: MONGODB_URI
          valueFrom:
            secretKeyRef:
              name: mongodb-secret
              key: uri
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: experiment-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: experiment-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Monitoring and Observability

```mermaid
graph TD
    subgraph "Observability Stack"
        subgraph "Metrics"
            PROM[Prometheus]
            GRAF[Grafana]
        end
        
        subgraph "Logging"
            FLUENT[Fluentd]
            ELASTIC[Elasticsearch]
            KIBANA[Kibana]
        end
        
        subgraph "Tracing"
            JAEGER[Jaeger]
            TEMPO[Tempo]
        end
        
        subgraph "Application Monitoring"
            APM[Application Insights]
            CUSTOM[Custom Dashboards]
        end
    end
    
    Services --> PROM
    Services --> FLUENT
    Services --> JAEGER
    Services --> APM
    
    PROM --> GRAF
    FLUENT --> ELASTIC
    ELASTIC --> KIBANA
    JAEGER --> TEMPO
```

## Performance Optimization Strategies

### 1. Go Services Benefits

**Why Go for specific services:**
- **Memory Efficiency**: 10x less memory than Python for high-throughput services
- **Concurrency**: Goroutines handle 100K+ concurrent connections
- **Low Latency**: Sub-millisecond response times for hot paths
- **CPU Efficiency**: Better utilization for compute-intensive tasks

**Benchmark Comparison:**
| Operation | Python Service | Go Service | Improvement |
|-----------|---------------|------------|-------------|
| Metrics Ingestion | 10K/sec | 100K/sec | 10x |
| P95 Latency | 50ms | 5ms | 10x |
| Memory Usage | 2GB | 200MB | 10x |
| Startup Time | 30s | 2s | 15x |

### 2. Caching Strategy

```mermaid
graph TD
    subgraph "Multi-Level Cache"
        L1[L1: In-Memory<br/>Application Cache]
        L2[L2: Redis<br/>Distributed Cache]
        L3[L3: CDN<br/>Edge Cache]
        DB[(Database)]
        
        Request --> L3
        L3 -->|Miss| L2
        L2 -->|Miss| L1
        L1 -->|Miss| DB
    end
```

## Cost Optimization

### Resource Allocation by Service

| Service | Language | Min Replicas | Max Replicas | CPU Request | Memory Request |
|---------|----------|--------------|--------------|-------------|----------------|
| Gateway | Node.js | 3 | 10 | 200m | 512Mi |
| Experiment | Python | 2 | 6 | 100m | 256Mi |
| Model Registry | Python | 2 | 6 | 100m | 256Mi |
| Dataset | Python | 2 | 8 | 200m | 512Mi |
| Observability | Python | 3 | 10 | 200m | 512Mi |
| Secrets | Go | 2 | 4 | 50m | 64Mi |
| Metrics Aggregator | Go | 3 | 8 | 100m | 128Mi |
| Event Processor | Go | 3 | 10 | 100m | 128Mi |

### Estimated Cost Savings

- **Infrastructure**: 30-40% reduction through efficient resource utilization
- **Development**: 50% faster feature deployment
- **Operations**: 60% reduction in incident resolution time

## Implementation Roadmap

### Quarter 1: Foundation
- [ ] Set up API Gateway with routing rules
- [ ] Extract Secrets Service (Go)
- [ ] Extract Observability Service
- [ ] Implement service discovery

### Quarter 2: Core Services
- [ ] Extract Dataset Service
- [ ] Extract Prompt Engineering Service
- [ ] Implement Metrics Aggregator (Go)
- [ ] Set up monitoring infrastructure

### Quarter 3: ML Services
- [ ] Extract Model Registry Service
- [ ] Extract Pipeline Service
- [ ] Implement Event Processor (Go)
- [ ] Migration of 50% traffic

### Quarter 4: Completion
- [ ] Extract Experiment Service
- [ ] Extract Inference Service
- [ ] Complete migration
- [ ] Decommission monolith

## Technology Stack Summary

### Primary Languages
- **Python**: ML services, complex business logic
- **Go**: High-performance services, workers
- **Node.js**: API Gateway (existing)

### Recommended Go Services
1. **Secrets Service**: Security and performance critical
2. **Metrics Aggregator**: High-throughput data processing
3. **Event Processor**: Real-time event handling
4. **Query Service**: Fast read operations
5. **Monitoring Workers**: System metrics collection

### Data Stores
- **MongoDB/Cosmos DB**: Primary document store
- **TimescaleDB**: Time-series metrics
- **Redis**: Caching and sessions
- **Azure Blob**: Object storage

### Infrastructure
- **Kubernetes**: Container orchestration
- **Kafka**: Event streaming
- **Istio**: Service mesh (optional)
- **ArgoCD**: GitOps deployment

## Conclusion

The proposed microservices architecture transforms the Genesis ML Platform into a modern, scalable system that:

1. **Enables Independent Scaling**: Each service scales based on its needs
2. **Improves Development Velocity**: Teams work independently
3. **Enhances Reliability**: Failure isolation and circuit breaking
4. **Optimizes Performance**: Go services for critical paths
5. **Reduces Costs**: Efficient resource utilization

The architecture maintains Python for ML-specific services while strategically introducing Go for performance-critical components, creating a best-of-both-worlds solution for the Genesis platform.

---

**Next Steps:**
1. Review and approve the architecture
2. Create detailed API specifications
3. Set up CI/CD pipelines
4. Begin Phase 1 implementation with Secrets Service