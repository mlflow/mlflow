# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Building and Testing

- `npm run build` - Compile TypeScript to JavaScript in dist/
- `npm run test` - Run Jest tests (65 tests covering all functionality)
- `npm run lint` - Run ESLint on TypeScript files
- `npm run lint:fix` - Run ESLint with auto-fix
- `npm run format` - Format code with Prettier
- `npm run format:check` - Check code formatting

### Single Test Execution

Use Jest's pattern matching: `npm test -- --testNamePattern="<test name>"` or `npm test <test-file-path>`

## Architecture Overview

This is a TypeScript SDK for MLflow Tracing that provides LLM observability. It's designed as a TypeScript port of the Python MLflow tracing implementation, maintaining API compatibility while leveraging OpenTelemetry for the underlying tracing infrastructure.

## Core Architecture

### Data Flow and Backend Communication

```
User Code → MLflow APIs → OpenTelemetry → Span Processor → Exporter → Backend
    ↓            ↓              ↓              ↓           ↓          ↓
withSpan()   LiveSpan    OTel Span    MlflowProcessor  MlflowClient  REST API
```

### Initialization Flow

1. User calls `configure({ tracking_uri, experiment_id })`
2. SDK reads Databricks credentials from `~/.databrickscfg`
3. OpenTelemetry tracer provider is initialized with MLflow processors
4. MLflow span processor and exporter are attached to OTel pipeline

### Span Lifecycle

1. **Creation**: User calls `withSpan()` or `startSpan()`
2. **OTel Integration**: Creates OpenTelemetry span for context propagation
3. **MLflow Wrapping**: Wraps OTel span in `LiveSpan` with MLflow metadata
4. **Registration**: Registers span with `InMemoryTraceManager`
5. **Execution**: User code runs within span context
6. **Completion**: Span ends, triggers export pipeline
7. **Export**: Only root spans trigger full trace export to backend

### Backend Communication Protocol

The SDK communicates with MLflow backends through REST APIs:

**Trace Lifecycle**:

- `POST /api/2.0/mlflow/traces` - Start new trace (creates TraceInfo)
- `PUT /api/2.0/mlflow/traces/{trace_id}` - End trace (uploads complete trace + spans)

**Trace Operations**:

- `GET /api/2.0/mlflow/traces/{trace_id}` - Retrieve single trace
- `POST /api/2.0/mlflow/traces/search` - Search traces with filtering
- `POST /api/2.0/mlflow/traces/batch` - Bulk trace upload

**Tag Management**:

- `POST /api/2.0/mlflow/traces/{trace_id}/tags` - Set trace tag
- `DELETE /api/2.0/mlflow/traces/{trace_id}/tags/{key}` - Delete trace tag

## Core Components

### Configuration System (`src/core/config.ts`)

- **Entry Point**: `configure()` must be called before tracing operations
- **Authentication**: Supports Databricks URIs (`databricks://profile` or `databricks`)
- **Credential Resolution**: Reads from `~/.databrickscfg` automatically
- **Validation**: Validates required fields (tracking_uri, experiment_id)

### High-Level APIs (`src/core/api.ts`)

**`withSpan(options?, callback)`**:

- Automatic span lifecycle management
- Supports both inline and options-based usage
- Handles async/sync callbacks automatically
- Sets outputs from return values if not manually set

**`startSpan(options)`**:

- Manual span creation requiring explicit `end()`
- Supports parent span relationships
- Returns `LiveSpan` for manipulation

### Entity System

**Span Hierarchy**:

- `ISpan` - Interface for all span types
- `Span` - Immutable completed spans (with caching)
- `LiveSpan` - Mutable active spans with setters
- `NoOpSpan` - Fallback when tracing disabled/fails

**Trace Structure**:

- `Trace` - Container holding `TraceInfo` + `TraceData`
- `TraceInfo` - Metadata (ID, timestamps, state, tags, metadata)
- `TraceData` - Collection of spans belonging to trace

**Supporting Entities**:

- `SpanEvent` - Point-in-time events (exceptions, logs)
- `SpanStatus` - Span completion status (OK/ERROR/UNSET)
- `TraceLocation` - Backend destination (experiment or inference table)

### Serialization System

All entities support JSON serialization for API communication:

- `toJson()` - Convert entity to API-compatible JSON
- `fromJson()` - Reconstruct entity from API JSON response
- Round-trip tested to ensure data integrity

**Key Serialization Patterns**:

- Snake_case JSON fields (matching Python MLflow)
- Timestamp conversion (ns ↔ ISO strings)
- Duration formatting (ms ↔ "1.5s" format)
- Nested object serialization for complex attributes

### Trace Management (`src/core/trace_manager.ts`)

**InMemoryTraceManager** (Singleton):

- Maps OpenTelemetry trace IDs → MLflow trace IDs
- Maintains active traces and spans in memory
- Thread-safe span registration and retrieval
- Provides trace export coordination

**Key Operations**:

- `registerTrace()` - Start new trace tracking
- `registerSpan()` - Add span to trace
- `getTrace()` - Retrieve trace context manager
- `popTrace()` - Remove and export completed trace

### OpenTelemetry Integration (`src/core/provider.ts`)

**Span Processing Pipeline**:

1. `MlflowSpanProcessor.onStart()` - Creates TraceInfo for root spans
2. Span execution with MLflow attribute management
3. `MlflowSpanProcessor.onEnd()` - Triggers export for root spans

**Context Management**:

- Uses OpenTelemetry's context propagation
- Parent-child span relationships automatically maintained
- Active span tracking for nested operations

### Export System (`src/exporters/mlflow.ts`)

**MlflowSpanExporter**:

- Exports complete traces (not individual spans)
- Currently stores in memory for testing
- Designed to integrate with MlflowClient for real export

**Export Triggers**:

- Only root span completion triggers export
- Child spans contribute to trace but don't trigger export
- Ensures complete traces are exported atomically

### MLflow Client (`src/clients/MlflowClient.ts`)

**Complete REST API Implementation**:

- Authentication via Bearer token from `.databrickscfg`
- Full error handling with HTTP status parsing
- Type-safe request/response interfaces
- Support for all MLflow tracing operations
- Configurable HTTP request timeouts (default 30s)

**Client Operations**:

- Trace lifecycle management
- Batch operations for performance
- Search and filtering capabilities
- Tag management for trace metadata
- Health checking for connectivity validation

**Timeout Configuration**:

- Default timeout: 30 seconds
- Environment variable: `MLFLOW_HTTP_REQUEST_TIMEOUT` (milliseconds)
- Constructor option: `timeoutMs` parameter
- Uses AbortController for clean request cancellation

**Error Handling**:

- HTTP error parsing with meaningful messages
- Timeout errors with clear messaging
- Graceful fallbacks to prevent user code breakage
- Retry logic considerations for production use

## Key Design Patterns

### OpenTelemetry Foundation

- All spans backed by OTel spans for ecosystem compatibility
- Context propagation handled by OpenTelemetry
- Processor/Exporter pattern for trace lifecycle management

### MLflow Compatibility

- API signatures match Python MLflow implementation
- JSON serialization compatible with MLflow REST API
- Entity structure mirrors Python classes

### Error Resilience

- NoOp spans when tracing fails
- Console warnings instead of throwing errors
- Graceful degradation to prevent application breakage

### Type Safety

- Full TypeScript coverage with strict typing
- Interface-based design for extensibility
- Comprehensive test coverage (65 tests)

### Memory Management

- In-memory trace storage with automatic cleanup
- Span deduplication before export
- Efficient attribute serialization/deserialization

## Backend Integration Points

### Authentication Flow

1. Parse `tracking_uri` for Databricks profile
2. Read credentials from `~/.databrickscfg`
3. Validate connectivity with health check
4. Use Bearer token for all API requests

### Trace Export Flow

1. Root span completion triggers export
2. Aggregate all child spans into complete trace
3. Serialize trace to MLflow JSON format
4. POST to backend with retry logic
5. Handle success/failure responses

### Data Consistency

- Atomic trace uploads (all spans together)
- Consistent timestamp formatting across entities
- Proper parent-child span relationships
- Metadata preservation through serialization

## Development Guidelines

### Adding New Features

- Follow existing entity patterns with toJson/fromJson
- Add comprehensive round-trip serialization tests
- Maintain OpenTelemetry integration patterns
- Update this documentation for architectural changes

### Testing Strategy

- Unit tests for each entity and component
- Round-trip serialization tests for all entities
- Integration tests for API workflows
- Mock objects for external dependencies

### Error Handling

- Use NoOp patterns for graceful degradation
- Log warnings for debugging without breaking execution
- Provide meaningful error messages for configuration issues
- Handle network failures gracefully in client operations
