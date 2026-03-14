import { AsyncLocalStorage } from 'node:async_hooks';

/**
 * Options for the tracingContext function.
 */
export interface TracingContextOptions {
  /**
   * Key-value pairs to inject into the trace's request_metadata (immutable after trace creation).
   */
  metadata?: Record<string, string>;

  /**
   * Key-value pairs to inject into the trace's tags.
   */
  tags?: Record<string, string>;

  /**
   * Whether tracing is enabled within the scope. If false, all tracing calls within the scope
   * will return NoOpSpan without creating any traces. If undefined, the value is inherited from
   * the outer scope.
   */
  enabled?: boolean;
}

interface UserTraceContext {
  metadata: Record<string, string>;
  tags: Record<string, string>;
  enabled: boolean | undefined;
}

const storage = new AsyncLocalStorage<UserTraceContext>();

/**
 * Get the configured trace metadata from the current tracing context scope.
 * Returns undefined if no context is active or metadata is empty.
 */
export function getConfiguredTraceMetadata(): Record<string, string> | undefined {
  const ctx = storage.getStore();
  if (!ctx || Object.keys(ctx.metadata).length === 0) {
    return undefined;
  }
  return ctx.metadata;
}

/**
 * Get the configured trace tags from the current tracing context scope.
 * Returns undefined if no context is active or tags are empty.
 */
export function getConfiguredTraceTags(): Record<string, string> | undefined {
  const ctx = storage.getStore();
  if (!ctx || Object.keys(ctx.tags).length === 0) {
    return undefined;
  }
  return ctx.tags;
}

/**
 * Check if tracing is enabled in the current context scope.
 * Returns undefined if no context is active or enabled was not set.
 */
export function isTracingEnabledInContext(): boolean | undefined {
  const ctx = storage.getStore();
  return ctx?.enabled;
}

/**
 * Run a function within a tracing context scope that injects metadata and/or tags into any
 * trace created within the scope, without creating a wrapper span. It can also be used to
 * selectively disable tracing within the scope.
 *
 * This is useful when you need to attach trace-level information (e.g. session IDs) to traces
 * produced by code you don't control like auto-instrumented libraries, or when you want to
 * suppress tracing for a specific code block.
 *
 * The context can be nested. When the same key is specified in multiple levels, the value
 * from the inner level takes precedence.
 *
 * @param options - Metadata, tags, and enabled flag to inject into traces
 * @param fn - The function to execute within the context scope
 * @returns The return value of the function
 *
 * @example
 * ```typescript
 * import * as mlflow from "@mlflow/core";
 *
 * // Inject metadata and tags into all traces created within the scope
 * mlflow.tracingContext(
 *   { metadata: { "mlflow.trace.session": "session-123" }, tags: { project: "my-project" } },
 *   () => {
 *     // Any trace created here will carry the metadata and tags
 *     agent.invoke("What is the capital of France?");
 *   }
 * );
 *
 * // Disable tracing within a specific scope
 * mlflow.tracingContext({ enabled: false }, () => {
 *   // No traces will be created inside this scope
 *   agent.invoke("This call will not be traced");
 * });
 * ```
 */
export function tracingContext<T>(options: TracingContextOptions, fn: () => T): T {
  const current = storage.getStore();

  // Merge with any outer context scope (inner wins on conflict)
  const mergedMetadata = { ...(current?.metadata ?? {}), ...(options.metadata ?? {}) };
  const mergedTags = { ...(current?.tags ?? {}), ...(options.tags ?? {}) };
  const resolvedEnabled =
    options.enabled !== undefined ? options.enabled : (current?.enabled ?? undefined);

  const newContext: UserTraceContext = {
    metadata: mergedMetadata,
    tags: mergedTags,
    enabled: resolvedEnabled,
  };

  return storage.run(newContext, fn);
}
