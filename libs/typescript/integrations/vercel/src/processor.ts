import { Context, SpanStatusCode } from '@opentelemetry/api';
import {
  Span,
  ReadableSpan,
  SpanProcessor,
  SpanExporter,
  BatchSpanProcessor,
} from '@opentelemetry/sdk-trace-base';
import {
  MlflowClient,
  SpanAttributeKey,
  TraceInfo,
  TraceMetadataKey,
  TraceState,
  calculateCostByModelAndTokenUsage,
  constructTraceIdV4,
  getUcLocationString,
  type TraceLocation,
} from '@mlflow/core';
import { translateSpanForMlflow } from './translate';

const DEFAULT_MAX_TRACE_AGE_MS = 5 * 60 * 1000;
const DEFAULT_MAX_ACTIVE_TRACES = 1000;
const PREVIEW_MAX_LENGTH = 1000;

const TOKEN_KEYS = [
  'input_tokens',
  'output_tokens',
  'total_tokens',
  'cache_read_input_tokens',
  'cache_creation_input_tokens',
] as const;
const COST_KEYS = ['input_cost', 'output_cost', 'total_cost'] as const;

type TokenKey = (typeof TOKEN_KEYS)[number];
type CostKey = (typeof COST_KEYS)[number];
interface TraceAccumulator {
  endedSpans: Map<string, ReadableSpan>;
  activeSpanIds: Set<string>;
  rootSpan?: ReadableSpan;
  lastUpdatedAtMs: number;
  finalization?: Promise<void>;
}

/** Configuration for persisting completed UC trace metadata through MLflow V4. */
export interface V4TraceInfoOptions {
  /** Authenticated MLflow client used to call the V4 CreateTraceInfo endpoint. */
  client: MlflowClient;
  /** Unity Catalog destination for the trace. */
  traceLocation: TraceLocation;
  /** Explicit trace metadata. These values override synthesized token and cost totals. */
  traceMetadata?: Record<string, string>;
  /** Maximum time to retain an incomplete trace. Defaults to five minutes. */
  maxTraceAgeMs?: number;
  /** Maximum number of traces retained by the processor. Defaults to 1,000. */
  maxActiveTraces?: number;
  /** Receives non-fatal persistence and accumulator-eviction errors. */
  onTraceInfoError?: (error: unknown, traceId: string) => void;
}

export interface MLflowSpanProcessorOptions {
  /** Enables V4 TraceInfo persistence for Unity Catalog backed traces. */
  traceInfo?: V4TraceInfoOptions;
}

interface ResolvedTraceInfoOptions extends V4TraceInfoOptions {
  location: string;
  maxTraceAgeMs: number;
  maxActiveTraces: number;
}

/**
 * A SpanProcessor that translates Vercel AI SDK span attributes into
 * MLflow's expected format before batching and exporting.
 *
 * When V4 TraceInfo persistence is configured, the processor retains bounded
 * per-trace state and writes trace-level token and cost totals after the root
 * and all locally observed active spans have ended.
 */
export class MLflowSpanProcessor implements SpanProcessor {
  private readonly _processor: SpanProcessor;
  private readonly _traceInfo?: ResolvedTraceInfoOptions;
  private readonly _traces = new Map<string, TraceAccumulator>();
  private readonly _pendingTraceInfoWrites = new Set<Promise<void>>();

  constructor(exporter: SpanExporter, options: MLflowSpanProcessorOptions = {}) {
    this._processor = new BatchSpanProcessor(exporter);

    if (options.traceInfo) {
      const location = getUcLocationString(options.traceInfo.traceLocation);
      if (!location) {
        throw new Error('traceInfo.traceLocation must be a Unity Catalog table-prefix location.');
      }

      const maxTraceAgeMs = options.traceInfo.maxTraceAgeMs ?? DEFAULT_MAX_TRACE_AGE_MS;
      const maxActiveTraces = options.traceInfo.maxActiveTraces ?? DEFAULT_MAX_ACTIVE_TRACES;
      if (!Number.isFinite(maxTraceAgeMs) || maxTraceAgeMs <= 0) {
        throw new Error('traceInfo.maxTraceAgeMs must be a positive finite number.');
      }
      if (!Number.isInteger(maxActiveTraces) || maxActiveTraces <= 0) {
        throw new Error('traceInfo.maxActiveTraces must be a positive integer.');
      }

      this._traceInfo = {
        ...options.traceInfo,
        location,
        maxTraceAgeMs,
        maxActiveTraces,
      };
    }
  }

  onStart(span: Span, parentContext: Context): void {
    if (this._traceInfo) {
      try {
        const traceId = span.spanContext().traceId;
        const accumulator = this.getOrCreateAccumulator(traceId, Date.now());
        accumulator?.activeSpanIds.add(span.spanContext().spanId);
      } catch (error) {
        this.reportError(error, span.spanContext().traceId);
      }
    }
    this._processor.onStart(span, parentContext);
  }

  onEnd(span: ReadableSpan): void {
    translateSpanForMlflow(span);

    if (this._traceInfo) {
      try {
        enrichSpanCost(span);
      } catch (error) {
        this.reportError(error, span.spanContext().traceId);
      }
      try {
        this.accumulateEndedSpan(span);
      } catch (error) {
        this.reportError(error, span.spanContext().traceId);
      }
    }

    // TraceInfo persistence is best effort and must never prevent OTLP export.
    this._processor.onEnd(span);
  }

  async forceFlush(): Promise<void> {
    try {
      await this._processor.forceFlush();
    } finally {
      await this.waitForPendingTraceInfoWrites();
    }
  }

  async shutdown(): Promise<void> {
    this.finalizeRootedTracesForShutdown();
    try {
      await this._processor.shutdown();
    } finally {
      await this.waitForPendingTraceInfoWrites();
    }
  }

  private accumulateEndedSpan(span: ReadableSpan): void {
    const now = Date.now();
    const traceId = span.spanContext().traceId;
    const accumulator = this.getOrCreateAccumulator(traceId, now);
    if (!accumulator || accumulator.finalization) {
      return;
    }

    const spanId = span.spanContext().spanId;
    accumulator.endedSpans.set(spanId, span);
    accumulator.activeSpanIds.delete(spanId);
    accumulator.lastUpdatedAtMs = now;

    if (isRootSpan(span)) {
      accumulator.rootSpan = span;
    }

    if (accumulator.rootSpan && accumulator.activeSpanIds.size === 0) {
      this.finalizeTrace(traceId, accumulator);
    }
  }

  private getOrCreateAccumulator(traceId: string, now: number): TraceAccumulator | undefined {
    const existing = this._traces.get(traceId);
    if (existing) {
      existing.lastUpdatedAtMs = now;
      this.evictExpiredTraces(now, traceId);
      return existing;
    }

    this.evictExpiredTraces(now);

    if (!this.ensureCapacity(traceId)) {
      return undefined;
    }

    const accumulator: TraceAccumulator = {
      endedSpans: new Map(),
      activeSpanIds: new Set(),
      lastUpdatedAtMs: now,
    };
    this._traces.set(traceId, accumulator);
    return accumulator;
  }

  private evictExpiredTraces(now: number, activeTraceId?: string): void {
    if (!this._traceInfo) {
      return;
    }
    for (const [traceId, accumulator] of this._traces) {
      if (
        traceId !== activeTraceId &&
        !accumulator.finalization &&
        now - accumulator.lastUpdatedAtMs >= this._traceInfo.maxTraceAgeMs
      ) {
        this._traces.delete(traceId);
        this.reportError(
          new Error(`Evicted incomplete trace after ${this._traceInfo.maxTraceAgeMs} ms.`),
          traceId,
        );
      }
    }
  }

  private ensureCapacity(incomingTraceId: string): boolean {
    if (!this._traceInfo || this._traces.size < this._traceInfo.maxActiveTraces) {
      return true;
    }

    let oldest: [string, TraceAccumulator] | undefined;
    for (const entry of this._traces) {
      if (
        !entry[1].finalization &&
        (!oldest || entry[1].lastUpdatedAtMs < oldest[1].lastUpdatedAtMs)
      ) {
        oldest = entry;
      }
    }

    if (oldest) {
      this._traces.delete(oldest[0]);
      this.reportError(
        new Error(
          `Evicted incomplete trace to enforce maxActiveTraces=${this._traceInfo.maxActiveTraces}.`,
        ),
        oldest[0],
      );
      return true;
    }

    this.reportError(
      new Error(
        `Skipped trace accumulation because maxActiveTraces=${this._traceInfo.maxActiveTraces} is full.`,
      ),
      incomingTraceId,
    );
    return false;
  }

  private finalizeTrace(traceId: string, accumulator: TraceAccumulator): void {
    const options = this._traceInfo;
    if (!options || !accumulator.rootSpan || accumulator.finalization) {
      return;
    }

    const rootSpan = accumulator.rootSpan;
    const spans = Array.from(accumulator.endedSpans.values());
    let traceInfo: TraceInfo;
    try {
      traceInfo = buildTraceInfo(rootSpan, spans, options);
    } catch (error) {
      this._traces.delete(traceId);
      this.reportError(error, traceId);
      return;
    }

    const write = Promise.resolve()
      .then(() => options.client.createTraceInfoV4(options.location, traceId, traceInfo))
      .then(() => undefined)
      .catch((error: unknown) => {
        this.reportError(error, traceId);
      })
      .finally(() => {
        this._pendingTraceInfoWrites.delete(write);
        if (this._traces.get(traceId) === accumulator) {
          this._traces.delete(traceId);
        }
      });

    accumulator.finalization = write;
    this._pendingTraceInfoWrites.add(write);
  }

  private finalizeRootedTracesForShutdown(): void {
    for (const [traceId, accumulator] of this._traces) {
      if (accumulator.finalization) {
        continue;
      }
      if (accumulator.rootSpan) {
        // Best effort: a root that ended before all locally started children
        // still gets metadata from the spans that ended before shutdown.
        this.finalizeTrace(traceId, accumulator);
      } else {
        this._traces.delete(traceId);
        this.reportError(
          new Error('Discarded incomplete trace without an ended root at shutdown.'),
          traceId,
        );
      }
    }
  }

  private async waitForPendingTraceInfoWrites(): Promise<void> {
    while (this._pendingTraceInfoWrites.size > 0) {
      await Promise.all(Array.from(this._pendingTraceInfoWrites));
    }
  }

  private reportError(error: unknown, traceId: string): void {
    if (this._traceInfo?.onTraceInfoError) {
      try {
        this._traceInfo.onTraceInfoError(error, traceId);
        return;
      } catch (callbackError) {
        console.error('MLflowSpanProcessor: onTraceInfoError callback failed', callbackError);
      }
    }
    console.error(`MLflowSpanProcessor: failed to persist TraceInfo for ${traceId}`, error);
  }
}

function enrichSpanCost(span: ReadableSpan): void {
  const attributes = span.attributes as Record<string, unknown>;
  if (attributes[SpanAttributeKey.LLM_COST] !== undefined) {
    return;
  }

  const usage = parseMetric(attributes[SpanAttributeKey.TOKEN_USAGE], TOKEN_KEYS, false);
  if (!usage) {
    return;
  }

  const cost = calculateCostByModelAndTokenUsage(
    parseStringAttribute(attributes[SpanAttributeKey.MODEL]),
    usage,
    parseStringAttribute(attributes[SpanAttributeKey.MODEL_PROVIDER]),
  );
  if (cost) {
    attributes[SpanAttributeKey.LLM_COST] = JSON.stringify(cost);
  }
}

function parseStringAttribute(value: unknown): string | undefined {
  if (typeof value !== 'string' || value.length === 0) {
    return undefined;
  }
  try {
    const parsed = JSON.parse(value) as unknown;
    return typeof parsed === 'string' ? parsed : value;
  } catch {
    return value;
  }
}

function buildTraceInfo(
  rootSpan: ReadableSpan,
  spans: ReadableSpan[],
  options: ResolvedTraceInfoOptions,
): TraceInfo {
  const otelTraceId = rootSpan.spanContext().traceId;
  const tokenUsage = aggregateMetric(spans, SpanAttributeKey.TOKEN_USAGE, TOKEN_KEYS);
  const cost = aggregateMetric(spans, SpanAttributeKey.LLM_COST, COST_KEYS, true);
  const rootAttributes = rootSpan.attributes as Record<string, unknown>;

  const synthesizedMetadata: Record<string, string> = {
    [TraceMetadataKey.SCHEMA_VERSION]: '4',
  };
  if (tokenUsage) {
    synthesizedMetadata[TraceMetadataKey.TOKEN_USAGE] = JSON.stringify(tokenUsage);
  }
  if (cost) {
    synthesizedMetadata[TraceMetadataKey.COST] = JSON.stringify(cost);
  }

  const rootTraceMetadata: Record<string, string> = {};
  for (const key of [TraceMetadataKey.TOKEN_USAGE, TraceMetadataKey.COST]) {
    if (typeof rootAttributes[key] === 'string') {
      rootTraceMetadata[key] = rootAttributes[key];
    }
  }

  const requestTime = hrTimeToMs(rootSpan.startTime);
  return new TraceInfo({
    traceId: constructTraceIdV4(options.location, otelTraceId),
    traceLocation: options.traceLocation,
    requestTime,
    executionDuration: Math.max(0, hrTimeToMs(rootSpan.endTime) - requestTime),
    state: rootSpan.status.code === SpanStatusCode.ERROR ? TraceState.ERROR : TraceState.OK,
    requestPreview: toPreview(rootAttributes[SpanAttributeKey.INPUTS]),
    responsePreview: toPreview(rootAttributes[SpanAttributeKey.OUTPUTS]),
    traceMetadata: {
      ...synthesizedMetadata,
      ...rootTraceMetadata,
      ...options.traceMetadata,
    },
  });
}

function aggregateMetric<K extends TokenKey | CostKey>(
  spans: ReadableSpan[],
  attributeKey: string,
  keys: readonly K[],
  deriveCostTotal = false,
): Partial<Record<K, number>> | undefined {
  const parentBySpanId = new Map<string, string | undefined>();
  const metricsBySpanId = new Map<string, Partial<Record<K, number>>>();

  for (const span of spans) {
    const spanId = span.spanContext().spanId;
    parentBySpanId.set(spanId, getParentSpanId(span));
    const metric = parseMetric(span.attributes[attributeKey], keys, deriveCostTotal);
    if (metric) {
      metricsBySpanId.set(spanId, metric);
    }
  }

  const excludedAncestors = new Set<string>();
  for (const spanId of metricsBySpanId.keys()) {
    const visited = new Set<string>();
    let parentId = parentBySpanId.get(spanId);
    while (parentId && !visited.has(parentId)) {
      visited.add(parentId);
      if (metricsBySpanId.has(parentId)) {
        excludedAncestors.add(parentId);
      }
      parentId = parentBySpanId.get(parentId);
    }
  }

  const result: Partial<Record<K, number>> = {};
  let hasValue = false;
  for (const [spanId, metric] of metricsBySpanId) {
    if (excludedAncestors.has(spanId)) {
      continue;
    }
    for (const key of keys) {
      const value = metric[key];
      if (value !== undefined) {
        result[key] = (result[key] ?? 0) + value;
        hasValue = true;
      }
    }
  }
  return hasValue ? result : undefined;
}

function parseMetric<K extends TokenKey | CostKey>(
  value: unknown,
  keys: readonly K[],
  deriveCostTotal: boolean,
): Partial<Record<K, number>> | undefined {
  let parsed = value;
  if (typeof value === 'string') {
    try {
      parsed = JSON.parse(value) as unknown;
    } catch {
      return undefined;
    }
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return undefined;
  }

  const record = parsed as Record<string, unknown>;
  const result: Partial<Record<K, number>> = {};
  for (const key of keys) {
    const candidate = record[key];
    if (typeof candidate === 'number' && Number.isFinite(candidate)) {
      result[key] = candidate;
    }
  }

  if (deriveCostTotal && !('total_cost' in result)) {
    const input = result['input_cost' as K];
    const output = result['output_cost' as K];
    if (input !== undefined || output !== undefined) {
      result['total_cost' as K] = (input ?? 0) + (output ?? 0);
    }
  }

  return Object.keys(result).length > 0 ? result : undefined;
}

function getParentSpanId(span: ReadableSpan | Span): string | undefined {
  const parentSpanContext = (span as { parentSpanContext?: { spanId?: string } }).parentSpanContext;
  if (parentSpanContext?.spanId) {
    return parentSpanContext.spanId;
  }
  return (span as unknown as { parentSpanId?: string }).parentSpanId;
}

function isRootSpan(span: ReadableSpan): boolean {
  return !getParentSpanId(span);
}

function hrTimeToMs(time: readonly [number, number]): number {
  return Math.floor(time[0] * 1000 + time[1] / 1_000_000);
}

function toPreview(value: unknown): string | undefined {
  if (typeof value !== 'string' || value.length === 0) {
    return undefined;
  }
  return value.length <= PREVIEW_MAX_LENGTH
    ? value
    : `${value.slice(0, PREVIEW_MAX_LENGTH - 3)}...`;
}
