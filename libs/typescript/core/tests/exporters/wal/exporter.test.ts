import { mkdtemp, rm } from 'fs/promises';
import { tmpdir } from 'os';
import { join } from 'path';
import { ROOT_CONTEXT, SpanKind } from '@opentelemetry/api';
import { ExportResult, ExportResultCode } from '@opentelemetry/core';
import {
  BasicTracerProvider,
  ReadableSpan as OTelReadableSpan,
  Span as OTelSpan,
} from '@opentelemetry/sdk-trace-base';
import { ProtobufTraceSerializer } from '@opentelemetry/otlp-transformer';
import { MlflowWalSpanExporter } from '../../../src/exporters/wal/exporter';
import { init, resetConfig } from '../../../src/core/config';
import { InMemoryTraceManager } from '../../../src/core/trace_manager';
import { Trace } from '../../../src/core/entities/trace';
import { Span as MlflowSpan, NoOpSpan } from '../../../src/core/entities/span';
import { TraceInfo } from '../../../src/core/entities/trace_info';
import { TraceData } from '../../../src/core/entities/trace_data';
import { createTraceLocationFromExperimentId } from '../../../src/core/entities/trace_location';
import { TraceState } from '../../../src/core/entities/trace_state';
import { SpanAttributeKey } from '../../../src/core/constants';
import type { WalRecord } from '../../../src/exporters/wal/types';

function makeTrace(traceId: string): Trace {
  const info = new TraceInfo({
    traceId,
    traceLocation: createTraceLocationFromExperimentId('exp-test'),
    requestTime: 1_700_000_000_000,
    state: TraceState.OK,
    executionDuration: 42,
    traceMetadata: { 'mlflow.trace.schemaVersion': '3' },
    tags: {},
    assessments: [],
  });
  const data = new TraceData([]);
  return new Trace(info, data);
}

/**
 * Build a trace whose `data.spans` carries real, ended OTel spans (wrapped in
 * MLflow `Span`s) so the exporter's OTLP serialization runs end-to-end
 */
function makeTraceWithSpans(traceId: string): Trace {
  const info = new TraceInfo({
    traceId,
    traceLocation: createTraceLocationFromExperimentId('exp-test'),
    requestTime: 1_700_000_000_000,
    state: TraceState.OK,
    executionDuration: 42,
    traceMetadata: { 'mlflow.trace.schemaVersion': '3' },
    tags: {},
    assessments: [],
  });

  const provider = new BasicTracerProvider();
  const tracer = provider.getTracer('wal-exporter-test');
  const otelSpan = tracer.startSpan(
    'tool-call',
    { kind: SpanKind.INTERNAL },
    ROOT_CONTEXT,
  ) as OTelSpan;
  otelSpan.setAttribute(SpanAttributeKey.TRACE_ID, JSON.stringify(traceId));
  otelSpan.end();

  return new Trace(info, new TraceData([new MlflowSpan(otelSpan)]));
}

function makeSpan(opts: { traceId: string; rootSpan: boolean; name?: string }): OTelReadableSpan {
  return {
    name: opts.name ?? (opts.rootSpan ? 'root' : 'child'),
    spanContext: () => ({ traceId: opts.traceId, spanId: 'span-id' }),
    parentSpanContext: opts.rootSpan ? undefined : { spanId: 'parent-span-id' },
  } as unknown as OTelReadableSpan;
}

function captureExportResult(): {
  callback: (r: ExportResult) => void;
  promise: Promise<ExportResult>;
} {
  let resolveFn: (r: ExportResult) => void = () => {};
  const promise = new Promise<ExportResult>((resolve) => {
    resolveFn = resolve;
  });
  return { callback: resolveFn, promise };
}

describe('wal/exporter', () => {
  let walDir: string;
  let popTraceSpy: jest.SpyInstance;
  // Capture the developer's pre-test value so we restore (not just unset)
  // in afterEach. Mirrors the pattern in paths.test.ts so running
  // `MLFLOW_WAL_DIR=/some/dir jest` doesn't lose the override for later
  // tests in the same worker.
  const originalWalDir = process.env.MLFLOW_WAL_DIR;

  beforeEach(async () => {
    walDir = await mkdtemp(join(tmpdir(), 'mlflow-wal-exporter-'));
    process.env.MLFLOW_WAL_DIR = walDir;

    init({ trackingUri: 'http://localhost:5000', experimentId: 'exp-test' });

    popTraceSpy = jest.spyOn(InMemoryTraceManager.getInstance(), 'popTrace');
  });

  afterEach(async () => {
    popTraceSpy.mockRestore();
    InMemoryTraceManager.reset();
    if (originalWalDir === undefined) {
      delete process.env.MLFLOW_WAL_DIR;
    } else {
      process.env.MLFLOW_WAL_DIR = originalWalDir;
    }
    await rm(walDir, { recursive: true, force: true });
  });

  it('submits one WAL record per root span via the injected submitter', async () => {
    const submit = jest.fn<Promise<void>, [WalRecord]>().mockResolvedValue(undefined);
    const exporter = new MlflowWalSpanExporter({ submit });

    popTraceSpy.mockReturnValue(makeTrace('tr-abc'));
    const span = makeSpan({ traceId: 'otel-1', rootSpan: true });

    const { callback, promise } = captureExportResult();
    exporter.export([span], callback);

    expect((await promise).code).toBe(ExportResultCode.SUCCESS);
    await exporter.forceFlush();

    expect(submit).toHaveBeenCalledTimes(1);
    const record = submit.mock.calls[0][0];
    expect(record.trackingUri).toBe('http://localhost:5000');
    expect(record.experimentId).toBe('exp-test');
    expect((record.traceInfo as { trace_id: string }).trace_id).toBe('tr-abc');
    expect(typeof record.id).toBe('string');
    expect(record.id.length).toBeGreaterThan(0);
  });

  it('skips non-root spans without invoking the submitter', async () => {
    const submit = jest.fn<Promise<void>, [WalRecord]>().mockResolvedValue(undefined);
    const exporter = new MlflowWalSpanExporter({ submit });

    const childSpan = makeSpan({ traceId: 'otel-1', rootSpan: false });
    const { callback, promise } = captureExportResult();
    exporter.export([childSpan], callback);

    expect((await promise).code).toBe(ExportResultCode.SUCCESS);
    expect(submit).not.toHaveBeenCalled();
    expect(popTraceSpy).not.toHaveBeenCalled();

    await exporter.forceFlush();
  });

  it('warns and skips when popTrace returns null', async () => {
    const submit = jest.fn<Promise<void>, [WalRecord]>().mockResolvedValue(undefined);
    const exporter = new MlflowWalSpanExporter({ submit });

    popTraceSpy.mockReturnValue(null);
    const span = makeSpan({ traceId: 'otel-missing', rootSpan: true, name: 'orphan' });

    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const { callback, promise } = captureExportResult();
      exporter.export([span], callback);

      expect((await promise).code).toBe(ExportResultCode.SUCCESS);
      expect(submit).not.toHaveBeenCalled();
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('No trace found for span orphan'),
      );

      await exporter.forceFlush();
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('handles a batch with mixed root and child spans', async () => {
    const submit = jest.fn<Promise<void>, [WalRecord]>().mockResolvedValue(undefined);
    const exporter = new MlflowWalSpanExporter({ submit });

    popTraceSpy.mockReturnValueOnce(makeTrace('tr-1')).mockReturnValueOnce(makeTrace('tr-2'));

    const spans = [
      makeSpan({ traceId: 'otel-1', rootSpan: true, name: 'root-1' }),
      makeSpan({ traceId: 'otel-1', rootSpan: false, name: 'child' }),
      makeSpan({ traceId: 'otel-2', rootSpan: true, name: 'root-2' }),
    ];
    const { callback, promise } = captureExportResult();
    exporter.export(spans, callback);

    expect((await promise).code).toBe(ExportResultCode.SUCCESS);
    expect(submit).toHaveBeenCalledTimes(2);
    expect(popTraceSpy).toHaveBeenCalledTimes(2);

    await exporter.forceFlush();

    const traceIds = submit.mock.calls
      .map(([record]) => (record.traceInfo as { trace_id: string }).trace_id)
      .sort();
    expect(traceIds).toEqual(['tr-1', 'tr-2']);
  });

  it('forceFlush resolves only once every in-flight submit has acked', async () => {
    let release: (() => void) | undefined;
    const submit = jest.fn<Promise<void>, [WalRecord]>().mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          release = resolve;
        }),
    );
    const exporter = new MlflowWalSpanExporter({ submit });

    popTraceSpy.mockReturnValue(makeTrace('tr-flush'));
    const span = makeSpan({ traceId: 'otel-1', rootSpan: true });

    const { callback } = captureExportResult();
    exporter.export([span], callback);

    let flushResolved = false;
    const flush = exporter.forceFlush().then(() => {
      flushResolved = true;
    });
    // The submit is parked on `release`; until we resolve it, forceFlush
    // must remain pending. Yield a few microtasks to give any premature
    // resolution a chance to surface.
    await new Promise((resolve) => setImmediate(resolve));
    await new Promise((resolve) => setImmediate(resolve));
    expect(flushResolved).toBe(false);

    release?.();
    await flush;
    expect(flushResolved).toBe(true);
  });

  it('shutdown awaits in-flight submits before resolving', async () => {
    let release: (() => void) | undefined;
    const submit = jest.fn<Promise<void>, [WalRecord]>().mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          release = resolve;
        }),
    );
    const exporter = new MlflowWalSpanExporter({ submit });

    popTraceSpy.mockReturnValue(makeTrace('tr-shutdown'));
    const { callback } = captureExportResult();
    exporter.export([makeSpan({ traceId: 'otel-1', rootSpan: true })], callback);

    let shutdownResolved = false;
    const shutdownPromise = exporter.shutdown().then(() => {
      shutdownResolved = true;
    });

    // The submit is parked; shutdown must remain pending until we
    // release it. Yield a few microtasks so any premature resolution
    // surfaces here rather than silently passing.
    await new Promise((resolve) => setImmediate(resolve));
    await new Promise((resolve) => setImmediate(resolve));
    expect(shutdownResolved).toBe(false);

    release?.();
    await shutdownPromise;
    expect(shutdownResolved).toBe(true);
  });

  it('shutdown rejects subsequent export calls with FAILED per OTel spec', async () => {
    // OTel SpanExporter.Shutdown spec: "After the call to Shutdown
    // subsequent calls to Export are not allowed and SHOULD return
    // a failure." Locks in the post-shutdown contract.
    const submit = jest.fn<Promise<void>, [WalRecord]>().mockResolvedValue(undefined);
    const exporter = new MlflowWalSpanExporter({ submit });
    await exporter.shutdown();

    popTraceSpy.mockReturnValue(makeTrace('tr-post-shutdown'));
    const { callback, promise } = captureExportResult();
    exporter.export([makeSpan({ traceId: 'otel-post', rootSpan: true })], callback);

    expect((await promise).code).toBe(ExportResultCode.FAILED);
    expect(submit).not.toHaveBeenCalled();
    expect(popTraceSpy).not.toHaveBeenCalled();
  });

  it('shutdown awaits all in-flight submits, not just the first to resolve', async () => {
    // Both `export()` calls run synchronously to completion before
    // `shutdown()` is invoked, so both submit promises are already
    // in `_pendingSubmits` when the drain loop takes its first
    // snapshot. (The `_shuttingDown` flag in `export()` makes the
    // "submit lands after the flag is set" scenario structurally
    // impossible in single-threaded JS — once the flag flips, every
    // export call short-circuits with FAILED before touching the
    // Set.
    let releaseFirst: (() => void) | undefined;
    let releaseSecond: (() => void) | undefined;
    const submit = jest
      .fn<Promise<void>, [WalRecord]>()
      .mockImplementationOnce(
        () =>
          new Promise<void>((resolve) => {
            releaseFirst = resolve;
          }),
      )
      .mockImplementationOnce(
        () =>
          new Promise<void>((resolve) => {
            releaseSecond = resolve;
          }),
      );
    const exporter = new MlflowWalSpanExporter({ submit });

    popTraceSpy
      .mockReturnValueOnce(makeTrace('tr-first'))
      .mockReturnValueOnce(makeTrace('tr-second'));

    // Two submits enter `_pendingSubmits` synchronously, parked on
    // their respective `release*` resolvers.
    const cb1 = captureExportResult();
    exporter.export([makeSpan({ traceId: 'otel-1', rootSpan: true })], cb1.callback);
    const cb2 = captureExportResult();
    exporter.export([makeSpan({ traceId: 'otel-2', rootSpan: true })], cb2.callback);

    let shutdownResolved = false;
    const shutdownPromise = exporter.shutdown().then(() => {
      shutdownResolved = true;
    });

    // Release submit #1 only. #2 is still parked, so `Promise.all`
    // inside the drain loop must remain pending — proving shutdown
    // awaits both, not just one.
    await new Promise((resolve) => setImmediate(resolve));
    releaseFirst?.();
    await new Promise((resolve) => setImmediate(resolve));
    await new Promise((resolve) => setImmediate(resolve));
    expect(shutdownResolved).toBe(false);

    releaseSecond?.();
    await shutdownPromise;
    expect(shutdownResolved).toBe(true);
    expect(submit).toHaveBeenCalledTimes(2);
  });

  it('returns FAILED and warns when getConfig() throws (init not called)', async () => {
    resetConfig();

    const submit = jest.fn<Promise<void>, [WalRecord]>().mockResolvedValue(undefined);
    const exporter = new MlflowWalSpanExporter({ submit });

    popTraceSpy.mockReturnValue(makeTrace('tr-no-config'));
    const span = makeSpan({ traceId: 'otel-no-config', rootSpan: true });

    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const { callback, promise } = captureExportResult();
      exporter.export([span], callback);

      const result = await promise;
      expect(result.code).toBe(ExportResultCode.FAILED);
      expect(submit).not.toHaveBeenCalled();
      expect(popTraceSpy).not.toHaveBeenCalled();
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Tracing config unavailable'),
        expect.any(Error),
      );
    } finally {
      warnSpy.mockRestore();
      // Restore config defensively so a downstream test that forgets
      // to re-init doesn't observe the cleared global state.
      init({ trackingUri: 'http://localhost:5000', experimentId: 'exp-test' });
    }
  });

  it('logs and swallows submit failures so OTel never observes them', async () => {
    const submit = jest
      .fn<Promise<void>, [WalRecord]>()
      .mockRejectedValue(new Error('daemon unreachable'));
    const exporter = new MlflowWalSpanExporter({ submit });

    popTraceSpy.mockReturnValue(makeTrace('tr-err'));
    const span = makeSpan({ traceId: 'otel-1', rootSpan: true });

    const errSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    try {
      const { callback, promise } = captureExportResult();
      exporter.export([span], callback);

      // OTel sees SUCCESS regardless of submit outcome — the exporter
      // owns durability semantics and must not bubble per-record
      // failures into the OTel result callback (which would surface as
      // an SDK-level error users can't act on).
      expect((await promise).code).toBe(ExportResultCode.SUCCESS);

      await exporter.forceFlush();
      expect(submit).toHaveBeenCalledTimes(1);
      expect(errSpy).toHaveBeenCalledWith(
        expect.stringContaining('Failed to submit WAL record for trace tr-err'),
        expect.any(Error),
      );
    } finally {
      errSpy.mockRestore();
    }
  });

  it('captures OTLP protobuf bytes on the record when the trace has spans', async () => {
    const submit = jest.fn<Promise<void>, [WalRecord]>().mockResolvedValue(undefined);
    const exporter = new MlflowWalSpanExporter({ submit });

    popTraceSpy.mockReturnValue(makeTraceWithSpans('tr-with-spans'));
    const span = makeSpan({ traceId: 'otel-1', rootSpan: true });

    const { callback, promise } = captureExportResult();
    exporter.export([span], callback);

    expect((await promise).code).toBe(ExportResultCode.SUCCESS);
    await exporter.forceFlush();

    const record = submit.mock.calls[0][0];
    expect(typeof record.otlpSpans).toBe('string');
    const decoded = Buffer.from(record.otlpSpans as string, 'base64');

    expect(decoded[0]).toBe(0x0a);
    // And the span itself must be encoded in the payload, not just an empty
    // envelope: its name is embedded as a UTF-8 string in the protobuf.
    expect(decoded.includes(Buffer.from('tool-call'))).toBe(true);
    // The JSON trace data is retained alongside the OTLP bytes (artifact fallback).
    expect((record.traceData as { spans: unknown[] }).spans).toHaveLength(1);
  });

  it('omits otlpSpans when the trace has no spans', async () => {
    const submit = jest.fn<Promise<void>, [WalRecord]>().mockResolvedValue(undefined);
    const exporter = new MlflowWalSpanExporter({ submit });

    popTraceSpy.mockReturnValue(makeTrace('tr-no-spans'));
    const span = makeSpan({ traceId: 'otel-1', rootSpan: true });

    const { callback, promise } = captureExportResult();
    exporter.export([span], callback);

    expect((await promise).code).toBe(ExportResultCode.SUCCESS);
    await exporter.forceFlush();

    const record = submit.mock.calls[0][0];
    expect(record.otlpSpans).toBeUndefined();
  });

  it('filters malformed spans (missing resource/scope) and still serializes the valid ones', async () => {
    const submit = jest.fn<Promise<void>, [WalRecord]>().mockResolvedValue(undefined);
    const exporter = new MlflowWalSpanExporter({ submit });

    const trace = makeTraceWithSpans('tr-mixed');
    trace.data.spans.push(new NoOpSpan());
    popTraceSpy.mockReturnValue(trace);
    const span = makeSpan({ traceId: 'otel-1', rootSpan: true });

    const { callback, promise } = captureExportResult();
    exporter.export([span], callback);

    expect((await promise).code).toBe(ExportResultCode.SUCCESS);
    await exporter.forceFlush();

    const record = submit.mock.calls[0][0];
    expect(typeof record.otlpSpans).toBe('string');
    const decoded = Buffer.from(record.otlpSpans as string, 'base64');
    expect(decoded[0]).toBe(0x0a);
    expect(decoded.includes(Buffer.from('tool-call'))).toBe(true);
  });

  it('warns and submits without otlpSpans when OTLP serialization throws', async () => {
    const serializeSpy = jest
      .spyOn(ProtobufTraceSerializer, 'serializeRequest')
      .mockImplementation(() => {
        throw new Error('boom');
      });
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const submit = jest.fn<Promise<void>, [WalRecord]>().mockResolvedValue(undefined);
    const exporter = new MlflowWalSpanExporter({ submit });

    popTraceSpy.mockReturnValue(makeTraceWithSpans('tr-serialize-fail'));
    const span = makeSpan({ traceId: 'otel-1', rootSpan: true });

    try {
      const { callback, promise } = captureExportResult();
      exporter.export([span], callback);

      expect((await promise).code).toBe(ExportResultCode.SUCCESS);
      await exporter.forceFlush();

      // The record is still submitted (artifact fallback), just without spans.
      expect(submit).toHaveBeenCalledTimes(1);
      expect(submit.mock.calls[0][0].otlpSpans).toBeUndefined();
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Failed to serialize spans to OTLP protobuf'),
        expect.any(Error),
      );
    } finally {
      serializeSpy.mockRestore();
      warnSpy.mockRestore();
    }
  });
});
