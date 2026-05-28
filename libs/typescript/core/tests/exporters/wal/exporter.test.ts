import { mkdtemp, rm } from 'fs/promises';
import { tmpdir } from 'os';
import { join } from 'path';
import { ExportResult, ExportResultCode } from '@opentelemetry/core';
import { ReadableSpan as OTelReadableSpan } from '@opentelemetry/sdk-trace-base';
import { MlflowWalSpanExporter } from '../../../src/exporters/wal/exporter';
import { init } from '../../../src/core/config';
import { InMemoryTraceManager } from '../../../src/core/trace_manager';
import { Trace } from '../../../src/core/entities/trace';
import { TraceInfo } from '../../../src/core/entities/trace_info';
import { TraceData } from '../../../src/core/entities/trace_data';
import { createTraceLocationFromExperimentId } from '../../../src/core/entities/trace_location';
import { TraceState } from '../../../src/core/entities/trace_state';
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

  it('shutdown defers to forceFlush', async () => {
    const exporter = new MlflowWalSpanExporter({ submit: () => Promise.resolve() });
    const flushSpy = jest.spyOn(exporter, 'forceFlush');
    await exporter.shutdown();
    expect(flushSpy).toHaveBeenCalledTimes(1);
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
});
