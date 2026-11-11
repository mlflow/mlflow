import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import { MlflowSpanExporter } from '../../src/exporters/mlflow';
import { serializeSpansToOtlpBytes } from '../../src/exporters/otlp';
import { InMemoryTraceManager } from '../../src/core/trace_manager';
import { getConfig } from '../../src/core/config';
import { MlflowHttpError } from '../../src/clients/utils';
import { MAX_OTLP_REQUEST_BYTES } from '../../src/core/constants';

jest.mock('../../src/exporters/otlp', () => {
  const actual = jest.requireActual<typeof import('../../src/exporters/otlp')>(
    '../../src/exporters/otlp',
  );
  return { ...actual, serializeSpansToOtlpBytes: jest.fn() };
});

jest.mock('../../src/core/trace_manager', () => {
  const instance = {
    popTrace: jest.fn(() => ({
      info: {
        traceId: 'tr-abc',
        traceLocation: {
          type: 'MLFLOW_EXPERIMENT',
          mlflowExperiment: { experimentId: 'exp-1' },
        },
        tags: {},
        traceMetadata: {},
        assessments: [],
      },
      data: { spans: [{ _span: { name: 'root-span' } }] },
    })),
    lastActiveTraceId: null,
  };
  return { InMemoryTraceManager: { getInstance: () => instance } };
});

const popTraceMock = (InMemoryTraceManager.getInstance() as unknown as { popTrace: jest.Mock })
  .popTrace;

jest.mock('../../src/core/config', () => ({
  getConfig: jest.fn(() => ({ experimentId: 'exp-1', trackingUri: 'http://localhost:5000' })),
}));

const mockedGetConfig = getConfig as jest.MockedFunction<typeof getConfig>;
const mockedSerialize = serializeSpansToOtlpBytes as jest.MockedFunction<
  typeof serializeSpansToOtlpBytes
>;

function makeRootSpan(): ReadableSpan {
  return {
    spanContext: () => ({ traceId: 'abc123', spanId: 'def456' }),
  } as unknown as ReadableSpan;
}

describe('MlflowSpanExporter', () => {
  let client: {
    createTrace: jest.Mock;
    uploadTraceData: jest.Mock;
    exportOtlpSpans: jest.Mock;
  };
  let exporter: MlflowSpanExporter;

  beforeEach(() => {
    jest.clearAllMocks();
    mockedGetConfig.mockReturnValue({
      experimentId: 'exp-1',
      trackingUri: 'http://localhost:5000',
    } as ReturnType<typeof getConfig>);
    mockedSerialize.mockReturnValue(undefined);
    client = {
      createTrace: jest.fn((info) => Promise.resolve(info)),
      uploadTraceData: jest.fn().mockResolvedValue(undefined),
      exportOtlpSpans: jest.fn().mockResolvedValue(undefined),
    };
    exporter = new MlflowSpanExporter(client as any);
  });

  it('sends OTLP spans to the tracking store without waiting for the artifact upload', async () => {
    const bytes = new Uint8Array([1, 2, 3]);
    mockedSerialize.mockReturnValue(bytes);
    let finishUpload!: () => void;
    client.uploadTraceData.mockReturnValue(
      new Promise<void>((resolve) => {
        finishUpload = resolve;
      }),
    );

    exporter.export([makeRootSpan()], () => {});
    await new Promise((resolve) => setImmediate(resolve));

    // The artifact upload is still in flight, yet the spans were already sent.
    expect(client.uploadTraceData).toHaveBeenCalledTimes(1);
    expect(client.exportOtlpSpans).toHaveBeenCalledTimes(1);
    expect(client.exportOtlpSpans).toHaveBeenCalledWith('exp-1', bytes);

    finishUpload();
    await exporter.forceFlush();
  });

  it('reports the export as failed when the artifact upload fails', async () => {
    const errorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    try {
      mockedSerialize.mockReturnValue(new Uint8Array([1]));
      client.uploadTraceData.mockRejectedValue(new Error('artifact store down'));

      exporter.export([makeRootSpan()], () => {});
      await exporter.forceFlush();

      expect(client.exportOtlpSpans).toHaveBeenCalledTimes(1);
      expect(errorSpy).toHaveBeenCalledWith(
        expect.stringContaining('Failed to export trace tr-abc'),
        expect.objectContaining({ message: 'artifact store down' }),
      );
    } finally {
      errorSpy.mockRestore();
    }
  });

  it('attributes spans to the experiment persisted by createTrace', async () => {
    mockedGetConfig.mockReturnValue({
      experimentId: 'exp-config',
      trackingUri: 'http://localhost:5000',
    } as ReturnType<typeof getConfig>);
    client.createTrace.mockImplementation((info) =>
      Promise.resolve({
        ...info,
        traceLocation: {
          type: 'MLFLOW_EXPERIMENT',
          mlflowExperiment: { experimentId: 'exp-trace' },
        },
      }),
    );
    const bytes = new Uint8Array([9]);
    mockedSerialize.mockReturnValue(bytes);

    exporter.export([makeRootSpan()], () => {});
    await exporter.forceFlush();

    expect(client.exportOtlpSpans).toHaveBeenCalledWith('exp-trace', bytes);
  });

  it('skips OTLP span ingestion for Databricks tracking URIs', async () => {
    mockedGetConfig.mockReturnValue({
      experimentId: 'exp-1',
      trackingUri: 'databricks',
    } as ReturnType<typeof getConfig>);
    mockedSerialize.mockReturnValue(new Uint8Array([1]));

    exporter.export([makeRootSpan()], () => {});
    await exporter.forceFlush();

    expect(mockedSerialize).toHaveBeenCalled();
    expect(client.exportOtlpSpans).not.toHaveBeenCalled();
    expect(client.uploadTraceData).toHaveBeenCalledTimes(1);
  });

  it('skips OTLP span ingestion when the trace has no MLflow experiment', async () => {
    client.createTrace.mockImplementation((info) =>
      Promise.resolve({ ...info, traceLocation: { type: 'INFERENCE_TABLE' } }),
    );
    mockedSerialize.mockReturnValue(new Uint8Array([1]));

    exporter.export([makeRootSpan()], () => {});
    await exporter.forceFlush();

    expect(mockedSerialize).not.toHaveBeenCalled();
    expect(client.exportOtlpSpans).not.toHaveBeenCalled();
  });

  it('skips OTLP span ingestion when the payload exceeds the request budget', async () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      mockedSerialize.mockReturnValue({
        length: MAX_OTLP_REQUEST_BYTES + 1,
      } as Uint8Array);

      exporter.export([makeRootSpan()], () => {});
      await exporter.forceFlush();

      expect(client.exportOtlpSpans).not.toHaveBeenCalled();
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('exceeding the'));
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('warns once and stops sending after span ingestion is reported unsupported', async () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      mockedSerialize.mockReturnValue(new Uint8Array([1]));
      client.exportOtlpSpans.mockRejectedValue(new MlflowHttpError(501, 'Not Implemented', ''));

      exporter.export([makeRootSpan()], () => {});
      await exporter.forceFlush();
      exporter.export([makeRootSpan()], () => {});
      await exporter.forceFlush();

      expect(client.exportOtlpSpans).toHaveBeenCalledTimes(1);
      expect(warnSpy).toHaveBeenCalledTimes(1);
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('not supported'));
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('latches when the server has no span-ingestion route', async () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      mockedSerialize.mockReturnValue(new Uint8Array([1]));
      client.exportOtlpSpans.mockRejectedValue(new MlflowHttpError(404, 'Not Found', ''));

      exporter.export([makeRootSpan()], () => {});
      await exporter.forceFlush();
      exporter.export([makeRootSpan()], () => {});
      await exporter.forceFlush();

      expect(client.exportOtlpSpans).toHaveBeenCalledTimes(1);
      expect(warnSpy).toHaveBeenCalledTimes(1);
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('keeps sending after a 404 that is about one request, not the route', async () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      mockedSerialize.mockReturnValue(new Uint8Array([1]));
      client.exportOtlpSpans.mockRejectedValue(
        new MlflowHttpError(
          404,
          'Not Found',
          JSON.stringify({ error_code: 'RESOURCE_DOES_NOT_EXIST', message: 'no such experiment' }),
        ),
      );

      exporter.export([makeRootSpan()], () => {});
      await exporter.forceFlush();
      exporter.export([makeRootSpan()], () => {});
      await exporter.forceFlush();

      expect(client.exportOtlpSpans).toHaveBeenCalledTimes(2);
      expect(warnSpy).toHaveBeenCalledTimes(2);
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Failed to send OTLP spans'));
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('still succeeds when the span-ingestion endpoint fails', async () => {
    mockedSerialize.mockReturnValue(new Uint8Array([1]));
    client.exportOtlpSpans.mockRejectedValue(new Error('501 not supported'));

    exporter.export([makeRootSpan()], () => {});
    await expect(exporter.forceFlush()).resolves.toBeUndefined();

    expect(client.uploadTraceData).toHaveBeenCalledTimes(1);
    expect(client.exportOtlpSpans).toHaveBeenCalledTimes(1);
  });

  it('skips non-root spans', async () => {
    exporter.export(
      [
        {
          spanContext: () => ({ traceId: 'abc', spanId: 'd' }),
          parentSpanContext: { spanId: 'p' },
        } as any,
      ],
      () => {},
    );
    await exporter.forceFlush();

    expect(popTraceMock).not.toHaveBeenCalled();
    expect(client.exportOtlpSpans).not.toHaveBeenCalled();
  });

  it('does not call the ingestion endpoint when serialization yields nothing', async () => {
    mockedSerialize.mockReturnValue(undefined);

    exporter.export([makeRootSpan()], () => {});
    await exporter.forceFlush();

    expect(client.exportOtlpSpans).not.toHaveBeenCalled();
    expect(client.uploadTraceData).toHaveBeenCalledTimes(1);
  });
});
