import { MlflowSpanExporter } from '../../src/exporters/mlflow';
import { serializeSpansToOtlpBytes } from '../../src/exporters/otlp';
import { InMemoryTraceManager } from '../../src/core/trace_manager';

jest.mock('../../src/exporters/otlp', () => ({
  serializeSpansToOtlpBytes: jest.fn(),
}));

jest.mock('../../src/core/trace_manager', () => {
  const instance = {
    popTrace: jest.fn(() => ({
      info: { traceId: 'tr-abc', tags: {}, traceMetadata: {}, assessments: [] },
      data: { spans: [{ _span: { name: 'root-span' } }] },
    })),
    lastActiveTraceId: null,
  };
  return { InMemoryTraceManager: { getInstance: () => instance } };
});

const popTraceMock = InMemoryTraceManager.getInstance().popTrace as jest.Mock;

jest.mock('../../src/core/config', () => ({
  getConfig: jest.fn(() => ({ experimentId: 'exp-1' })),
}));

const mockedSerialize = serializeSpansToOtlpBytes as jest.MockedFunction<
  typeof serializeSpansToOtlpBytes
>;

function makeRootSpan() {
  return { spanContext: () => ({ traceId: 'abc123', spanId: 'def456' }) } as any;
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
    client = {
      createTrace: jest.fn((info) => Promise.resolve(info)),
      uploadTraceData: jest.fn().mockResolvedValue(undefined),
      exportOtlpSpans: jest.fn().mockResolvedValue(undefined),
    };
    exporter = new MlflowSpanExporter(client as any);
  });

  it('sends OTLP spans to the tracking store after the artifact upload', async () => {
    const bytes = new Uint8Array([1, 2, 3]);
    mockedSerialize.mockReturnValue(bytes);

    exporter.export([makeRootSpan()], () => {});
    await exporter.forceFlush();

    expect(client.exportOtlpSpans).toHaveBeenCalledTimes(1);
    expect(client.exportOtlpSpans).toHaveBeenCalledWith('exp-1', bytes);
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
