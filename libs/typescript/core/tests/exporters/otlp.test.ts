import { ROOT_CONTEXT, TraceFlags, trace as otelTrace } from '@opentelemetry/api';
import { ProtobufTraceSerializer } from '@opentelemetry/otlp-transformer';
import {
  BasicTracerProvider,
  ReadableSpan as OTelReadableSpan,
} from '@opentelemetry/sdk-trace-base';
import { MlflowHttpError } from '../../src/clients/utils';
import {
  isSpanIngestionUnsupportedError,
  sendOtlpSpansIfSupported,
  serializeSpansToOtlpBase64,
  serializeSpansToOtlpBytes,
  supportsOtlpSpanIngestion,
} from '../../src/exporters/otlp';
import { MAX_OTLP_REQUEST_BYTES } from '../../src/core/constants';

function makeSerializableSpan(): OTelReadableSpan {
  return {
    name: 'span',
    resource: {},
    instrumentationScope: {},
    attributes: {},
  } as unknown as OTelReadableSpan;
}

describe('exporters/otlp', () => {
  describe('supportsOtlpSpanIngestion', () => {
    it.each(['databricks', 'databricks://profile'])('rejects %s', (trackingUri) => {
      expect(supportsOtlpSpanIngestion(trackingUri)).toBe(false);
    });

    it.each(['http://localhost:5000', 'https://mlflow.example.com'])(
      'accepts %s',
      (trackingUri) => {
        expect(supportsOtlpSpanIngestion(trackingUri)).toBe(true);
      },
    );
  });

  describe('isSpanIngestionUnsupportedError', () => {
    const mlflowErrorBody = JSON.stringify({
      error_code: 'RESOURCE_DOES_NOT_EXIST',
      message: 'no such experiment',
    });

    it.each([
      ['501 from a store that cannot log spans', new MlflowHttpError(501, 'Not Implemented', '')],
      ['405', new MlflowHttpError(405, 'Method Not Allowed', '')],
      ['404 from a server without the route', new MlflowHttpError(404, 'Not Found', '')],
    ])('treats %s as unsupported', (_name, err) => {
      expect(isSpanIngestionUnsupportedError(err)).toBe(true);
    });

    it.each([
      ['404 carrying an MLflow error_code', new MlflowHttpError(404, 'Not Found', mlflowErrorBody)],
      ['500', new MlflowHttpError(500, 'Internal Server Error', '')],
      ['a network error', new Error('ECONNRESET')],
    ])('does not treat %s as unsupported', (_name, err) => {
      expect(isSpanIngestionUnsupportedError(err)).toBe(false);
    });
  });

  describe('serialization', () => {
    afterEach(() => {
      jest.restoreAllMocks();
    });

    it('decodes JSON-encoded attributes and turns a remote parent into a link', () => {
      const serializeSpy = jest.spyOn(ProtobufTraceSerializer, 'serializeRequest');
      const remoteParent = {
        traceId: '0af7651916cd43dd8448eb211c80319c',
        spanId: 'b7ad6b7169203331',
        traceFlags: TraceFlags.SAMPLED,
        isRemote: true,
      };
      const span = new BasicTracerProvider()
        .getTracer('otlp-test')
        .startSpan('otlp-test-span', {}, otelTrace.setSpanContext(ROOT_CONTEXT, remoteParent));
      span.setAttribute('mlflow.spanType', JSON.stringify('LLM'));
      span.setAttribute('not.json', '"unterminated');
      span.setAttribute('plain', 'text');
      span.setAttribute('count', 3);
      span.end();
      const readable = span as unknown as OTelReadableSpan;

      const bytes = serializeSpansToOtlpBytes([readable], 'http://localhost:5000');

      const [serialized] = serializeSpy.mock.calls[0][0];
      expect(serialized.attributes).toEqual({
        'mlflow.spanType': 'LLM',
        'not.json': '"unterminated',
        plain: 'text',
        count: 3,
      });
      expect(serialized.parentSpanContext).toBeUndefined();
      expect(serialized.links).toEqual([{ context: remoteParent }]);
      // The live span is left untouched.
      expect(readable.attributes['mlflow.spanType']).toBe('"LLM"');
      expect(readable.parentSpanContext?.spanId).toBe(remoteParent.spanId);
      expect(Buffer.from(bytes as Uint8Array).includes(Buffer.from('otlp-test-span'))).toBe(true);
    });

    it('does not serialize spans for Databricks tracking URIs', () => {
      const serializeSpy = jest.spyOn(ProtobufTraceSerializer, 'serializeRequest');

      expect(serializeSpansToOtlpBytes([makeSerializableSpan()], 'databricks')).toBeUndefined();
      expect(
        serializeSpansToOtlpBase64([makeSerializableSpan()], 'databricks://profile'),
      ).toBeUndefined();
      expect(serializeSpy).not.toHaveBeenCalled();
    });
  });

  describe('sendOtlpSpansIfSupported', () => {
    it('skips Databricks tracking URIs', async () => {
      const send = jest.fn().mockResolvedValue(undefined);

      await expect(
        sendOtlpSpansIfSupported('databricks', 'exp-1', new Uint8Array([1]), send, 'trace tr-1'),
      ).resolves.toBe(false);

      expect(send).not.toHaveBeenCalled();
    });

    it('skips when there is no payload', async () => {
      const send = jest.fn().mockResolvedValue(undefined);

      await expect(
        sendOtlpSpansIfSupported('http://localhost:5000', 'exp-1', undefined, send, 'trace tr-1'),
      ).resolves.toBe(false);

      expect(send).not.toHaveBeenCalled();
    });

    it('skips payloads that exceed the request budget', async () => {
      const send = jest.fn().mockResolvedValue(undefined);
      const warn = jest.fn();
      const bytes = { length: MAX_OTLP_REQUEST_BYTES + 1 } as Uint8Array;

      await expect(
        sendOtlpSpansIfSupported('http://localhost:5000', 'exp-1', bytes, send, 'trace tr-1', warn),
      ).resolves.toBe(false);

      expect(send).not.toHaveBeenCalled();
      expect(warn).toHaveBeenCalledWith(expect.stringContaining('exceeding the'));
    });

    it('forwards the payload for supported tracking URIs', async () => {
      const send = jest.fn().mockResolvedValue(undefined);
      const bytes = new Uint8Array([1, 2, 3]);

      await expect(
        sendOtlpSpansIfSupported('http://localhost:5000', 'exp-1', bytes, send, 'trace tr-1'),
      ).resolves.toBe(true);

      expect(send).toHaveBeenCalledWith('exp-1', bytes);
    });
  });
});
