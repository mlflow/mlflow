import { ProtobufTraceSerializer } from '@opentelemetry/otlp-transformer';
import { ReadableSpan as OTelReadableSpan } from '@opentelemetry/sdk-trace-base';
import {
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

  describe('serialization', () => {
    afterEach(() => {
      jest.restoreAllMocks();
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
