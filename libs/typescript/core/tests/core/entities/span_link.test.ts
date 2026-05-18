import { SpanLink } from '../../../src/core/entities/span_link';

const TRACE_ID = 'tr-0123456789abcdef0123456789abcdef';
const SPAN_ID = '0123456789abcdef';

describe('SpanLink', () => {
  describe('constructor', () => {
    it('should create a span link with all parameters', () => {
      const link = new SpanLink({
        traceId: TRACE_ID,
        spanId: SPAN_ID,
        attributes: { type: 'causality', priority: 1 },
      });

      expect(link.traceId).toBe(TRACE_ID);
      expect(link.spanId).toBe(SPAN_ID);
      expect(link.attributes).toEqual({ type: 'causality', priority: 1 });
    });

    it('should create a span link without attributes', () => {
      const link = new SpanLink({
        traceId: TRACE_ID,
        spanId: SPAN_ID,
      });

      expect(link.traceId).toBe(TRACE_ID);
      expect(link.spanId).toBe(SPAN_ID);
      expect(link.attributes).toEqual({});
    });
  });

  describe('toJson', () => {
    it('should serialize with attributes', () => {
      const link = new SpanLink({
        traceId: TRACE_ID,
        spanId: SPAN_ID,
        attributes: { type: 'causality' },
      });

      expect(link.toJson()).toEqual({
        trace_id: TRACE_ID,
        span_id: SPAN_ID,
        attributes: { type: 'causality' },
      });
    });

    it('should serialize without attributes as null', () => {
      const link = new SpanLink({
        traceId: TRACE_ID,
        spanId: SPAN_ID,
      });

      expect(link.toJson()).toEqual({
        trace_id: TRACE_ID,
        span_id: SPAN_ID,
        attributes: null,
      });
    });
  });

  describe('fromJson', () => {
    it('should deserialize with attributes', () => {
      const json = {
        trace_id: TRACE_ID,
        span_id: SPAN_ID,
        attributes: { type: 'test' },
      };

      const link = SpanLink.fromJson(json);

      expect(link.traceId).toBe(TRACE_ID);
      expect(link.spanId).toBe(SPAN_ID);
      expect(link.attributes).toEqual({ type: 'test' });
    });

    it('should deserialize without attributes', () => {
      const json = {
        trace_id: TRACE_ID,
        span_id: SPAN_ID,
      };

      const link = SpanLink.fromJson(json);

      expect(link.traceId).toBe(TRACE_ID);
      expect(link.spanId).toBe(SPAN_ID);
      expect(link.attributes).toEqual({});
    });

    it('should round-trip through toJson and fromJson', () => {
      const original = new SpanLink({
        traceId: TRACE_ID,
        spanId: SPAN_ID,
        attributes: { kind: 'follows_from', weight: 0.5 },
      });

      const reconstructed = SpanLink.fromJson(original.toJson());

      expect(reconstructed.traceId).toBe(original.traceId);
      expect(reconstructed.spanId).toBe(original.spanId);
      expect(reconstructed.attributes).toEqual(original.attributes);
    });
  });
});
