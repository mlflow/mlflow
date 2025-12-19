import { TraceState, fromOtelStatus } from '../../../src/core/entities/trace_state';
import { SpanStatusCode } from '@opentelemetry/api';

describe('TraceState', () => {
  describe('enum values', () => {
    it('should have the correct enum values', () => {
      expect(TraceState.STATE_UNSPECIFIED).toBe('STATE_UNSPECIFIED');
      expect(TraceState.OK).toBe('OK');
      expect(TraceState.ERROR).toBe('ERROR');
      expect(TraceState.IN_PROGRESS).toBe('IN_PROGRESS');
    });
  });

  describe('fromOtelStatus', () => {
    it('should convert OpenTelemetry OK status to TraceState.OK', () => {
      const result = fromOtelStatus(SpanStatusCode.OK);
      expect(result).toBe(TraceState.OK);
    });

    it('should convert OpenTelemetry ERROR status to TraceState.ERROR', () => {
      const result = fromOtelStatus(SpanStatusCode.ERROR);
      expect(result).toBe(TraceState.ERROR);
    });

    it('should convert OpenTelemetry UNSET status to TraceState.STATE_UNSPECIFIED', () => {
      const result = fromOtelStatus(SpanStatusCode.UNSET);
      expect(result).toBe(TraceState.STATE_UNSPECIFIED);
    });
  });
});
