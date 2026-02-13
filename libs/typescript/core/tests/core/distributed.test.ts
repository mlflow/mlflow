import { context, trace as otelTrace } from '@opentelemetry/api';
import {
  getTracingContextHeadersForHttpRequest,
  withTracingContextFromHeaders,
  withTracingContextFromHeadersAsync,
} from '../../src/core/distributed';
import { InMemoryTraceManager } from '../../src/core/trace_manager';
import { TraceInfo } from '../../src/core/entities/trace_info';
import { createTraceLocationFromExperimentId } from '../../src/core/entities/trace_location';
import { TraceState } from '../../src/core/entities/trace_state';
import { TRACE_ID_PREFIX } from '../../src/core/constants';
import * as mlflow from '../../src';
import { TEST_TRACKING_URI } from '../helper';

describe('Distributed Tracing', () => {
  beforeAll(() => {
    mlflow.init({
      trackingUri: TEST_TRACKING_URI,
      experimentId: '0',
    });
  });

  beforeEach(() => {
    InMemoryTraceManager.reset();
  });

  afterEach(() => {
    InMemoryTraceManager.reset();
  });

  describe('getTracingContextHeadersForHttpRequest', () => {
    it('should return traceparent header when inside active span', () => {
      mlflow.withSpan(
        (span) => {
          const headers = getTracingContextHeadersForHttpRequest();

          expect(headers).toHaveProperty('traceparent');
          expect(headers.traceparent).toMatch(/^00-[0-9a-f]{32}-[0-9a-f]{16}-01$/);

          // Verify trace and span IDs match
          const parts = headers.traceparent.split('-');
          const headerTraceId = parts[1];
          const headerSpanId = parts[2];

          // The OTel trace ID is embedded in the MLflow trace ID after the prefix
          const otelTraceId = span.traceId.replace(TRACE_ID_PREFIX, '');
          expect(headerTraceId).toBe(otelTraceId);

          // Span ID should match
          expect(headerSpanId).toBe(span.spanId);
        },
        { name: 'test-span' },
      );
    });

    it('should return empty object when no active span', () => {
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();

      const headers = getTracingContextHeadersForHttpRequest();

      expect(headers).toEqual({});
      expect(consoleSpy).toHaveBeenCalledWith(
        'No active span found for fetching the trace context from. Returning an empty header.',
      );

      consoleSpy.mockRestore();
    });
  });

  describe('withTracingContextFromHeaders', () => {
    it('should throw error when headers missing traceparent', () => {
      expect(() => {
        withTracingContextFromHeaders({}, () => {});
      }).toThrow("do not contain the required key 'traceparent'");
    });

    it('should handle case-insensitive header keys', () => {
      mlflow.withSpan(
        () => {
          const headers = getTracingContextHeadersForHttpRequest();

          // Simulate uppercase header key (like some frameworks do)
          const uppercaseHeaders = {
            Traceparent: headers.traceparent,
          };

          expect(() => {
            withTracingContextFromHeaders(uppercaseHeaders, () => {
              // Should not throw
            });
          }).not.toThrow();
        },
        { name: 'client-span' },
      );
    });

    it('should handle array header values', () => {
      mlflow.withSpan(
        () => {
          const headers = getTracingContextHeadersForHttpRequest();

          // Simulate array header values (like Express sometimes provides)
          const arrayHeaders: Record<string, string | string[]> = {
            traceparent: [headers.traceparent],
          };

          expect(() => {
            withTracingContextFromHeaders(arrayHeaders, () => {
              // Should not throw
            });
          }).not.toThrow();
        },
        { name: 'client-span' },
      );
    });

    it('should extract trace context and allow child spans to be created', () => {
      // Create a traceparent header manually
      const traceId = '0af7651916cd43dd8448eb211c80319c';
      const parentSpanId = 'b7ad6b7169203331';
      const traceparent = `00-${traceId}-${parentSpanId}-01`;

      let extractedTraceId: string | undefined;

      withTracingContextFromHeaders({ traceparent }, () => {
        // Within this context, we should be able to get the span context
        const extractedSpanContext = otelTrace.getSpanContext(context.active());
        extractedTraceId = extractedSpanContext?.traceId;
      });

      expect(extractedTraceId).toBe(traceId);
    });

    it('should register remote trace in InMemoryTraceManager', () => {
      const traceId = '0af7651916cd43dd8448eb211c80319c';
      const parentSpanId = 'b7ad6b7169203331';
      const traceparent = `00-${traceId}-${parentSpanId}-01`;

      const traceManager = InMemoryTraceManager.getInstance();

      withTracingContextFromHeaders({ traceparent }, () => {
        // Inside the context, the trace should be registered
        const mlflowTraceId = traceManager.getMlflowTraceIdFromOtelId(traceId);
        expect(mlflowTraceId).toBe(TRACE_ID_PREFIX + traceId);
      });

      // After exiting, the trace should be cleaned up
      const mlflowTraceId = traceManager.getMlflowTraceIdFromOtelId(traceId);
      expect(mlflowTraceId).toBeNull();
    });

    it('should clean up remote trace even if callback throws', () => {
      const traceId = '0af7651916cd43dd8448eb211c80319c';
      const parentSpanId = 'b7ad6b7169203331';
      const traceparent = `00-${traceId}-${parentSpanId}-01`;

      const traceManager = InMemoryTraceManager.getInstance();

      expect(() => {
        withTracingContextFromHeaders({ traceparent }, () => {
          throw new Error('Test error');
        });
      }).toThrow('Test error');

      // After exiting (even with error), the trace should be cleaned up
      const mlflowTraceId = traceManager.getMlflowTraceIdFromOtelId(traceId);
      expect(mlflowTraceId).toBeNull();
    });
  });

  describe('withTracingContextFromHeadersAsync', () => {
    it('should work with async callbacks', async () => {
      const traceId = '0af7651916cd43dd8448eb211c80319c';
      const parentSpanId = 'b7ad6b7169203331';
      const traceparent = `00-${traceId}-${parentSpanId}-01`;

      let callbackExecuted = false;

      await withTracingContextFromHeadersAsync({ traceparent }, async () => {
        await new Promise((resolve) => setTimeout(resolve, 10));
        callbackExecuted = true;
      });

      expect(callbackExecuted).toBe(true);
    });

    it('should clean up remote trace after async callback completes', async () => {
      const traceId = '0af7651916cd43dd8448eb211c80319c';
      const parentSpanId = 'b7ad6b7169203331';
      const traceparent = `00-${traceId}-${parentSpanId}-01`;

      const traceManager = InMemoryTraceManager.getInstance();

      await withTracingContextFromHeadersAsync({ traceparent }, async () => {
        // Inside the context, the trace should be registered
        const mlflowTraceId = traceManager.getMlflowTraceIdFromOtelId(traceId);
        expect(mlflowTraceId).toBe(TRACE_ID_PREFIX + traceId);
        await new Promise((resolve) => setTimeout(resolve, 10));
      });

      // After exiting, the trace should be cleaned up
      const mlflowTraceId = traceManager.getMlflowTraceIdFromOtelId(traceId);
      expect(mlflowTraceId).toBeNull();
    });

    it('should clean up remote trace even if async callback rejects', async () => {
      const traceId = '0af7651916cd43dd8448eb211c80319c';
      const parentSpanId = 'b7ad6b7169203331';
      const traceparent = `00-${traceId}-${parentSpanId}-01`;

      const traceManager = InMemoryTraceManager.getInstance();

      await expect(
        withTracingContextFromHeadersAsync({ traceparent }, async () => {
          await new Promise((resolve) => setTimeout(resolve, 10));
          throw new Error('Async test error');
        }),
      ).rejects.toThrow('Async test error');

      // After rejection, the trace should be cleaned up
      const mlflowTraceId = traceManager.getMlflowTraceIdFromOtelId(traceId);
      expect(mlflowTraceId).toBeNull();
    });
  });

  describe('InMemoryTraceManager remote trace handling', () => {
    it('should not export remote traces when popTrace is called', () => {
      const traceManager = InMemoryTraceManager.getInstance();

      const otelTraceId = '0af7651916cd43dd8448eb211c80319c';
      const mlflowTraceId = TRACE_ID_PREFIX + otelTraceId;

      const traceInfo = new TraceInfo({
        traceId: mlflowTraceId,
        traceLocation: createTraceLocationFromExperimentId('1'),
        requestTime: Date.now(),
        state: TraceState.IN_PROGRESS,
        traceMetadata: {},
        tags: {},
      });

      // Register as a remote trace
      traceManager.registerTrace(otelTraceId, traceInfo, true /* isRemoteTrace */);

      // Verify it's registered
      expect(traceManager.getMlflowTraceIdFromOtelId(otelTraceId)).toBe(mlflowTraceId);

      // Pop the trace - should return null for remote traces
      const poppedTrace = traceManager.popTrace(otelTraceId);
      expect(poppedTrace).toBeNull();

      // Verify it's cleaned up
      expect(traceManager.getMlflowTraceIdFromOtelId(otelTraceId)).toBeNull();
    });

    it('should export non-remote traces normally', () => {
      const traceManager = InMemoryTraceManager.getInstance();

      const otelTraceId = '0af7651916cd43dd8448eb211c80319c';
      const mlflowTraceId = TRACE_ID_PREFIX + otelTraceId;

      const traceInfo = new TraceInfo({
        traceId: mlflowTraceId,
        traceLocation: createTraceLocationFromExperimentId('1'),
        requestTime: Date.now(),
        state: TraceState.IN_PROGRESS,
        traceMetadata: {},
        tags: {},
      });

      // Register as a normal (non-remote) trace
      traceManager.registerTrace(otelTraceId, traceInfo, false /* isRemoteTrace */);

      // Pop the trace - should return the trace for normal traces
      const poppedTrace = traceManager.popTrace(otelTraceId);
      expect(poppedTrace).not.toBeNull();
      expect(poppedTrace?.info.traceId).toBe(mlflowTraceId);
    });
  });
});
