import { MlflowClient } from '../../src/clients';
import { getLastActiveTraceId, flushTraces, init, startSpan, withSpan } from '../../src';
import { SpanType } from '../../src/core/constants';
import { LiveSpan } from '../../src/core/entities/span';
import { SpanStatus, SpanStatusCode } from '../../src/core/entities/span_status';
import { TraceState } from '../../src/core/entities/trace_state';
import { convertHrTimeToMs } from '../../src/core/utils';
import { Trace } from '../../src/core/entities/trace';
import { TEST_TRACKING_URI } from '../jest.setup';

describe('API', () => {
  let client: MlflowClient;
  let experimentId: string;

  const getLastActiveTrace = async (): Promise<Trace> => {
    await flushTraces();
    const traceId = getLastActiveTraceId();
    const trace = await client.getTrace(traceId!);
    return trace;
  };

  beforeAll(async () => {
    client = new MlflowClient({ host: TEST_TRACKING_URI });

    // Create a new experiment
    const experimentName = `test-experiment-${Date.now()}-${Math.random().toString(36).substring(2, 15)}`;
    experimentId = await client.createExperiment(experimentName);
    init({
      trackingUri: TEST_TRACKING_URI,
      experimentId: experimentId
    });
  });

  afterAll(async () => {
    await client.deleteExperiment(experimentId);
  });

  describe('startSpan', () => {
    it('should create a span with span type', async () => {
      const span = startSpan({ name: 'test-span' });
      expect(span).toBeInstanceOf(LiveSpan);

      span.setInputs({ prompt: 'Hello, world!' });
      span.setOutputs({ response: 'Hello, world!' });
      span.setAttributes({ model: 'gpt-4' });
      span.setStatus('OK');
      span.end();

      // Validate traces pushed to the backend
      await flushTraces();
      const trace = await client.getTrace(span.traceId);

      expect(trace.info.traceId).toBe(span.traceId);
      expect(trace.info.state).toBe(TraceState.OK);
      expect(trace.info.requestTime).toBeCloseTo(convertHrTimeToMs(span.startTime));
      expect(trace.info.executionDuration).toBeCloseTo(
        convertHrTimeToMs(span.endTime!) - convertHrTimeToMs(span.startTime)
      );
      return; // TODO: Remove this once we implement trace data export

      expect(trace.data.spans.length).toBe(1);
      const loggedSpan = trace.data.spans[0];
      expect(loggedSpan.traceId).toBe(span.traceId);
      expect(loggedSpan.name).toBe('test-span');
      expect(loggedSpan.spanType).toBe(SpanType.UNKNOWN);
      expect(loggedSpan.inputs).toEqual({ prompt: 'Hello, world!' });
      expect(loggedSpan.outputs).toEqual({ response: 'Hello, world!' });
      expect(loggedSpan.attributes['model']).toBe('gpt-4');
      expect(loggedSpan.startTime).toBe(span.startTime);
      expect(loggedSpan.endTime).toBe(span.endTime);
      expect(loggedSpan.status?.statusCode).toBe(SpanStatusCode.OK);
    });

    it('should create a span with other options', async () => {
      const span = startSpan({
        name: 'test-span',
        span_type: SpanType.LLM,
        inputs: { prompt: 'Hello, world!' },
        attributes: { model: 'gpt-4' },
        startTimeNs: 1e9 // 1 second
      });
      span.end({
        outputs: { response: 'Hello, world!' },
        attributes: { model: 'gpt-4' },
        status: SpanStatusCode.ERROR,
        endTimeNs: 3e9 // 3 seconds
      });

      // Validate traces pushed to the backend
      const trace = await getLastActiveTrace();
      expect(trace.info.traceId).toBe(span.traceId);
      expect(trace.info.state).toBe(TraceState.ERROR);
      expect(trace.info.requestTime).toBeCloseTo(1e3); // requestTime is in milliseconds
      expect(trace.info.executionDuration).toBeCloseTo(2e3); // executionDuration is in milliseconds

      return; // TODO: Remove this once we implement trace data export
      const loggedSpan = trace.data.spans[0];
      expect(loggedSpan.traceId).toBe(span.traceId);
      expect(loggedSpan.name).toBe('test-span');
      expect(loggedSpan.spanType).toBe(SpanType.LLM);
      expect(loggedSpan.inputs).toEqual({ prompt: 'Hello, world!' });
      expect(loggedSpan.outputs).toEqual({ response: 'Hello, world!' });
      expect(loggedSpan.attributes['model']).toBe('gpt-4');
      expect(loggedSpan.startTime[0]).toBe(1);
      expect(loggedSpan.startTime[1]).toBe(0);
      expect(loggedSpan.endTime?.[0]).toBe(3);
      expect(loggedSpan.endTime?.[1]).toBe(0);
      expect(loggedSpan.status?.statusCode).toBe(SpanStatusCode.ERROR);
    });

    it('should create a span with an exception', async () => {
      const span = startSpan({ name: 'test-span-with-exception', span_type: SpanType.LLM });
      expect(span).toBeInstanceOf(LiveSpan);

      span.recordException(new Error('test-error'));
      span.end({ status: new SpanStatus(SpanStatusCode.ERROR, 'test-error') });

      // Validate traces pushed to the backend
      await flushTraces();
      const trace = await client.getTrace(span.traceId);
      expect(trace.info.state).toBe(TraceState.ERROR);
      expect(trace.info.requestTime).toBeCloseTo(convertHrTimeToMs(span.startTime));
      expect(trace.info.executionDuration).toBeCloseTo(
        convertHrTimeToMs(span.endTime!) - convertHrTimeToMs(span.startTime)
      );

      return; // TODO: Remove this once we implement trace data export

      expect(trace.data.spans.length).toBe(1);

      const loggedSpan = trace.data.spans[0];
      expect(loggedSpan.status?.statusCode).toBe(SpanStatusCode.ERROR);
      expect(loggedSpan.status?.description).toBe('test-error');
    });

    it('should create nested spans', async () => {
      const parentSpan = startSpan({ name: 'parent-span' });
      const childSpan1 = startSpan({ name: 'child-span-1', parent: parentSpan });
      childSpan1.end();
      const childSpan2 = startSpan({ name: 'child-span-2', parent: parentSpan });
      const childSpan3 = startSpan({ name: 'child-span-3', parent: childSpan2 });
      childSpan3.end();
      childSpan2.end();

      // This should not be a child of parentSpan
      const independentSpan = startSpan({ name: 'independent-span' });
      independentSpan.end();

      parentSpan.end();

      return; // TODO: Remove this once we implement trace data export

      const trace1 = await client.getTrace(parentSpan.traceId);
      expect(trace1.data.spans.length).toBe(1);
      expect(trace1.data.spans[0].name).toBe('independent-span');

      const trace2 = await client.getTrace(independentSpan.traceId);
      expect(trace2.data.spans.length).toBe(4);
      expect(trace2.data.spans[0].name).toBe('parent-span');
      expect(trace2.data.spans[1].name).toBe('child-span-1');
      expect(trace2.data.spans[1].parentId).toBe(trace2.data.spans[0].spanId);
      expect(trace2.data.spans[2].name).toBe('child-span-2');
      expect(trace2.data.spans[2].parentId).toBe(trace2.data.spans[0].spanId);
      expect(trace2.data.spans[3].name).toBe('child-span-3');
      expect(trace2.data.spans[3].parentId).toBe(trace2.data.spans[2].spanId);
    });
  });

  describe('withSpan', () => {
    describe('inline usage pattern', () => {
      it('should execute synchronous callback and auto-set outputs', async () => {
        const result = withSpan((span) => {
          span.setInputs({ a: 5, b: 3 });
          return 5 + 3; // Auto-set outputs from return value
        });

        expect(result).toBe(8);

        const trace = await getLastActiveTrace();
        expect(trace.info.state).toBe(TraceState.OK);

        return; // TODO: Remove this once we implement trace data export

        const loggedSpan = trace.data.spans[0];
        expect(loggedSpan.name).toBe('span');
        expect(loggedSpan.inputs).toEqual({ a: 5, b: 3 });
        expect(loggedSpan.outputs).toBe(8); // Auto-set from return value
        expect(loggedSpan.status?.statusCode).toBe(SpanStatusCode.OK);
      });

      it('should execute synchronous callback with explicit outputs', async () => {
        const result = withSpan((span) => {
          span.setInputs({ a: 5, b: 3 });
          const sum = 5 + 3;
          span.setOutputs({ result: sum });
          return sum;
        });

        expect(result).toBe(8);

        const trace = await getLastActiveTrace();
        expect(trace.info.state).toBe(TraceState.OK);

        return; // TODO: Remove this once we implement trace data export

        const loggedSpan = trace.data.spans[0];
        expect(loggedSpan.outputs).toEqual({ result: 8 }); // Explicit outputs take precedence
      });

      it('should execute asynchronous callback and auto-set outputs', async () => {
        const result = await withSpan(async (span) => {
          span.setInputs({ delay: 100 });
          await new Promise((resolve) => setTimeout(resolve, 10));
          const value = 'async result';
          return value;
        });

        expect(result).toBe('async result');

        const trace = await getLastActiveTrace();
        expect(trace.info.state).toBe(TraceState.OK);

        return; // TODO: Remove this once we implement trace data export

        const loggedSpan = trace.data.spans[0];
        expect(loggedSpan.inputs).toEqual({ delay: 100 });
        expect(loggedSpan.outputs).toBe('async result');
        expect(loggedSpan.status?.statusCode).toBe(SpanStatusCode.OK);
      });

      it('should handle synchronous errors', async () => {
        expect(() => {
          void withSpan((span) => {
            span.setInputs({ operation: 'divide by zero' });
            throw new Error('Division by zero');
          });
        }).toThrow('Division by zero');

        const trace = await getLastActiveTrace();
        expect(trace.info.state).toBe(TraceState.ERROR);

        return; // TODO: Remove this once we implement trace data export

        const loggedSpan = trace.data.spans[0];
        expect(loggedSpan.status?.statusCode).toBe(SpanStatusCode.ERROR);
        expect(loggedSpan.status?.description).toBe('Division by zero');
      });

      it('should handle asynchronous errors', async () => {
        await expect(
          withSpan(async (span) => {
            span.setInputs({ operation: 'async error' });
            await new Promise((resolve) => setTimeout(resolve, 10));
            throw new Error('Async error');
          })
        ).rejects.toThrow('Async error');

        const trace = await getLastActiveTrace();
        expect(trace.info.state).toBe(TraceState.ERROR);

        return; // TODO: Remove this once we implement trace data export

        const loggedSpan = trace.data.spans[0];
        expect(loggedSpan.status?.statusCode).toBe(SpanStatusCode.ERROR);
        expect(loggedSpan.status?.description).toBe('Async error');
      });
    });

    describe('options-based usage pattern', () => {
      it('should execute with pre-configured options', async () => {
        // Example: Creating a traced addition function
        const add = (a: number, b: number) => {
          const sum = withSpan(
            (span) => {
              const result = a + b;
              span.setOutputs(result);
              return result;
            },
            {
              name: 'add',
              span_type: SpanType.TOOL,
              inputs: { a, b },
              attributes: { operation: 'addition' }
            }
          );
          return sum;
        };

        const result = add(10, 20);
        expect(result).toBe(30);

        const trace = await getLastActiveTrace();
        expect(trace.info.state).toBe(TraceState.OK);

        return; // TODO: Remove this once we implement trace data export

        const loggedSpan = trace.data.spans[0];
        expect(loggedSpan.name).toBe('add');
        expect(loggedSpan.spanType).toBe(SpanType.TOOL);
        expect(loggedSpan.inputs).toEqual({ a: 10, b: 20 });
        expect(loggedSpan.outputs).toEqual(30);
        expect(loggedSpan.attributes['operation']).toBe('addition');
      });

      it('should execute async with pre-configured options', async () => {
        // Example: Creating a traced async function that fetches user data
        const fetchUserData = (userId: string) => {
          return withSpan(
            async (span) => {
              // Simulate API call
              await new Promise((resolve) => setTimeout(resolve, 10));

              // Mock user data
              const userData = { id: userId, name: 'John Doe', email: 'john@example.com' };

              span.setOutputs(userData);
              span.setAttributes({ responseTime: '10ms' });

              return userData;
            },
            {
              name: 'fetchUserData',
              span_type: SpanType.RETRIEVER,
              inputs: { userId },
              attributes: { service: 'user-api', version: '1.0' }
            }
          );
        };

        const result = await fetchUserData('user-123');
        expect(result).toEqual({ id: 'user-123', name: 'John Doe', email: 'john@example.com' });

        const trace = await getLastActiveTrace();
        expect(trace.info.state).toBe(TraceState.OK);

        return; // TODO: Remove this once we implement trace data export

        const loggedSpan = trace.data.spans[0];
        expect(loggedSpan.name).toBe('fetchUserData');
        expect(loggedSpan.spanType).toBe(SpanType.RETRIEVER);
        expect(loggedSpan.inputs).toEqual({ userId: 'user-123' });
        expect(loggedSpan.outputs).toEqual({
          id: 'user-123',
          name: 'John Doe',
          email: 'john@example.com'
        });
        expect(loggedSpan.attributes['service']).toBe('user-api');
        expect(loggedSpan.attributes['version']).toBe('1.0');
        expect(loggedSpan.attributes['responseTime']).toBe('10ms');
      });

      it('should handle nested spans with automatic parent-child relationship', async () => {
        const result = withSpan(
          (parentSpan) => {
            // Nested withSpan call - should automatically be a child of the parent
            const childResult = withSpan(
              (childSpan) => {
                childSpan.setAttributes({ nested: true });
                return 'child result';
              },
              {
                name: 'child',
                inputs: { nested: true }
              }
            );

            parentSpan.setOutputs({ childResult });
            return childResult;
          },
          {
            name: 'parent',
            inputs: { operation: 'parent operation' }
          }
        );

        expect(result).toBe('child result');

        const trace = await getLastActiveTrace();
        expect(trace.info.state).toBe(TraceState.OK);

        return; // TODO: Remove this once we implement trace data export
        expect(trace.data.spans.length).toBe(2);

        const parentSpan = trace.data.spans.find((s) => s.name === 'parent');
        expect(parentSpan?.inputs).toEqual({ operation: 'parent operation' });
        expect(parentSpan?.outputs).toEqual({ childResult: 'child result' });

        const childSpan = trace.data.spans.find((s) => s.name === 'child');
        expect(childSpan?.parentId).toBe(parentSpan?.spanId);
        expect(childSpan?.inputs).toEqual({ nested: true });
        expect(childSpan?.outputs).toEqual('child result');
        expect(childSpan?.attributes['nested']).toBe(true);
      });
    });

    describe('edge cases', () => {
      it('should handle null return values', async () => {
        const result = withSpan((span) => {
          span.setInputs({ test: true });
          return null;
        });

        expect(result).toBeNull();

        const trace = await getLastActiveTrace();
        expect(trace.info.state).toBe(TraceState.OK);

        return; // TODO: Remove this once we implement trace data export

        const loggedSpan = trace.data.spans[0];
        expect(loggedSpan.outputs).toBe(null); // Should auto-set null
      });

      it('should handle complex return objects', async () => {
        const complexObject = {
          data: [1, 2, 3],
          metadata: { type: 'array', length: 3 }
        };

        const result = withSpan((span) => {
          span.setInputs({ operation: 'create complex object' });
          return complexObject;
        });

        expect(result).toEqual(complexObject);

        const trace = await getLastActiveTrace();
        expect(trace.info.state).toBe(TraceState.OK);

        return; // TODO: Remove this once we implement trace data export

        const loggedSpan = trace.data.spans[0];
        expect(loggedSpan.outputs).toEqual(complexObject);
      });

      it('should handle multiple async tasks with proper parent-child relationships', async () => {
        // Test complex async scenario with multiple traces and nested spans
        const results = await Promise.all([
          // First trace with nested async operations
          withSpan(
            async (parentSpan) => {
              // Simulate some async work
              await new Promise((resolve) => setTimeout(resolve, 5));

              // Create multiple child spans in parallel
              const childResults = await Promise.all([
                withSpan(
                  async (childSpan) => {
                    await new Promise((resolve) => setTimeout(resolve, 3));
                    childSpan.setOutputs({ processed: true });
                    return 'child1-result';
                  },
                  { name: 'trace1-child1', inputs: { childId: 1 } }
                ),
                withSpan(
                  async (childSpan) => {
                    await new Promise((resolve) => setTimeout(resolve, 2));
                    // Nested grandchild
                    const grandchildResult = await withSpan(
                      async () => {
                        await new Promise((resolve) => setTimeout(resolve, 1));
                        return 'grandchild-result';
                      },
                      { name: 'trace1-grandchild', inputs: { grandchildId: 1 } }
                    );
                    childSpan.setOutputs({ grandchildResult });
                    return 'child2-result';
                  },
                  { name: 'trace1-child2', inputs: { childId: 2 } }
                )
              ]);

              parentSpan.setOutputs({ childResults });
              return childResults;
            },
            { name: 'trace1-parent', inputs: { traceId: 1 } }
          ),

          // Second independent trace running concurrently
          withSpan(
            async (parentSpan) => {
              await new Promise((resolve) => setTimeout(resolve, 4));

              const result = await withSpan(
                async () => {
                  await new Promise((resolve) => setTimeout(resolve, 2));
                  return 'trace2-child-result';
                },
                { name: 'trace2-child', inputs: { childId: 1 } }
              );

              parentSpan.setOutputs({ childResult: result });
              return result;
            },
            { name: 'trace2-parent', inputs: { traceId: 2 } }
          ),

          // Third independent trace with synchronous operation
          withSpan(
            (span) => {
              span.setOutputs({ immediate: true });
              return 'trace3-result';
            },
            { name: 'trace3-simple', inputs: { traceId: 3 } }
          )
        ]);

        // Verify results
        expect(results[0]).toEqual(['child1-result', 'child2-result']);
        expect(results[1]).toBe('trace2-child-result');
        expect(results[2]).toBe('trace3-result');

        // TODO: Uncomment this once we implement trace data export

        // // Verify first trace (most complex with grandchild)
        // const trace1 = traces.find((t) => t.data.spans.some((s) => s.name === 'trace1-parent'));
        // expect(trace1).toBeDefined();
        // expect(trace1!.data.spans.length).toBe(4); // parent + 2 children + 1 grandchild

        // const trace1Parent = trace1!.data.spans.find((s) => s.name === 'trace1-parent');
        // const trace1Child1 = trace1!.data.spans.find((s) => s.name === 'trace1-child1');
        // const trace1Child2 = trace1!.data.spans.find((s) => s.name === 'trace1-child2');
        // const trace1Grandchild = trace1!.data.spans.find((s) => s.name === 'trace1-grandchild');

        // expect(trace1Parent).toBeDefined();
        // expect(trace1Child1).toBeDefined();
        // expect(trace1Child2).toBeDefined();
        // expect(trace1Grandchild).toBeDefined();

        // // Verify parent-child relationships in trace1
        // expect(trace1Child1!.parentId).toBe(trace1Parent!.spanId);
        // expect(trace1Child2!.parentId).toBe(trace1Parent!.spanId);
        // expect(trace1Grandchild!.parentId).toBe(trace1Child2!.spanId);

        // // Verify outputs
        // expect(trace1Parent!.outputs).toEqual({ childResults: ['child1-result', 'child2-result'] });
        // expect(trace1Child1!.outputs).toEqual({ processed: true });
        // expect(trace1Child2!.outputs).toEqual({ grandchildResult: 'grandchild-result' });
        // expect(trace1Grandchild!.outputs).toBe('grandchild-result');

        // // Verify second trace
        // const trace2 = traces.find((t) => t.data.spans.some((s) => s.name === 'trace2-parent'));
        // expect(trace2).toBeDefined();
        // expect(trace2!.data.spans.length).toBe(2); // parent + 1 child

        // const trace2Parent = trace2!.data.spans.find((s) => s.name === 'trace2-parent');
        // const trace2Child = trace2!.data.spans.find((s) => s.name === 'trace2-child');

        // expect(trace2Parent).toBeDefined();
        // expect(trace2Child).toBeDefined();
        // expect(trace2Child!.parentId).toBe(trace2Parent!.spanId);
        // expect(trace2Parent!.outputs).toEqual({ childResult: 'trace2-child-result' });
        // expect(trace2Child!.outputs).toBe('trace2-child-result');

        // // Verify third trace (simple synchronous)
        // const trace3 = traces.find((t) => t.data.spans.some((s) => s.name === 'trace3-simple'));
        // expect(trace3).toBeDefined();
        // expect(trace3!.data.spans.length).toBe(1);

        // const trace3Span = trace3!.data.spans[0];
        // expect(trace3Span.name).toBe('trace3-simple');
        // expect(trace3Span.parentId).toBeNull();
        // expect(trace3Span.outputs).toEqual({ immediate: true });
      });
    });
  });
});
