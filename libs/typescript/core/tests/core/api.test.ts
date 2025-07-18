import { MlflowClient } from '../../src/clients';
import * as mlflow from '../../src';
import { SpanType } from '../../src/core/constants';
import { LiveSpan } from '../../src/core/entities/span';
import { SpanStatus, SpanStatusCode } from '../../src/core/entities/span_status';
import { TraceState } from '../../src/core/entities/trace_state';
import { convertHrTimeToMs } from '../../src/core/utils';
import { Trace } from '../../src/core/entities/trace';
import { TEST_TRACKING_URI } from '../helper';

describe('API', () => {
  let client: MlflowClient;
  let experimentId: string;

  const getLastActiveTrace = async (): Promise<Trace> => {
    await mlflow.flushTraces();
    const traceId = mlflow.getLastActiveTraceId();
    const trace = await client.getTrace(traceId!);
    return trace;
  };

  beforeAll(async () => {
    client = new MlflowClient({ trackingUri: TEST_TRACKING_URI, host: TEST_TRACKING_URI });

    // Create a new experiment
    const experimentName = `test-experiment-${Date.now()}-${Math.random().toString(36).substring(2, 15)}`;
    experimentId = await client.createExperiment(experimentName);
    mlflow.init({
      trackingUri: TEST_TRACKING_URI,
      experimentId: experimentId
    });
  });

  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterAll(async () => {
    await client.deleteExperiment(experimentId);
  });

  describe('startSpan', () => {
    it('should create a span with span type', async () => {
      const span = mlflow.startSpan({ name: 'test-span' });
      expect(span).toBeInstanceOf(LiveSpan);

      span.setInputs({ prompt: 'Hello, world!' });
      span.setOutputs({ response: 'Hello, world!' });
      span.setAttributes({ model: 'gpt-4' });
      span.setStatus('OK');
      span.end();

      // Validate traces pushed to the backend
      await mlflow.flushTraces();
      const trace = await client.getTrace(span.traceId);

      expect(trace.info.traceId).toBe(span.traceId);
      expect(trace.info.state).toBe(TraceState.OK);
      expect(trace.info.requestTime).toBeCloseTo(convertHrTimeToMs(span.startTime));
      expect(trace.info.executionDuration).toBeCloseTo(
        convertHrTimeToMs(span.endTime!) - convertHrTimeToMs(span.startTime)
      );

      expect(trace.data.spans.length).toBe(1);
      const loggedSpan = trace.data.spans[0];
      expect(loggedSpan.traceId).toBe(span.traceId);
      expect(loggedSpan.name).toBe('test-span');
      expect(loggedSpan.spanType).toBe(SpanType.UNKNOWN);
      expect(loggedSpan.inputs).toEqual({ prompt: 'Hello, world!' });
      expect(loggedSpan.outputs).toEqual({ response: 'Hello, world!' });
      expect(loggedSpan.attributes['model']).toBe('gpt-4');
      expect(loggedSpan.startTime).toEqual(span.startTime);
      expect(loggedSpan.endTime).toEqual(span.endTime);
      expect(loggedSpan.status?.statusCode).toBe(SpanStatusCode.OK);
    });

    it('should create a span with other options', async () => {
      const span = mlflow.startSpan({
        name: 'test-span',
        spanType: SpanType.LLM,
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
      const span = mlflow.startSpan({ name: 'test-span-with-exception', spanType: SpanType.LLM });
      expect(span).toBeInstanceOf(LiveSpan);

      span.recordException(new Error('test-error'));
      span.end({ status: new SpanStatus(SpanStatusCode.ERROR, 'test-error') });

      // Validate traces pushed to the backend
      await mlflow.flushTraces();
      const trace = await client.getTrace(span.traceId);
      expect(trace.info.state).toBe(TraceState.ERROR);
      expect(trace.info.requestTime).toBeCloseTo(convertHrTimeToMs(span.startTime));
      expect(trace.info.executionDuration).toBeCloseTo(
        convertHrTimeToMs(span.endTime!) - convertHrTimeToMs(span.startTime)
      );
      expect(trace.data.spans.length).toBe(1);

      const loggedSpan = trace.data.spans[0];
      expect(loggedSpan.status?.statusCode).toBe(SpanStatusCode.ERROR);
      expect(loggedSpan.status?.description).toBe('test-error');
    });

    it('should create nested spans', async () => {
      const parentSpan = mlflow.startSpan({ name: 'parent-span' });
      const childSpan1 = mlflow.startSpan({ name: 'child-span-1', parent: parentSpan });
      childSpan1.end();
      const childSpan2 = mlflow.startSpan({ name: 'child-span-2', parent: parentSpan });
      const childSpan3 = mlflow.startSpan({ name: 'child-span-3', parent: childSpan2 });
      childSpan3.end();
      childSpan2.end();

      // This should not be a child of parentSpan
      const independentSpan = mlflow.startSpan({ name: 'independent-span' });
      independentSpan.end();

      parentSpan.end();

      await mlflow.flushTraces();
      const trace1 = await client.getTrace(independentSpan.traceId);
      expect(trace1.data.spans.length).toBe(1);
      expect(trace1.data.spans[0].name).toBe('independent-span');

      const trace2 = await client.getTrace(parentSpan.traceId);
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
        const result = mlflow.withSpan((span) => {
          span.setInputs({ a: 5, b: 3 });
          return 5 + 3; // Auto-set outputs from return value
        });

        expect(result).toBe(8);

        const trace = await getLastActiveTrace();
        expect(trace.info.state).toBe(TraceState.OK);

        const loggedSpan = trace.data.spans[0];
        expect(loggedSpan.name).toBe('span');
        expect(loggedSpan.inputs).toEqual({ a: 5, b: 3 });
        expect(loggedSpan.outputs).toBe(8); // Auto-set from return value
        expect(loggedSpan.status?.statusCode).toBe(SpanStatusCode.OK);
      });

      it('should execute synchronous callback with explicit outputs', async () => {
        const result = mlflow.withSpan((span) => {
          span.setInputs({ a: 5, b: 3 });
          const sum = 5 + 3;
          span.setOutputs({ result: sum });
          return sum;
        });

        expect(result).toBe(8);

        const trace = await getLastActiveTrace();
        expect(trace.info.state).toBe(TraceState.OK);

        const loggedSpan = trace.data.spans[0];
        expect(loggedSpan.outputs).toEqual({ result: 8 }); // Explicit outputs take precedence
      });

      it('should execute asynchronous callback and auto-set outputs', async () => {
        const result = await mlflow.withSpan(async (span) => {
          span.setInputs({ delay: 100 });
          await new Promise((resolve) => setTimeout(resolve, 10));
          const value = 'async result';
          return value;
        });

        expect(result).toBe('async result');

        const trace = await getLastActiveTrace();
        expect(trace.info.state).toBe(TraceState.OK);

        const loggedSpan = trace.data.spans[0];
        expect(loggedSpan.inputs).toEqual({ delay: 100 });
        expect(loggedSpan.outputs).toBe('async result');
        expect(loggedSpan.status?.statusCode).toBe(SpanStatusCode.OK);
      });

      it('should handle synchronous errors', async () => {
        expect(() => {
          void mlflow.withSpan((span) => {
            span.setInputs({ operation: 'divide by zero' });
            throw new Error('Division by zero');
          });
        }).toThrow('Division by zero');

        const trace = await getLastActiveTrace();
        expect(trace.info.state).toBe(TraceState.ERROR);

        const loggedSpan = trace.data.spans[0];
        expect(loggedSpan.status?.statusCode).toBe(SpanStatusCode.ERROR);
        expect(loggedSpan.status?.description).toBe('Division by zero');
      });

      it('should handle asynchronous errors', async () => {
        await expect(
          mlflow.withSpan(async (span) => {
            span.setInputs({ operation: 'async error' });
            await new Promise((resolve) => setTimeout(resolve, 10));
            throw new Error('Async error');
          })
        ).rejects.toThrow('Async error');

        const trace = await getLastActiveTrace();
        expect(trace.info.state).toBe(TraceState.ERROR);

        const loggedSpan = trace.data.spans[0];
        expect(loggedSpan.status?.statusCode).toBe(SpanStatusCode.ERROR);
        expect(loggedSpan.status?.description).toBe('Async error');
      });
    });

    describe('options-based usage pattern', () => {
      it('should execute with pre-configured options', async () => {
        // Example: Creating a traced addition function
        const add = (a: number, b: number) => {
          const sum = mlflow.withSpan(
            (span) => {
              const result = a + b;
              span.setOutputs(result);
              return result;
            },
            {
              name: 'add',
              spanType: SpanType.TOOL,
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
          return mlflow.withSpan(
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
              spanType: SpanType.RETRIEVER,
              inputs: { userId },
              attributes: { service: 'user-api', version: 1 }
            }
          );
        };

        const result = await fetchUserData('user-123');
        expect(result).toEqual({ id: 'user-123', name: 'John Doe', email: 'john@example.com' });

        const trace = await getLastActiveTrace();
        expect(trace.info.state).toBe(TraceState.OK);

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
        expect(loggedSpan.attributes['version']).toBe(1);
        expect(loggedSpan.attributes['responseTime']).toBe('10ms');
      });

      it('should handle nested spans with automatic parent-child relationship', async () => {
        const result = mlflow.withSpan(
          (parentSpan) => {
            // Nested withSpan call - should automatically be a child of the parent
            const childResult = mlflow.withSpan(
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
        const result = mlflow.withSpan((span) => {
          span.setInputs({ test: true });
          return null;
        });

        expect(result).toBeNull();

        const trace = await getLastActiveTrace();
        expect(trace.info.state).toBe(TraceState.OK);

        const loggedSpan = trace.data.spans[0];
        expect(loggedSpan.outputs).toBe(null); // Should auto-set null
      });

      it('should handle complex return objects', async () => {
        const complexObject = {
          data: [1, 2, 3],
          metadata: { type: 'array', length: 3 }
        };

        const result = mlflow.withSpan((span) => {
          span.setInputs({ operation: 'create complex object' });
          return complexObject;
        });

        expect(result).toEqual(complexObject);

        const trace = await getLastActiveTrace();
        expect(trace.info.state).toBe(TraceState.OK);

        const loggedSpan = trace.data.spans[0];
        expect(loggedSpan.outputs).toEqual(complexObject);
      });

      it('should handle multiple async tasks with proper parent-child relationships', async () => {
        // Test complex async scenario with multiple traces and nested spans

        const traceIds: Record<string, string> = {};
        const results = await Promise.all([
          // First trace with nested async operations
          mlflow.withSpan(
            async (parentSpan) => {
              // Simulate some async work
              await new Promise((resolve) => setTimeout(resolve, 5));
              traceIds['trace1-parent'] = parentSpan.traceId;

              // Create multiple child spans in parallel
              const childResults = await Promise.all([
                mlflow.withSpan(
                  async (childSpan) => {
                    await new Promise((resolve) => setTimeout(resolve, 3));
                    childSpan.setOutputs({ processed: true });
                    return 'child1-result';
                  },
                  { name: 'trace1-child1', inputs: { childId: 1 } }
                ),
                mlflow.withSpan(
                  async (childSpan) => {
                    await new Promise((resolve) => setTimeout(resolve, 2));
                    // Nested grandchild
                    const grandchildResult = await mlflow.withSpan(
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
          mlflow.withSpan(
            async (parentSpan) => {
              await new Promise((resolve) => setTimeout(resolve, 4));
              traceIds['trace2-parent'] = parentSpan.traceId;
              const result = await mlflow.withSpan(
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
          mlflow.withSpan(
            (span) => {
              traceIds['trace3-simple'] = span.traceId;
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

        await mlflow.flushTraces();

        // Verify first trace (most complex with grandchild)
        const trace1 = await client.getTrace(traceIds['trace1-parent']);
        expect(trace1).toBeDefined();
        expect(trace1.data.spans.length).toBe(4); // parent + 2 children + 1 grandchild

        const trace1Parent = trace1.data.spans.find((s) => s.name === 'trace1-parent');
        const trace1Child1 = trace1.data.spans.find((s) => s.name === 'trace1-child1');
        const trace1Child2 = trace1.data.spans.find((s) => s.name === 'trace1-child2');
        const trace1Grandchild = trace1.data.spans.find((s) => s.name === 'trace1-grandchild');

        expect(trace1Parent).toBeDefined();
        expect(trace1Child1).toBeDefined();
        expect(trace1Child2).toBeDefined();
        expect(trace1Grandchild).toBeDefined();

        // Verify parent-child relationships in trace1
        expect(trace1Parent?.parentId).toBeNull();
        expect(trace1Child1?.parentId).toBe(trace1Parent?.spanId);
        expect(trace1Child2?.parentId).toBe(trace1Parent?.spanId);
        expect(trace1Grandchild?.parentId).toBe(trace1Child2?.spanId);

        // Verify outputs
        expect(trace1Parent?.outputs).toEqual({ childResults: ['child1-result', 'child2-result'] });
        expect(trace1Child1?.outputs).toEqual({ processed: true });
        expect(trace1Child2?.outputs).toEqual({ grandchildResult: 'grandchild-result' });
        expect(trace1Grandchild?.outputs).toBe('grandchild-result');

        // Verify second trace
        const trace2 = await client.getTrace(traceIds['trace2-parent']);
        expect(trace2).toBeDefined();
        expect(trace2.data.spans.length).toBe(2); // parent + 1 child

        const trace2Parent = trace2.data.spans.find((s) => s.name === 'trace2-parent');
        const trace2Child = trace2.data.spans.find((s) => s.name === 'trace2-child');

        expect(trace2Parent).toBeDefined();
        expect(trace2Child).toBeDefined();
        expect(trace2Child?.parentId).toBe(trace2Parent?.spanId);
        expect(trace2Parent?.outputs).toEqual({ childResult: 'trace2-child-result' });
        expect(trace2Child?.outputs).toBe('trace2-child-result');

        // Verify third trace (simple synchronous)
        const trace3 = await client.getTrace(traceIds['trace3-simple']);
        expect(trace3).toBeDefined();
        expect(trace3.data.spans.length).toBe(1);

        const trace3Span = trace3.data.spans[0];
        expect(trace3Span.name).toBe('trace3-simple');
        expect(trace3Span.parentId).toBeNull();
        expect(trace3Span.outputs).toEqual({ immediate: true });
      });
    });
  });

  describe('trace wrapper function', () => {
    it('should trace a synchronous function', async () => {
      const myFunc = (x: number) => x * 2;

      const tracedFunc = mlflow.trace(myFunc);
      const result = tracedFunc(5);
      expect(result).toBe(10);

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');
      expect(trace.data.spans.length).toBe(1);

      const loggedSpan = trace.data.spans[0];
      expect(loggedSpan.name).toBe('myFunc');
      expect(loggedSpan.spanType).toBe(SpanType.UNKNOWN);
      expect(loggedSpan.inputs).toEqual({ x: 5 });
      expect(loggedSpan.outputs).toEqual(10);
      expect(loggedSpan.startTime).toBeDefined();
      expect(loggedSpan.endTime).toBeDefined();
    });

    it('should trace a synchronous function with options', async () => {
      const myFunc = (a: number, b: number) => a + b;

      const tracedFunc = mlflow.trace(myFunc, {
        name: 'custom_span_name',
        spanType: SpanType.LLM,
        attributes: { key: 'value' }
      });

      const result = tracedFunc(2, 3);
      expect(result).toBe(5);

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');
      expect(trace.data.spans.length).toBe(1);

      const loggedSpan = trace.data.spans[0];
      expect(loggedSpan.name).toBe('custom_span_name');
      expect(loggedSpan.spanType).toBe(SpanType.LLM);
      expect(loggedSpan.attributes['key']).toBe('value');
    });

    it('should trace an async function', async () => {
      async function myAsyncFunc(a: number, b: number): Promise<number> {
        await new Promise((resolve) => setTimeout(resolve, 10));
        return a + b;
      }

      const tracedFunc = mlflow.trace(myAsyncFunc, {
        name: 'async_span',
        spanType: SpanType.CHAIN
      });

      const result = await tracedFunc(10, 20);
      expect(result).toBe(30);

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');
      expect(trace.data.spans.length).toBe(1);

      const loggedSpan = trace.data.spans[0];
      expect(loggedSpan.name).toBe('async_span');
      expect(loggedSpan.spanType).toBe(SpanType.CHAIN);
      expect(loggedSpan.inputs).toEqual({ a: 10, b: 20 });
      expect(loggedSpan.outputs).toEqual(30);
      expect(loggedSpan.startTime).toBeDefined();
      expect(loggedSpan.endTime).toBeDefined();
    });

    it('should support inline function declaration', async () => {
      const myFunc = mlflow.trace(
        function myInlineFunc(a: number, b: number) {
          return a + b;
        },
        { spanType: SpanType.AGENT }
      );

      const result = myFunc(3, 4);
      expect(result).toBe(7);

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');
      expect(trace.data.spans.length).toBe(1);

      const loggedSpan = trace.data.spans[0];
      expect(loggedSpan.name).toBe('myInlineFunc');
      expect(loggedSpan.spanType).toBe(SpanType.AGENT);
      expect(loggedSpan.inputs).toEqual({ a: 3, b: 4 });
      expect(loggedSpan.outputs).toEqual(7);
      expect(loggedSpan.startTime).toBeDefined();
      expect(loggedSpan.endTime).toBeDefined();
    });

    it('should support inline async function', async () => {
      const myFunc = mlflow.trace(
        async (prompt: string) => {
          await new Promise((resolve) => setTimeout(resolve, 5));
          return `Response to: ${prompt}`;
        },
        { spanType: SpanType.LLM }
      );

      const result = await myFunc('Hello');
      expect(result).toBe('Response to: Hello');

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');
      expect(trace.data.spans.length).toBe(1);

      const loggedSpan = trace.data.spans[0];
      expect(loggedSpan.name).toBe('span'); // arrow function doesn't have a name
      expect(loggedSpan.spanType).toBe(SpanType.LLM);
      expect(loggedSpan.inputs).toEqual({ prompt: 'Hello' });
      expect(loggedSpan.outputs).toEqual('Response to: Hello');
      expect(loggedSpan.startTime).toBeDefined();
      expect(loggedSpan.endTime).toBeDefined();
    });
  });

  it('should handle synchronous errors in the traced function', async () => {
    const errorFunc = mlflow.trace(() => {
      throw new Error('Test error');
    });

    expect(() => errorFunc()).toThrow('Test error');

    const trace = await getLastActiveTrace();
    expect(trace.info.state).toBe('ERROR');
    expect(trace.data.spans.length).toBe(1);

    const loggedSpan = trace.data.spans[0];
    expect(loggedSpan.name).toBe('span'); // arrow function doesn't have a name
    expect(loggedSpan.inputs).toEqual({});
    expect(loggedSpan.outputs).toBeUndefined();
    expect(loggedSpan.startTime).toBeDefined();
    expect(loggedSpan.endTime).toBeDefined();
    expect(loggedSpan.status?.statusCode).toBe(SpanStatusCode.ERROR);
  });

  it('should handle async errors in the traced function', async () => {
    const errorFunc = mlflow.trace(
      async () => {
        await new Promise((resolve) => setTimeout(resolve, 5));
        throw new Error('Async test error');
      },
      { name: 'async_error_span', spanType: SpanType.LLM }
    );

    await expect(errorFunc()).rejects.toThrow('Async test error');

    const trace = await getLastActiveTrace();
    expect(trace.info.state).toBe('ERROR');
    expect(trace.data.spans.length).toBe(1);

    const loggedSpan = trace.data.spans[0];
    expect(loggedSpan.name).toBe('async_error_span');
    expect(loggedSpan.spanType).toBe(SpanType.LLM);
    expect(loggedSpan.inputs).toEqual({});
    expect(loggedSpan.outputs).toBeUndefined();
    expect(loggedSpan.startTime).toBeDefined();
    expect(loggedSpan.endTime).toBeDefined();
    expect(loggedSpan.status?.statusCode).toBe(SpanStatusCode.ERROR);
  });

  describe('getCurrentActiveSpan', () => {
    it('should return undefined when no span is active', () => {
      const activeSpan = mlflow.getCurrentActiveSpan();
      expect(activeSpan).toBeNull();
    });

    it('should return the current active span within withSpan context', () => {
      void mlflow.withSpan(
        (span) => {
          const activeSpan = mlflow.getCurrentActiveSpan();
          expect(activeSpan).not.toBeNull();
          expect(activeSpan?.spanId).toBe(span.spanId);
          expect(activeSpan?.traceId).toBe(span.traceId);
        },
        { name: 'test-span' }
      );
    });

    it('should return the current active span within traced function', () => {
      const tracedFunc = mlflow.trace(() => {
        const activeSpan = mlflow.getCurrentActiveSpan();
        expect(activeSpan).not.toBeNull();
        expect(activeSpan?.name).toBe('span'); // Default span name when function name is not available
        return 'result';
      });

      tracedFunc();
    });
  });

  describe('Exception Safety', () => {
    it('should return NoOpSpan when OTel tracer fails', () => {
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();

      // Mock the tracer to throw an error
      const getTracerSpy = jest
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        .spyOn(require('../../src/core/provider'), 'getTracer')
        .mockImplementation(() => {
          throw new Error('Tracer creation failed');
        });

      const span = mlflow.startSpan({ name: 'test-span' });

      expect(span.constructor.name).toBe('NoOpSpan');
      expect(consoleSpy).toHaveBeenCalledWith('Failed to start span', expect.any(Error));

      consoleSpy.mockRestore();
      getTracerSpy.mockRestore();
    });

    it('should fallback to raw args when argument mapping fails', () => {
      const consoleDebugSpy = jest.spyOn(console, 'debug').mockImplementation();

      // Mock mapArgsToObject to throw an error
      const mapArgsToObjectSpy = jest
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        .spyOn(require('../../src/core/utils'), 'mapArgsToObject')
        .mockImplementation(() => {
          throw new Error('Argument mapping failed');
        });

      const testFunc = (a: number, b: number) => a + b;
      const tracedFunc = mlflow.trace(testFunc);

      const result = tracedFunc(5, 10);
      expect(result).toBe(15);

      expect(consoleDebugSpy).toHaveBeenCalledWith(
        'Failed to map arguments values to names',
        expect.any(Error)
      );

      consoleDebugSpy.mockRestore();
      mapArgsToObjectSpy.mockRestore();
    });
  });

  describe('trace wrapper with class methods', () => {
    class TestClass {
      private value = 42;

      // Regular method
      multiply(x: number): number {
        return this.value * x;
      }

      // Async method
      async asyncMultiply(x: number): Promise<number> {
        await new Promise((resolve) => setTimeout(resolve, 10));
        return this.value * x;
      }

      // Static method
      static staticAdd(a: number, b: number): number {
        return a + b;
      }
    }

    class DecoratedTestClass {
      private value = 42;

      // Regular method with decorator
      @mlflow.trace()
      multiply(x: number): number {
        return this.value * x;
      }

      // Async method with decorator
      @mlflow.trace({ name: 'async_multiply_decorated', spanType: SpanType.TOOL })
      async asyncMultiply(x: number): Promise<number> {
        await new Promise((resolve) => setTimeout(resolve, 10));
        return this.value * x;
      }

      // Static method with decorator
      @mlflow.trace({ name: 'static_add_decorated', spanType: SpanType.TOOL })
      static staticAdd(a: number, b: number): number {
        return a + b;
      }
    }

    it('should trace a class method using decorator syntax', async () => {
      const testInstance = new DecoratedTestClass();

      const result = testInstance.multiply(5);
      expect(result).toBe(210); // 42 * 5

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');
      expect(trace.data.spans.length).toBe(1);

      const loggedSpan = trace.data.spans[0];
      expect(loggedSpan.name).toBe('multiply');
      expect(loggedSpan.spanType).toBe(SpanType.UNKNOWN);
      expect(loggedSpan.inputs).toEqual({ x: 5 });
      expect(loggedSpan.outputs).toEqual(210);
    });

    it('should trace an async class method using decorator syntax', async () => {
      const testInstance = new DecoratedTestClass();

      const result = await testInstance.asyncMultiply(3);
      expect(result).toBe(126); // 42 * 3

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');
      expect(trace.data.spans.length).toBe(1);

      const loggedSpan = trace.data.spans[0];
      expect(loggedSpan.name).toBe('async_multiply_decorated');
      expect(loggedSpan.spanType).toBe(SpanType.TOOL);
      expect(loggedSpan.inputs).toEqual({ x: 3 });
      expect(loggedSpan.outputs).toEqual(126);
    });

    it('should trace a static method using decorator syntax', async () => {
      const result = DecoratedTestClass.staticAdd(10, 20);
      expect(result).toBe(30);

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');
      expect(trace.data.spans.length).toBe(1);

      const loggedSpan = trace.data.spans[0];
      expect(loggedSpan.name).toBe('static_add_decorated');
      expect(loggedSpan.spanType).toBe(SpanType.TOOL);
      expect(loggedSpan.inputs).toEqual({ a: 10, b: 20 });
      expect(loggedSpan.outputs).toEqual(30);
    });

    it('should work with bound methods using function syntax', async () => {
      const testInstance = new TestClass();
      const tracedMethod = mlflow.trace(testInstance.multiply.bind(testInstance), {
        name: 'bound_multiply_method',
        spanType: SpanType.TOOL
      });

      const result = tracedMethod(7);
      expect(result).toBe(294); // 42 * 7

      const trace = await getLastActiveTrace();
      expect(trace.info.state).toBe('OK');
      expect(trace.data.spans.length).toBe(1);

      const loggedSpan = trace.data.spans[0];
      expect(loggedSpan.name).toBe('bound_multiply_method');
      expect(loggedSpan.spanType).toBe(SpanType.TOOL);
      expect(loggedSpan.inputs).toEqual({ args: [7] }); // Bound functions lose parameter names
      expect(loggedSpan.outputs).toEqual(294);
    });
  });

  describe('updateCurrentTrace', () => {
    it('should warn when called outside of a trace context', () => {
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();

      mlflow.updateCurrentTrace({ tags: { test: 'value' } });

      expect(consoleSpy).toHaveBeenCalledWith(
        'No active trace found. Please create a span using `withSpan` or ' +
          '`@trace` before calling `updateCurrentTrace`.'
      );

      consoleSpy.mockRestore();
    });

    it('should update trace tags and metadata within withSpan context', async () => {
      void mlflow.withSpan(
        (_span) => {
          mlflow.updateCurrentTrace({
            tags: {
              fruit: 'apple',
              color: 'red'
            },
            metadata: {
              'mlflow.source.name': 'test.ts',
              'mlflow.source.git.commit': '1234567890'
            }
          });
        },
        { name: 'test-span' }
      );

      const trace = await getLastActiveTrace();
      expect(trace.info.tags.fruit).toBe('apple');
      expect(trace.info.tags.color).toBe('red');
      expect(trace.info.traceMetadata['mlflow.source.name']).toBe('test.ts');
      expect(trace.info.traceMetadata['mlflow.source.git.commit']).toBe('1234567890');
    });

    it('should update client request ID', async () => {
      const predict = mlflow.trace(function predict(question: string) {
        mlflow.updateCurrentTrace({ clientRequestId: 'req-12345' });
        return `answer to ${question}`;
      });

      predict('What is the capital of France?');

      const trace = await getLastActiveTrace();
      expect(trace.info.clientRequestId).toBe('req-12345');
    });

    it('should update request and response previews', async () => {
      void mlflow.withSpan(
        (_span) => {
          mlflow.updateCurrentTrace({
            requestPreview: 'Custom request preview',
            responsePreview: 'Custom response preview'
          });
        },
        { name: 'test-span' }
      );

      const trace = await getLastActiveTrace();
      expect(trace.info.requestPreview).toBe('Custom request preview');
      expect(trace.info.responsePreview).toBe('Custom response preview');
    });
  });
});
