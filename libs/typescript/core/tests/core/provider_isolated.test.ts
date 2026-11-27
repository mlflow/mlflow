import { context, propagation, trace as otelTrace } from '@opentelemetry/api';
import { B3Propagator } from '@opentelemetry/propagator-b3';
import { BasicTracerProvider } from '@opentelemetry/sdk-trace-base';
import { MlflowClient } from '../../src/clients';
import { init } from '../../src/core/config';
import { flushTraces } from '../../src/core/provider';
import {
  getCurrentActiveSpan,
  startSpan,
  trace,
  updateCurrentTrace,
  withSpan,
} from '../../src/core/api';
import { NoOpSpan } from '../../src/core/entities/span';
import { InMemoryTraceManager } from '../../src/core/trace_manager';
import { TEST_TRACKING_URI } from '../helper';

beforeEach(() => {
  jest
    .spyOn(MlflowClient.prototype, 'createTrace')
    .mockImplementation((trace) => Promise.resolve(trace));
  jest.spyOn(MlflowClient.prototype, 'uploadTraceData').mockResolvedValue(undefined);
  jest.spyOn(MlflowClient.prototype, 'exportOtlpSpans').mockResolvedValue(undefined);
});

afterEach(async () => {
  await flushTraces();
  jest.restoreAllMocks();
  otelTrace.disable();
  propagation.disable();
});

function withEnv<T>(env: Record<string, string | undefined>, fn: () => T): T {
  const previous = Object.fromEntries(Object.keys(env).map((key) => [key, process.env[key]]));
  const assign = (values: Record<string, string | undefined>) => {
    for (const [key, value] of Object.entries(values)) {
      if (value === undefined) {
        delete process.env[key];
      } else {
        process.env[key] = value;
      }
    }
  };
  assign(env);
  try {
    return fn();
  } finally {
    assign(previous);
  }
}

function spanResource(env: Record<string, string | undefined>, registerGlobal: boolean) {
  withEnv(env, () =>
    init({
      trackingUri: TEST_TRACKING_URI,
      experimentId: '0',
      registerGlobalTracerProvider: registerGlobal,
    }),
  );
  const span = startSpan({ name: 'resource' });
  span.end();
  return span._span.resource.attributes;
}

it('runs traced code without init', () => {
  expect(withSpan(() => 42, { name: 'uninitialized' })).toBe(42);
  expect(trace((a: number, b: number) => a + b)(1, 2)).toBe(3);
});

it.each([true, false])(
  'reconfigures the client with global registration %s',
  async (registerGlobalTracerProvider) => {
    const hosts: string[] = [];
    jest.spyOn(MlflowClient.prototype, 'createTrace').mockImplementation(function (
      this: MlflowClient,
      trace,
    ) {
      hosts.push(this.getHost());
      return Promise.resolve(trace);
    });
    const options = { experimentId: '0', registerGlobalTracerProvider };
    init({ ...options, trackingUri: 'http://localhost:6781' });
    startSpan({ name: 'before' }).end();
    await flushTraces();
    init({ ...options, trackingUri: 'http://localhost:6782' });
    startSpan({ name: 'after' }).end();
    await flushTraces();
    expect(hosts).toEqual(['http://localhost:6781', 'http://localhost:6782']);
  },
);

it('records MLflow spans without replacing an existing global provider', () => {
  const host = new BasicTracerProvider();
  expect(otelTrace.setGlobalTracerProvider(host)).toBe(true);
  const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
  try {
    init({ trackingUri: TEST_TRACKING_URI, experimentId: '0' });
    void host.getTracer('host').startActiveSpan('host', (hostSpan) => {
      try {
        const span = startSpan({ name: 'mlflow root' });
        expect(span.traceId).toMatch(/^tr-/);
        expect(span.parentId).toBeNull();
        span.end();
        return withSpan((active) => {
          expect(active.traceId).toMatch(/^tr-/);
          expect(active.parentId).toBeNull();
        });
      } finally {
        hostSpan.end();
      }
    });
    expect(otelTrace.getTracer('host')).toBe(host.getTracer('host'));
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('Global tracer registration failed'));
  } finally {
    otelTrace.disable();
  }
});

it('reconfigures an isolated client without touching the host global provider', async () => {
  const host = new BasicTracerProvider();
  expect(otelTrace.setGlobalTracerProvider(host)).toBe(true);
  const hosts: string[] = [];
  jest.spyOn(MlflowClient.prototype, 'createTrace').mockImplementation(function (
    this: MlflowClient,
    trace,
  ) {
    hosts.push(this.getHost());
    return Promise.resolve(trace);
  });
  const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
  try {
    const options = { experimentId: '0', registerGlobalTracerProvider: false };
    init({ ...options, trackingUri: 'http://localhost:6781' });
    startSpan({ name: 'before' }).end();
    await flushTraces();
    init({ ...options, trackingUri: 'http://localhost:6782' });
    startSpan({ name: 'after' }).end();
    await flushTraces();

    expect(hosts).toEqual(['http://localhost:6781', 'http://localhost:6782']);
    expect(otelTrace.getTracer('host')).toBe(host.getTracer('host'));
    expect(warn).not.toHaveBeenCalled();
  } finally {
    otelTrace.disable();
  }
});

describe('init() with registerGlobalTracerProvider: false', () => {
  beforeAll(() => {
    init({
      trackingUri: TEST_TRACKING_URI,
      experimentId: '0',
      registerGlobalTracerProvider: false,
    });
  });

  it('does not record spans that other libraries start through the global OpenTelemetry API', () => {
    const foreignSpan = otelTrace.getTracer('some-llm-sdk').startSpan('llm.request');
    foreignSpan.end();

    expect(foreignSpan.isRecording()).toBe(false);
    expect(
      InMemoryTraceManager.getInstance().getMlflowTraceIdFromOtelId(
        foreignSpan.spanContext().traceId,
      ),
    ).toBeNull();
  });

  it('still records spans created with the MLflow API', () => {
    const span = startSpan({ name: 'root' });
    const child = startSpan({ name: 'child', parent: span });
    child.end();
    span.end();

    expect(span.traceId).toMatch(/^tr-/);
    expect(child.traceId).toBe(span.traceId);
    expect(child.parentId).toBe(span.spanId);
  });

  it('starts new traces under a foreign active span', () => {
    const foreignSpan = new BasicTracerProvider().getTracer('host').startSpan('host');
    try {
      context.with(otelTrace.setSpan(context.active(), foreignSpan), () => {
        const span = startSpan({ name: 'root' });
        span.end();
        expect(span.traceId).toMatch(/^tr-/);
        expect(span.parentId).toBeNull();
        void withSpan(
          (active) => {
            expect(active.traceId).toMatch(/^tr-/);
            expect(active.parentId).toBeNull();
            updateCurrentTrace({ tags: { isolated: 'yes' } });
            expect(
              InMemoryTraceManager.getInstance().getTrace(active.traceId)?.info.tags.isolated,
            ).toBe('yes');
          },
          { name: 'active root' },
        );
      });
    } finally {
      foreignSpan.end();
    }
  });

  it('keeps the active MLflow parent through a host span', async () => {
    const host = new BasicTracerProvider().getTracer('host');
    await withSpan(
      (outer) =>
        host.startActiveSpan('host', (hostSpan) => {
          try {
            expect(getCurrentActiveSpan()).toBe(outer);
            updateCurrentTrace({ tags: { nested: 'yes' } });
            expect(
              InMemoryTraceManager.getInstance().getTrace(outer.traceId)?.info.tags.nested,
            ).toBe('yes');
            const child = startSpan({ name: 'manual child' });
            expect(child.parentId).toBe(outer.spanId);
            child.end();
            return withSpan((inner) => {
              expect(inner.traceId).toBe(outer.traceId);
              expect(inner.parentId).toBe(outer.spanId);
            });
          } finally {
            hostSpan.end();
          }
        }),
      { name: 'outer' },
    );
  });

  it('keeps an incoming remote trace ID', () => {
    const traceId = '0af7651916cd43dd8448eb211c80319c';
    const remote = otelTrace.setSpanContext(context.active(), {
      traceId,
      spanId: 'b7ad6b7169203331',
      traceFlags: 1,
      isRemote: true,
    });
    context.with(remote, () => {
      const span = startSpan({ name: 'remote child' });
      expect(span.traceId).toBe('tr-' + traceId);
      span.end();
    });
  });

  it('still nests spans through the active context', async () => {
    let parentSpanId: string | undefined;
    let childParentId: string | null | undefined;

    await withSpan(
      async (parent) => {
        parentSpanId = parent.spanId;
        await Promise.resolve();
        expect(getCurrentActiveSpan()?.spanId).toBe(parent.spanId);
        void withSpan(
          (child) => {
            childParentId = child.parentId;
          },
          { name: 'child' },
        );
      },
      { name: 'parent' },
    );

    expect(parentSpanId).toBeDefined();
    expect(childParentId).toBe(parentSpanId);
  });
});

describe.each([true, false])('init() environment with global registration %s', (registerGlobal) => {
  it('returns no-op spans when OTEL_SDK_DISABLED is true', () => {
    jest.spyOn(console, 'warn').mockImplementation(() => {});
    withEnv({ OTEL_SDK_DISABLED: 'true' }, () =>
      init({
        trackingUri: TEST_TRACKING_URI,
        experimentId: '0',
        registerGlobalTracerProvider: registerGlobal,
      }),
    );
    expect(startSpan({ name: 'disabled' })).toBeInstanceOf(NoOpSpan);
  });

  it('detects env, process and host attributes by default', () => {
    const attributes = spanResource(
      { OTEL_SERVICE_NAME: 'svc', OTEL_NODE_RESOURCE_DETECTORS: undefined },
      registerGlobal,
    );
    expect(attributes['service.name']).toBe('svc');
    expect(attributes['process.pid']).toBe(process.pid);
    expect(attributes['host.arch']).toBeDefined();
  });

  it('honors OTEL_NODE_RESOURCE_DETECTORS', () => {
    const attributes = spanResource(
      {
        OTEL_SERVICE_NAME: 'svc',
        OTEL_RESOURCE_ATTRIBUTES: 'team=tracing',
        OTEL_NODE_RESOURCE_DETECTORS: 'env',
      },
      registerGlobal,
    );
    expect(attributes['service.name']).toBe('svc');
    expect(attributes['team']).toBe('tracing');
    expect(Object.keys(attributes).filter((key) => /^(host|process)\./.test(key))).toEqual([]);
  });
});

describe('global propagator', () => {
  const initWith = (env: Record<string, string | undefined>, registerGlobal = true) =>
    withEnv(env, () =>
      init({
        trackingUri: TEST_TRACKING_URI,
        experimentId: '0',
        registerGlobalTracerProvider: registerGlobal,
      }),
    );

  it.each([
    [undefined, ['traceparent', 'tracestate', 'baggage']],
    ['none', []],
    ['b3', ['b3']],
    ['tracecontext,b3', ['traceparent', 'tracestate', 'b3']],
  ])('follows OTEL_PROPAGATORS=%s', (value, fields) => {
    initWith({ OTEL_PROPAGATORS: value });
    expect(propagation.fields()).toEqual(fields);
  });

  it('replaces its own propagator on reinitialization', () => {
    initWith({ OTEL_PROPAGATORS: 'b3' });
    initWith({ OTEL_PROPAGATORS: 'tracecontext' });
    expect(propagation.fields()).toEqual(['traceparent', 'tracestate']);
  });

  it.each([true, false])(
    'keeps a host propagator registered after its own on reinitialization with global registration %s',
    (registerGlobal) => {
      initWith({ OTEL_PROPAGATORS: undefined });
      propagation.disable();
      propagation.setGlobalPropagator(new B3Propagator());
      initWith({ OTEL_PROPAGATORS: undefined }, registerGlobal);
      expect(propagation.fields()).toEqual(['b3']);
    },
  );

  it('registers no propagator with registerGlobalTracerProvider: false', () => {
    initWith({ OTEL_PROPAGATORS: undefined }, false);
    expect(propagation.fields()).toEqual([]);
  });
});
