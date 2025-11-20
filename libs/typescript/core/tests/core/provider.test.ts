import { trace } from '@opentelemetry/api';
import { MlflowSpanProcessor } from '../../src/exporters/mlflow';

const mockGetConfig = jest.fn();

jest.mock('../../src/core/config', () => ({
  getConfig: () => mockGetConfig()
}));

jest.mock('../../src/clients', () => ({
  MlflowClient: jest.fn().mockImplementation(() => ({}))
}));

jest.mock('@opentelemetry/sdk-node', () => {
  return {
    NodeSDK: jest.fn().mockImplementation((config) => ({
      config,
      start: jest.fn(),
      shutdown: jest.fn().mockResolvedValue(undefined)
    }))
  };
});

describe('initializeSDK', () => {
  const baseConfig = {
    trackingUri: 'http://localhost:5000',
    experimentId: '0',
    host: 'http://localhost:5000'
  };

  beforeEach(() => {
    jest.resetModules();
    jest.clearAllMocks();
    mockGetConfig.mockReturnValue(baseConfig);
    delete process.env.MLFLOW_USE_DEFAULT_TRACER_PROVIDER;
  });

  it('uses NodeSDK-managed provider when env flag is not disabled', () => {
    const tracerSpy = jest.spyOn(trace, 'getTracerProvider').mockReturnValue({} as unknown as any);

    jest.isolateModules(() => {
      const { initializeSDK } = require('../../src/core/provider');
      initializeSDK();
    });

    const { NodeSDK } = require('@opentelemetry/sdk-node');
    expect(NodeSDK).toHaveBeenCalledTimes(1);
    const nodeConfig = NodeSDK.mock.calls[0][0];
    expect(nodeConfig.spanProcessors).toHaveLength(1);
    expect(nodeConfig.spanProcessors[0]).toBeInstanceOf(MlflowSpanProcessor);

    const instance = NodeSDK.mock.instances[0];
    expect(instance.start).toHaveBeenCalledTimes(1);

    tracerSpy.mockRestore();
  });

  it('attaches MlflowSpanProcessor to existing global provider when env flag is false', () => {
    process.env.MLFLOW_USE_DEFAULT_TRACER_PROVIDER = 'false';
    const existingProcessor = { name: 'existing' };
    const spanProcessors: unknown[] = [existingProcessor];
    const provider = {
      _activeSpanProcessor: {
        _spanProcessors: spanProcessors
      }
    };
    const tracerSpy = jest.spyOn(trace, 'getTracerProvider').mockReturnValue(provider as any);

    jest.isolateModules(() => {
      const { initializeSDK } = require('../../src/core/provider');
      initializeSDK();
    });

    expect(spanProcessors).toHaveLength(2);
    const attached = spanProcessors[1];
    expect(attached).toBeInstanceOf(MlflowSpanProcessor);

    const { NodeSDK } = require('@opentelemetry/sdk-node');
    expect(NodeSDK).not.toHaveBeenCalled();

    tracerSpy.mockRestore();
  });

  it('replaces previous MlflowSpanProcessor when reinitializing in shared mode', () => {
    process.env.MLFLOW_USE_DEFAULT_TRACER_PROVIDER = 'false';
    const existingProcessor = { name: 'existing' };
    const spanProcessors: unknown[] = [existingProcessor];
    const provider = {
      _activeSpanProcessor: {
        _spanProcessors: spanProcessors
      }
    };
    const tracerSpy = jest.spyOn(trace, 'getTracerProvider').mockReturnValue(provider as any);

    jest.isolateModules(() => {
      const { initializeSDK } = require('../../src/core/provider');
      initializeSDK();
      const firstMlflowProcessor = spanProcessors[spanProcessors.length - 1];
      initializeSDK();
      const secondMlflowProcessor = spanProcessors[spanProcessors.length - 1];

      expect(firstMlflowProcessor).toBeInstanceOf(MlflowSpanProcessor);
      expect(secondMlflowProcessor).toBeInstanceOf(MlflowSpanProcessor);
      expect(secondMlflowProcessor).not.toBe(firstMlflowProcessor);

      const attachedMlflowProcessors = spanProcessors.filter(
        (processor) => processor instanceof MlflowSpanProcessor
      );
      expect(attachedMlflowProcessors).toHaveLength(1);
      expect(spanProcessors).toHaveLength(2);
    });

    tracerSpy.mockRestore();
  });
});
