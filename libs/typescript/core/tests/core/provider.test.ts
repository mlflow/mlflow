import { BatchSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { MlflowSpanProcessor } from '../../src/exporters/mlflow';

const mockGetConfig = jest.fn();
const mockGetOtlpConfig = jest.fn();
const mockShouldUseOtlpExporter = jest.fn();
const mockCreateOtlpTraceExporter = jest.fn();
const mockGetTracerProvider = jest.fn();

jest.mock('../../src/core/config', () => ({
  getConfig: () => mockGetConfig()
}));

jest.mock('../../src/core/otlp_config', () => ({
  getOtlpConfig: () => mockGetOtlpConfig(),
  shouldUseOtlpExporter: () => mockShouldUseOtlpExporter()
}));

jest.mock('../../src/core/otlp_exporter_factory', () => ({
  createOtlpTraceExporter: () => mockCreateOtlpTraceExporter()
}));

jest.mock('../../src/clients', () => ({
  MlflowClient: jest.fn().mockImplementation(() => ({}))
}));

jest.mock('@opentelemetry/api', () => {
  const actual = jest.requireActual('@opentelemetry/api');
  return {
    ...actual,
    trace: {
      ...actual.trace,
      getTracerProvider: () => mockGetTracerProvider()
    }
  };
});

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
    jest.clearAllMocks();
    mockGetConfig.mockReturnValue(baseConfig);
    mockGetOtlpConfig.mockReturnValue({
      endpoint: undefined,
      protocol: 'http/protobuf',
      headers: {},
      metricsEndpoint: undefined,
      metricsProtocol: 'http/protobuf',
      enableExporter: true,
      dualExportEnabled: false
    });
    mockShouldUseOtlpExporter.mockReturnValue(false);
    mockCreateOtlpTraceExporter.mockReturnValue(null);
    delete process.env.MLFLOW_USE_DEFAULT_TRACER_PROVIDER;
    mockGetTracerProvider.mockReset();
    mockGetTracerProvider.mockReturnValue(undefined);
  });

  it('uses NodeSDK-managed provider when env flag is not disabled', () => {
    process.env.MLFLOW_USE_DEFAULT_TRACER_PROVIDER = 'true';
    mockGetTracerProvider.mockReturnValue({});

    const { initializeSDK } = require('../../src/core/provider');
    initializeSDK();

    const { NodeSDK } = require('@opentelemetry/sdk-node');
    expect(NodeSDK).toHaveBeenCalledTimes(1);
    const nodeConfig = NodeSDK.mock.calls[0][0];
    expect(nodeConfig.spanProcessors).toHaveLength(1);
    expect(nodeConfig.spanProcessors[0]).toBeInstanceOf(MlflowSpanProcessor);

    const instance = NodeSDK.mock.results[0]?.value as { start: jest.Mock } | undefined;
    expect(instance).toBeDefined();
    expect(instance?.start).toHaveBeenCalledTimes(1);
    delete process.env.MLFLOW_USE_DEFAULT_TRACER_PROVIDER;
  });

  it('attaches MlflowSpanProcessor to existing global provider when env flag is false', () => {
    process.env.MLFLOW_USE_DEFAULT_TRACER_PROVIDER = 'false';
    const existingProcessor = { name: 'existing' };
    const provider = {
      _activeSpanProcessor: {
        _spanProcessors: [existingProcessor]
      }
    };
    mockGetTracerProvider.mockReturnValue(provider as any);

    const { initializeSDK } = require('../../src/core/provider');
    initializeSDK();

    const attachedProcessors = provider._activeSpanProcessor!._spanProcessors!;
    expect(attachedProcessors).toHaveLength(2);
    const attached = attachedProcessors[attachedProcessors.length - 1];
    expect(attached).toBeInstanceOf(MlflowSpanProcessor);

    const { NodeSDK } = require('@opentelemetry/sdk-node');
    expect(NodeSDK).not.toHaveBeenCalled();
    delete process.env.MLFLOW_USE_DEFAULT_TRACER_PROVIDER;
  });

  it('replaces previous MlflowSpanProcessor when reinitializing in shared mode', () => {
    process.env.MLFLOW_USE_DEFAULT_TRACER_PROVIDER = 'false';
    const existingProcessor = { name: 'existing' };
    const provider = {
      _activeSpanProcessor: {
        _spanProcessors: [existingProcessor]
      }
    };
    mockGetTracerProvider.mockReturnValue(provider as any);

    const { initializeSDK } = require('../../src/core/provider');
    initializeSDK();
    const firstAttachedProcessors = provider._activeSpanProcessor!._spanProcessors!;
    const firstMlflowProcessor = firstAttachedProcessors[firstAttachedProcessors.length - 1];
    initializeSDK();
    const secondAttachedProcessors = provider._activeSpanProcessor!._spanProcessors!;
    const secondMlflowProcessor = secondAttachedProcessors[secondAttachedProcessors.length - 1];

    expect(firstMlflowProcessor).toBeInstanceOf(MlflowSpanProcessor);
    expect(secondMlflowProcessor).toBeInstanceOf(MlflowSpanProcessor);
    expect(secondMlflowProcessor).not.toBe(firstMlflowProcessor);

    const attachedMlflowProcessors = secondAttachedProcessors.filter(
      (processor) => processor instanceof MlflowSpanProcessor
    );
    expect(attachedMlflowProcessors).toHaveLength(1);
    expect(secondAttachedProcessors).toHaveLength(2);
    delete process.env.MLFLOW_USE_DEFAULT_TRACER_PROVIDER;
  });

  it('adds OTLP BatchSpanProcessor when exporter is configured and dual export enabled', () => {
    process.env.MLFLOW_USE_DEFAULT_TRACER_PROVIDER = 'true';
    mockShouldUseOtlpExporter.mockReturnValue(true);
    mockGetOtlpConfig.mockReturnValue({
      endpoint: 'http://collector:4318/v1/traces',
      protocol: 'http/protobuf',
      headers: { authorization: 'Bearer abc' },
      metricsEndpoint: undefined,
      metricsProtocol: 'http/protobuf',
      enableExporter: true,
      dualExportEnabled: true
    });
    const dummyExporter = {
      export: jest.fn(),
      shutdown: jest.fn().mockResolvedValue(undefined),
      forceFlush: jest.fn().mockResolvedValue(undefined)
    };
    mockCreateOtlpTraceExporter.mockReturnValue(dummyExporter);

    const { initializeSDK } = require('../../src/core/provider');
    initializeSDK();

    const { NodeSDK } = require('@opentelemetry/sdk-node');
    expect(NodeSDK).toHaveBeenCalledTimes(1);
    const nodeConfig = NodeSDK.mock.calls[0][0];
    expect(nodeConfig.spanProcessors).toHaveLength(2);
    expect(nodeConfig.spanProcessors[0]).toBeInstanceOf(BatchSpanProcessor);
    expect(nodeConfig.spanProcessors[1]).toBeInstanceOf(MlflowSpanProcessor);
    delete process.env.MLFLOW_USE_DEFAULT_TRACER_PROVIDER;
  });

  it('skips MlflowSpanProcessor when dual export disabled and OTLP exporter present', () => {
    process.env.MLFLOW_USE_DEFAULT_TRACER_PROVIDER = 'true';
    mockShouldUseOtlpExporter.mockReturnValue(true);
    mockGetOtlpConfig.mockReturnValue({
      endpoint: 'http://collector:4318/v1/traces',
      protocol: 'http/protobuf',
      headers: {},
      metricsEndpoint: undefined,
      metricsProtocol: 'http/protobuf',
      enableExporter: true,
      dualExportEnabled: false
    });
    const dummyExporter = {
      export: jest.fn(),
      shutdown: jest.fn().mockResolvedValue(undefined),
      forceFlush: jest.fn().mockResolvedValue(undefined)
    };
    mockCreateOtlpTraceExporter.mockReturnValue(dummyExporter);

    const { initializeSDK } = require('../../src/core/provider');
    initializeSDK();

    const { NodeSDK } = require('@opentelemetry/sdk-node');
    const nodeConfig = NodeSDK.mock.calls[0][0];
    expect(nodeConfig.spanProcessors).toHaveLength(1);
    expect(nodeConfig.spanProcessors[0]).toBeInstanceOf(BatchSpanProcessor);
    delete process.env.MLFLOW_USE_DEFAULT_TRACER_PROVIDER;
  });
});
