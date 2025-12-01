import { getOtlpConfig, shouldUseOtlpExporter } from '../../src/core/otlp_config';

describe('otlp_config', () => {
  const ORIGINAL_ENV = process.env;

  beforeEach(() => {
    process.env = { ...ORIGINAL_ENV };
    delete process.env.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT;
    delete process.env.OTEL_EXPORTER_OTLP_ENDPOINT;
    delete process.env.OTEL_EXPORTER_OTLP_TRACES_PROTOCOL;
    delete process.env.OTEL_EXPORTER_OTLP_PROTOCOL;
    delete process.env.OTEL_EXPORTER_OTLP_TRACES_HEADERS;
    delete process.env.OTEL_EXPORTER_OTLP_HEADERS;
    delete process.env.MLFLOW_ENABLE_OTLP_EXPORTER;
    delete process.env.MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT;
  });

  afterAll(() => {
    process.env = ORIGINAL_ENV;
  });

  it('should disable exporter when endpoint missing', () => {
    const config = getOtlpConfig();
    expect(config.endpoint).toBeUndefined();
    expect(config.protocol).toBe('http/protobuf');
    expect(config.enableExporter).toBe(true);
    expect(shouldUseOtlpExporter(config)).toBe(false);
  });

  it('should read endpoint, protocol, headers, and flags from env', () => {
    process.env.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT = 'https://collector.example.com/v1/traces';
    process.env.OTEL_EXPORTER_OTLP_TRACES_PROTOCOL = 'grpc';
    process.env.OTEL_EXPORTER_OTLP_TRACES_HEADERS = 'x-token=abc,y=z';
    process.env.MLFLOW_ENABLE_OTLP_EXPORTER = 'false';
    process.env.MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT = 'true';

    const config = getOtlpConfig();
    expect(config.endpoint).toBe('https://collector.example.com/v1/traces');
    expect(config.protocol).toBe('grpc');
    expect(config.headers).toEqual({ 'x-token': 'abc', y: 'z' });
    expect(config.enableExporter).toBe(false);
    expect(config.dualExportEnabled).toBe(true);
    expect(shouldUseOtlpExporter(config)).toBe(false);
  });

  it('should fall back to generic OTLP endpoint and headers when trace-specific not set', () => {
    process.env.OTEL_EXPORTER_OTLP_ENDPOINT = 'https://generic-collector:4318';
    process.env.OTEL_EXPORTER_OTLP_HEADERS = 'authorization=Bearer 123';
    process.env.MLFLOW_ENABLE_OTLP_EXPORTER = '1';

    const config = getOtlpConfig();
    expect(config.endpoint).toBe('https://generic-collector:4318');
    expect(config.headers).toEqual({ authorization: 'Bearer 123' });
    expect(shouldUseOtlpExporter(config)).toBe(true);
  });
});
