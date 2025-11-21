/**
 * Helper utilities for determining OTLP export settings.
 * Mirrors the Python logic in mlflow/tracing/utils/otlp.py.
 */

export type OtlpProtocol = 'grpc' | 'http/protobuf';

export interface OtlpConfig {
  endpoint?: string;
  protocol: OtlpProtocol;
  headers: Record<string, string>;
  metricsEndpoint?: string;
  metricsProtocol: OtlpProtocol;
  enableExporter: boolean;
  dualExportEnabled: boolean;
}

const TRUE_VALUES = new Set(['1', 'true', 'yes', 'on']);
const FALSE_VALUES = new Set(['0', 'false', 'no', 'off']);

export function getOtlpConfig(): OtlpConfig {
  const tracesEndpoint = process.env.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT ??
    process.env.OTEL_EXPORTER_OTLP_ENDPOINT;
  const metricsEndpoint = process.env.OTEL_EXPORTER_OTLP_METRICS_ENDPOINT ??
    process.env.OTEL_EXPORTER_OTLP_ENDPOINT;

  return {
    endpoint: tracesEndpoint,
    protocol: getProtocol('OTEL_EXPORTER_OTLP_TRACES_PROTOCOL', 'http/protobuf'),
    headers: getHeaders(),
    metricsEndpoint,
    metricsProtocol: getProtocol('OTEL_EXPORTER_OTLP_METRICS_PROTOCOL', 'http/protobuf'),
    enableExporter: getBooleanEnv('MLFLOW_ENABLE_OTLP_EXPORTER', true),
    dualExportEnabled: getBooleanEnv('MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT', false)
  };
}

export function shouldUseOtlpExporter(config: OtlpConfig): boolean {
  return config.enableExporter && !!config.endpoint;
}

function getProtocol(envVar: string, defaultValue: OtlpProtocol): OtlpProtocol {
  const envValue = process.env[envVar] ?? process.env.OTEL_EXPORTER_OTLP_PROTOCOL;
  if (!envValue) {
    return defaultValue;
  }
  if (envValue === 'grpc' || envValue === 'http/protobuf') {
    return envValue;
  }
  return defaultValue;
}

function getHeaders(): Record<string, string> {
  const headerString =
    process.env.OTEL_EXPORTER_OTLP_TRACES_HEADERS ?? process.env.OTEL_EXPORTER_OTLP_HEADERS;
  if (!headerString) {
    return {};
  }

  const result: Record<string, string> = {};
  for (const pair of headerString.split(',')) {
    const [rawKey, rawValue] = pair.split('=');
    if (!rawKey || rawValue === undefined) {
      continue;
    }
    const key = rawKey.trim();
    const value = rawValue.trim();
    if (key && value) {
      result[key] = value;
    }
  }
  return result;
}

function getBooleanEnv(name: string, defaultValue: boolean): boolean {
  const value = process.env[name];
  if (value === undefined) {
    return defaultValue;
  }
  const normalized = value.trim().toLowerCase();
  if (TRUE_VALUES.has(normalized)) {
    return true;
  }
  if (FALSE_VALUES.has(normalized)) {
    return false;
  }
  return defaultValue;
}
