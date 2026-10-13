import type { IntlShape } from 'react-intl';

export const TRACE_ARCHIVAL_RETENTION_TAG_KEY = 'mlflow.trace.archivalRetention';
export const TRACE_ARCHIVAL_RETENTION_PATTERN = /^[1-9][0-9]*[mhd]$/;
export const TRACE_ARCHIVAL_RETENTION_MAX_LENGTH = 32;
export const TRACE_ARCHIVAL_URI_PATTERN = /^[a-zA-Z][a-zA-Z0-9+.-]*:.+$/;

export type TraceArchivalRetentionValidationResult = {
  valid: boolean;
  error?: string;
};

export type TraceArchivalLocationValidationResult = {
  valid: boolean;
  error?: string;
};

export type TraceArchivalRetentionUnit = 'm' | 'h' | 'd';
export const DEFAULT_TRACE_ARCHIVAL_RETENTION_UNIT: TraceArchivalRetentionUnit = 'd';

type TraceArchivalRetentionTagPayload = {
  type: 'duration';
  value: string;
};

type TraceArchivalIntl = Pick<IntlShape, 'formatMessage'>;

const getTraceArchivalRetentionTooLongMessage = (intl: TraceArchivalIntl) =>
  intl.formatMessage(
    {
      defaultMessage: 'Trace archival retention must be at most {maxLength} characters.',
      description: 'Validation error for trace archival retention exceeding the maximum length',
    },
    { maxLength: TRACE_ARCHIVAL_RETENTION_MAX_LENGTH },
  );

const getTraceArchivalRetentionFormatMessage = (intl: TraceArchivalIntl) =>
  intl.formatMessage({
    defaultMessage:
      "Trace archival retention must use the format '<int><unit>', where unit is one of 'm', 'h', or 'd'.",
    description: 'Validation error for invalid trace archival retention format',
  });

const getTraceArchivalLocationFormatMessage = (intl: TraceArchivalIntl) =>
  intl.formatMessage({
    defaultMessage: 'Trace archival location must look like a URI, for example s3://bucket/path.',
    description: 'Validation error for invalid trace archival location URI',
  });

const getTraceArchivalLocationProxySchemeMessage = (intl: TraceArchivalIntl) =>
  intl.formatMessage({
    defaultMessage: 'Trace archival location cannot use the proxy-only `mlflow-artifacts:` scheme.',
    description: 'Validation error for unsupported trace archival proxy URI scheme',
  });

const formatRetentionUnitForDisplay = (amount: string, unit: TraceArchivalRetentionUnit, intl: TraceArchivalIntl) => {
  const count = Number(amount);
  switch (unit) {
    case 'd':
      return intl.formatMessage(
        {
          defaultMessage: '{count} {count, plural, one {day} other {days}}',
          description: 'Formatted trace archival retention in days',
        },
        { count },
      );
    case 'h':
      return intl.formatMessage(
        {
          defaultMessage: '{count} {count, plural, one {hour} other {hours}}',
          description: 'Formatted trace archival retention in hours',
        },
        { count },
      );
    case 'm':
      return intl.formatMessage(
        {
          defaultMessage: '{count} {count, plural, one {minute} other {minutes}}',
          description: 'Formatted trace archival retention in minutes',
        },
        { count },
      );
  }
};

export const validateTraceArchivalRetention = (
  value: string,
  intl: TraceArchivalIntl,
): TraceArchivalRetentionValidationResult => {
  const trimmedValue = value.trim();
  if (!trimmedValue) {
    return { valid: true };
  }

  if (trimmedValue.length > TRACE_ARCHIVAL_RETENTION_MAX_LENGTH) {
    return {
      valid: false,
      error: getTraceArchivalRetentionTooLongMessage(intl),
    };
  }

  if (!TRACE_ARCHIVAL_RETENTION_PATTERN.test(trimmedValue)) {
    return {
      valid: false,
      error: getTraceArchivalRetentionFormatMessage(intl),
    };
  }

  return { valid: true };
};

export const validateTraceArchivalLocation = (
  value: string,
  intl: TraceArchivalIntl,
): TraceArchivalLocationValidationResult => {
  const trimmedValue = value.trim();
  if (!trimmedValue) {
    return { valid: true };
  }

  if (!TRACE_ARCHIVAL_URI_PATTERN.test(trimmedValue)) {
    return {
      valid: false,
      error: getTraceArchivalLocationFormatMessage(intl),
    };
  }

  if (trimmedValue.toLowerCase().startsWith('mlflow-artifacts:')) {
    return {
      valid: false,
      error: getTraceArchivalLocationProxySchemeMessage(intl),
    };
  }

  return { valid: true };
};

export const parseTraceArchivalRetention = (
  value: string | null | undefined,
): { amount: string; unit: TraceArchivalRetentionUnit } => {
  const trimmedValue = value?.trim() ?? '';
  const match = trimmedValue.match(/^([1-9][0-9]*)([mhd])$/);
  if (!match) {
    return { amount: '', unit: DEFAULT_TRACE_ARCHIVAL_RETENTION_UNIT };
  }

  return {
    amount: match[1],
    unit: match[2] as TraceArchivalRetentionUnit,
  };
};

export const formatTraceArchivalRetention = (amount: string, unit: TraceArchivalRetentionUnit) => {
  const trimmedAmount = amount.trim();
  return trimmedAmount ? `${trimmedAmount}${unit}` : '';
};

export const getTraceArchivalRetentionValidationError = (
  amount: string,
  unit: TraceArchivalRetentionUnit,
  intl: TraceArchivalIntl,
) => {
  const result = validateTraceArchivalRetention(formatTraceArchivalRetention(amount, unit), intl);
  return result.valid ? undefined : result.error;
};

export const formatTraceArchivalRetentionForDisplay = (value: string | null | undefined, intl: TraceArchivalIntl) => {
  const trimmedValue = value?.trim() ?? '';
  if (!trimmedValue) {
    return '';
  }

  if (!TRACE_ARCHIVAL_RETENTION_PATTERN.test(trimmedValue)) {
    return '';
  }

  const { amount, unit } = parseTraceArchivalRetention(trimmedValue);
  return formatRetentionUnitForDisplay(amount, unit, intl);
};

export const encodeTraceArchivalRetentionTag = (value: string) =>
  JSON.stringify({
    type: 'duration',
    value: value.trim(),
  } satisfies TraceArchivalRetentionTagPayload);

export const decodeTraceArchivalRetentionTag = (value?: string | null) => {
  const trimmedValue = value?.trim() ?? '';
  if (!trimmedValue) {
    return '';
  }

  try {
    const payload = JSON.parse(trimmedValue);
    if (
      typeof payload === 'object' &&
      payload !== null &&
      payload.type === 'duration' &&
      typeof payload.value === 'string'
    ) {
      return payload.value.trim();
    }
  } catch {
    return '';
  }

  return '';
};
