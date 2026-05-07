import { describe, expect, test } from '@jest/globals';
import { createIntl, createIntlCache } from 'react-intl';
import {
  DEFAULT_TRACE_ARCHIVAL_RETENTION_UNIT,
  decodeTraceArchivalRetentionTag,
  encodeTraceArchivalRetentionTag,
  formatTraceArchivalRetention,
  formatTraceArchivalRetentionForDisplay,
  getTraceArchivalRetentionValidationError,
  parseTraceArchivalRetention,
  validateTraceArchivalLocation,
  validateTraceArchivalRetention,
} from './traceArchival';

const intl = createIntl({ locale: 'en' }, createIntlCache());

describe('traceArchival', () => {
  test('treats invalid retention strings as empty input state', () => {
    expect(parseTraceArchivalRetention('abcd')).toEqual({
      amount: '',
      unit: DEFAULT_TRACE_ARCHIVAL_RETENTION_UNIT,
    });
    expect(parseTraceArchivalRetention('0d')).toEqual({
      amount: '',
      unit: DEFAULT_TRACE_ARCHIVAL_RETENTION_UNIT,
    });
  });

  test('parses valid retention strings', () => {
    expect(parseTraceArchivalRetention('30d')).toEqual({
      amount: '30',
      unit: 'd',
    });
  });

  test('validates retention strings with localized messages', () => {
    expect(validateTraceArchivalRetention('30d', intl)).toEqual({ valid: true });
    expect(validateTraceArchivalRetention('0d', intl)).toEqual({
      valid: false,
      error: "Trace archival retention must use the format <int><unit>, where unit is one of 'm', 'h', or 'd'.",
    });
  });

  test('validates trace archival locations with localized messages', () => {
    expect(validateTraceArchivalLocation('s3://bucket/path', intl)).toEqual({ valid: true });
    expect(validateTraceArchivalLocation('a', intl)).toEqual({
      valid: false,
      error: 'Trace archival location must look like a URI, for example s3://bucket/path.',
    });
  });

  test('formats retention strings from amount and unit', () => {
    expect(formatTraceArchivalRetention(' 14 ', 'd')).toBe('14d');
    expect(formatTraceArchivalRetention('', 'd')).toBe('');
  });

  test('returns validation errors from amount and unit pairs', () => {
    expect(getTraceArchivalRetentionValidationError('14', 'd', intl)).toBeUndefined();
    expect(getTraceArchivalRetentionValidationError('0', 'd', intl)).toBe(
      "Trace archival retention must use the format <int><unit>, where unit is one of 'm', 'h', or 'd'.",
    );
  });

  test('formats valid retention strings for display and hides malformed ones', () => {
    expect(formatTraceArchivalRetentionForDisplay('30d', intl)).toBe('30 days');
    expect(formatTraceArchivalRetentionForDisplay('1h', intl)).toBe('1 hour');
    expect(formatTraceArchivalRetentionForDisplay('future-format', intl)).toBe('');
  });

  test('round-trips encoded retention tags and ignores malformed payloads', () => {
    expect(decodeTraceArchivalRetentionTag(encodeTraceArchivalRetentionTag('30d'))).toBe('30d');
    expect(decodeTraceArchivalRetentionTag('{"type":"other","value":"30d"}')).toBe('');
    expect(decodeTraceArchivalRetentionTag('not-json')).toBe('');
  });
});
