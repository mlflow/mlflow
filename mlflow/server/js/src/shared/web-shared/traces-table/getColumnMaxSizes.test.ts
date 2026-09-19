import { describe, expect, test } from '@jest/globals';
import { createIntl } from '@databricks/i18n';
import { COLUMN_SIZES } from './constants';
import { getContentColumnMaxSizes } from './getColumnMaxSizes';
import { makeTrace } from './test-utils/mockTraces';

const intl = createIntl({ locale: 'en' });

describe('getContentColumnMaxSizes', () => {
  test('keeps compact columns at least as wide as their defaults', () => {
    const sizes = getContentColumnMaxSizes([makeTrace('short')], intl);
    expect(sizes.tokens).toBeGreaterThanOrEqual(COLUMN_SIZES.tokens.size);
    expect(sizes.start_time).toBeGreaterThanOrEqual(COLUMN_SIZES.start_time.size);
  });

  test('expands the resize ceiling to reveal the longest value on the page', () => {
    const short = getContentColumnMaxSizes([makeTrace('short')], intl);
    const long = getContentColumnMaxSizes([makeTrace('trace-id-that-is-much-longer-than-the-default-value')], intl);
    expect(long.trace_id).toBeGreaterThan(short.trace_id ?? 0);
  });

  test('returns usable default ceilings for an empty page', () => {
    const sizes = getContentColumnMaxSizes([], intl);
    for (const columnId of ['trace_id', 'start_time', 'session', 'duration', 'state', 'tokens', 'cost'] as const) {
      expect(sizes[columnId]).toBeGreaterThanOrEqual(COLUMN_SIZES[columnId].size);
    }
  });
});
