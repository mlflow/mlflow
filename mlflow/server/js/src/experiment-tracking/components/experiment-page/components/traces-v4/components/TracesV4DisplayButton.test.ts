import { describe, expect, test } from '@jest/globals';
import { parseSortValue } from './TracesV4DisplayButton';

describe('parseSortValue', () => {
  test('preserves the colon in an assessment column id', () => {
    expect(parseSortValue('assessment:quality:desc')).toEqual(['assessment:quality', 'desc']);
  });

  test('parses a standard sort column', () => {
    expect(parseSortValue('state:asc')).toEqual(['state', 'asc']);
  });
});
