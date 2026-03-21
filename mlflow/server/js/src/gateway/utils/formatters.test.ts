import { describe, it, expect } from '@jest/globals';
import { formatTokens, formatCost, extractModelDate, extractModelVersion, sortModelsByDate } from './formatters';

describe('formatTokens', () => {
  it('returns null for null or undefined', () => {
    expect(formatTokens(null)).toBeNull();
    expect(formatTokens(undefined)).toBeNull();
  });

  it('formats millions', () => {
    expect(formatTokens(1_000_000)).toBe('1.0M');
    expect(formatTokens(2_500_000)).toBe('2.5M');
  });

  it('formats thousands', () => {
    expect(formatTokens(1_000)).toBe('1K');
    expect(formatTokens(128_000)).toBe('128K');
  });

  it('returns raw number for small values', () => {
    expect(formatTokens(500)).toBe('500');
  });
});

describe('formatCost', () => {
  it('returns null for null or undefined', () => {
    expect(formatCost(null)).toBeNull();
    expect(formatCost(undefined)).toBeNull();
  });

  it('returns Free for zero', () => {
    expect(formatCost(0)).toBe('Free');
  });

  it('formats cost per million tokens', () => {
    expect(formatCost(0.000001)).toBe('$1.00/1M');
    expect(formatCost(0.00001)).toBe('$10.00/1M');
  });

  it('uses more precision for very small costs', () => {
    expect(formatCost(0.000000001)).toBe('$0.0010/1M');
  });
});

describe('extractModelDate', () => {
  it('extracts YYYYMMDD format', () => {
    const date = extractModelDate('gpt-4-20231215');
    expect(date?.getFullYear()).toBe(2023);
    expect(date?.getMonth()).toBe(11);
    expect(date?.getDate()).toBe(15);
  });

  it('extracts ISO format YYYY-MM-DD', () => {
    const date = extractModelDate('claude-3-2024-01-15');
    expect(date?.getFullYear()).toBe(2024);
    expect(date?.getMonth()).toBe(0);
    expect(date?.getDate()).toBe(15);
  });

  it('returns null for no date', () => {
    expect(extractModelDate('gpt-4-turbo')).toBeNull();
  });
});

describe('extractModelVersion', () => {
  it('extracts dot versions', () => {
    expect(extractModelVersion('gpt-4.5-turbo')).toBe(4.5);
    expect(extractModelVersion('claude-3.5-sonnet')).toBe(3.5);
  });

  it('extracts o-series versions', () => {
    expect(extractModelVersion('o1-preview')).toBe(1);
    expect(extractModelVersion('o3-mini')).toBe(3);
  });

  it('extracts single digit versions', () => {
    expect(extractModelVersion('gpt-4-turbo')).toBe(4);
    expect(extractModelVersion('claude-3-opus')).toBe(3);
  });

  it('returns 0 for no version', () => {
    expect(extractModelVersion('some-model')).toBe(0);
  });
});

describe('sortModelsByDate', () => {
  it('sorts by version first (descending)', () => {
    const models = [{ model: 'gpt-3.5-turbo' }, { model: 'gpt-4-turbo' }, { model: 'gpt-4.5-preview' }];
    const sorted = sortModelsByDate(models);
    expect(sorted.map((m) => m.model)).toEqual(['gpt-4.5-preview', 'gpt-4-turbo', 'gpt-3.5-turbo']);
  });

  it('sorts by date for same version (newest first)', () => {
    const models = [{ model: 'gpt-4-20231101' }, { model: 'gpt-4-20231215' }, { model: 'gpt-4-20231001' }];
    const sorted = sortModelsByDate(models);
    expect(sorted.map((m) => m.model)).toEqual(['gpt-4-20231215', 'gpt-4-20231101', 'gpt-4-20231001']);
  });

  it('does not mutate original array', () => {
    const models = [{ model: 'b-model' }, { model: 'a-model' }];
    const original = [...models];
    sortModelsByDate(models);
    expect(models).toEqual(original);
  });
});
