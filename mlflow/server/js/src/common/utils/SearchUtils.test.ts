import { describe, it, expect } from '@jest/globals';
import { buildSearchFilterClause } from './SearchUtils';

describe('buildSearchFilterClause', () => {
  it('returns undefined for undefined input', () => {
    expect(buildSearchFilterClause(undefined)).toBeUndefined();
  });

  it('returns undefined for empty string', () => {
    expect(buildSearchFilterClause('')).toBeUndefined();
  });

  it('wraps plain text in ILIKE pattern', () => {
    expect(buildSearchFilterClause('my-server')).toBe("name ILIKE '%my-server%'");
  });

  it('escapes single quotes', () => {
    expect(buildSearchFilterClause("it's")).toBe("name ILIKE '%it''s%'");
  });

  it('escapes percent signs', () => {
    expect(buildSearchFilterClause('100%')).toBe("name ILIKE '%100\\%%'");
  });

  it('escapes underscores', () => {
    expect(buildSearchFilterClause('my_server')).toBe("name ILIKE '%my\\_server%'");
  });

  it('passes through ILIKE filter syntax', () => {
    expect(buildSearchFilterClause('name ILIKE "%test%"')).toBe('name ILIKE "%test%"');
  });

  it('passes through LIKE filter syntax', () => {
    expect(buildSearchFilterClause('name LIKE "test%"')).toBe('name LIKE "test%"');
  });

  it('passes through equals filter syntax', () => {
    expect(buildSearchFilterClause('tags.env = "prod"')).toBe('tags.env = "prod"');
  });

  it('passes through not-equals filter syntax', () => {
    expect(buildSearchFilterClause('tags.status != "archived"')).toBe('tags.status != "archived"');
  });

  it('passes through comparison operators', () => {
    expect(buildSearchFilterClause('count > 10')).toBe('count > 10');
    expect(buildSearchFilterClause('count < 10')).toBe('count < 10');
    expect(buildSearchFilterClause('count >= 10')).toBe('count >= 10');
    expect(buildSearchFilterClause('count <= 10')).toBe('count <= 10');
  });

  it('passes through IN filter syntax', () => {
    expect(buildSearchFilterClause('status IN ("active", "paused")')).toBe('status IN ("active", "paused")');
  });

  it('is case-insensitive for SQL keywords', () => {
    expect(buildSearchFilterClause('name ilike "%test%"')).toBe('name ilike "%test%"');
  });

  it('requires whitespace before SQL keywords to avoid false positives', () => {
    expect(buildSearchFilterClause('prompt-ILIKE-test')).toBe("name ILIKE '%prompt-ILIKE-test%'");
    expect(buildSearchFilterClause('prompt-LIKE-test')).toBe("name ILIKE '%prompt-LIKE-test%'");
  });
});
