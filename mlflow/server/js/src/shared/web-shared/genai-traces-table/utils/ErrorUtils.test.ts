import { describe, it, expect } from '@jest/globals';
import { isSqlWarehouseTimeoutError } from './ErrorUtils';

describe('isSqlWarehouseTimeoutError', () => {
  it('returns true for "Timeout while getting statement result"', () => {
    const error = new Error('SQL query failed: Timeout while getting statement result');
    expect(isSqlWarehouseTimeoutError(error)).toBe(true);
  });

  it('returns true for "Timeout while issuing SQL query"', () => {
    const error = new Error('SQL query failed: Timeout while issuing SQL query');
    expect(isSqlWarehouseTimeoutError(error)).toBe(true);
  });

  it('returns true for "Timeout while waiting for SQL query to complete"', () => {
    const error = new Error('SQL query failed: Timeout while waiting for SQL query to complete after 50s');
    expect(isSqlWarehouseTimeoutError(error)).toBe(true);
  });

  it('returns true for "underlying SQL request timed out"', () => {
    const error = new Error('The underlying SQL request timed out after 60 seconds');
    expect(isSqlWarehouseTimeoutError(error)).toBe(true);
  });

  it('returns false for unrelated errors', () => {
    const error = new Error('Network request failed');
    expect(isSqlWarehouseTimeoutError(error)).toBe(false);
  });

  it('returns false for generic SQL errors without timeout', () => {
    const error = new Error('SQL query failed: syntax error');
    expect(isSqlWarehouseTimeoutError(error)).toBe(false);
  });

  it('returns false for null', () => {
    expect(isSqlWarehouseTimeoutError(null)).toBe(false);
  });

  it('returns false for undefined', () => {
    expect(isSqlWarehouseTimeoutError(undefined)).toBe(false);
  });

  it('returns false for error with empty message', () => {
    const error = new Error('');
    expect(isSqlWarehouseTimeoutError(error)).toBe(false);
  });
});
