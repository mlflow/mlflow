import { describe, it, expect } from '@jest/globals';
import { getReadableErrorMessage, isSecretNameConflict } from './errorUtils';

describe('errorUtils', () => {
  describe('getReadableErrorMessage', () => {
    it('returns null for null error', () => {
      expect(getReadableErrorMessage(null)).toBeNull();
    });

    it('returns null for error with no message', () => {
      expect(getReadableErrorMessage({} as Error)).toBeNull();
    });

    it('detects endpoint name unique constraint error', () => {
      const error = new Error('UNIQUE constraint failed: endpoints.name');
      expect(getReadableErrorMessage(error)).toBe(
        'An endpoint with this name already exists. Please choose a different name.',
      );
    });

    it('detects secret name unique constraint error for secret entity', () => {
      const error = new Error('UNIQUE constraint failed: secrets.secret_name');
      expect(getReadableErrorMessage(error, { entityType: 'secret' })).toBe(
        'A secret with this name already exists. Please choose a different name or use an existing secret.',
      );
    });

    it('detects secret name unique constraint error for non-secret entity', () => {
      const error = new Error('UNIQUE constraint failed: secrets.secret_name');
      expect(getReadableErrorMessage(error)).toBe(
        'An API key with this name already exists. Please choose a different name or use an existing API key.',
      );
    });

    it('detects generic unique constraint error', () => {
      const error = new Error('duplicate key value violates unique constraint');
      expect(getReadableErrorMessage(error)).toBe(
        'A record with this value already exists. Please use a unique value.',
      );
    });

    it('truncates long error messages', () => {
      const longMessage = 'x'.repeat(250);
      const error = new Error(longMessage);
      expect(getReadableErrorMessage(error, { action: 'creating', entityType: 'endpoint' })).toBe(
        'An error occurred while creating the endpoint. Please try again.',
      );
    });

    it('returns original message for short errors', () => {
      const error = new Error('Some short error');
      expect(getReadableErrorMessage(error)).toBe('Some short error');
    });
  });

  describe('isSecretNameConflict', () => {
    it('returns true for secret name unique constraint error', () => {
      const error = new Error('UNIQUE constraint failed: secrets.secret_name');
      expect(isSecretNameConflict(error)).toBe(true);
    });

    it('returns false for endpoint name error', () => {
      const error = new Error('UNIQUE constraint failed: endpoints.name');
      expect(isSecretNameConflict(error)).toBe(false);
    });

    it('returns false for non-unique-constraint error', () => {
      const error = new Error('Some random error');
      expect(isSecretNameConflict(error)).toBe(false);
    });

    it('returns false for null/undefined', () => {
      expect(isSecretNameConflict(null)).toBe(false);
      expect(isSecretNameConflict(undefined)).toBe(false);
    });
  });
});
