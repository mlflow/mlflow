import { describe, test, expect } from '@jest/globals';
import {
  isUniqueConstraintError,
  isEndpointNameError,
  isSecretNameError,
  isSecretNameConflict,
  getReadableErrorMessage,
} from './errorUtils';

describe('errorUtils', () => {
  describe('isUniqueConstraintError', () => {
    test('detects SQLite unique constraint error', () => {
      expect(isUniqueConstraintError('UNIQUE constraint failed: endpoints.name')).toBe(true);
    });

    test('detects PostgreSQL unique constraint error', () => {
      expect(isUniqueConstraintError('duplicate key value violates unique constraint')).toBe(true);
    });

    test('detects MySQL duplicate entry error', () => {
      expect(isUniqueConstraintError("Duplicate entry 'test' for key 'name'")).toBe(true);
    });

    test('detects generic already exists error', () => {
      expect(isUniqueConstraintError('An endpoint with this name already exists')).toBe(true);
    });

    test('returns false for non-unique errors', () => {
      expect(isUniqueConstraintError('Network timeout')).toBe(false);
      expect(isUniqueConstraintError('Permission denied')).toBe(false);
    });
  });

  describe('isEndpointNameError', () => {
    test('detects endpoint name errors', () => {
      expect(isEndpointNameError('UNIQUE constraint failed: endpoints.name')).toBe(true);
      expect(isEndpointNameError('endpoint_name must be unique')).toBe(true);
    });

    test('returns false for non-endpoint errors', () => {
      expect(isEndpointNameError('UNIQUE constraint failed: secrets.secret_name')).toBe(false);
    });
  });

  describe('isSecretNameError', () => {
    test('detects secret name errors', () => {
      expect(isSecretNameError('UNIQUE constraint failed: secrets.secret_name')).toBe(true);
      expect(isSecretNameError('secret_name must be unique')).toBe(true);
      expect(isSecretNameError('A secret with name "test" already exists')).toBe(true);
    });

    test('returns false for non-secret errors', () => {
      expect(isSecretNameError('UNIQUE constraint failed: endpoints.name')).toBe(false);
    });
  });

  describe('isSecretNameConflict', () => {
    test('returns true for secret name unique constraint errors', () => {
      const error = new Error('UNIQUE constraint failed: secrets.secret_name');
      expect(isSecretNameConflict(error)).toBe(true);
    });

    test('returns false for other errors', () => {
      const error = new Error('Network error');
      expect(isSecretNameConflict(error)).toBe(false);
    });

    test('handles null/undefined', () => {
      expect(isSecretNameConflict(null)).toBe(false);
      expect(isSecretNameConflict(undefined)).toBe(false);
    });
  });

  describe('getReadableErrorMessage', () => {
    test('returns null for null error', () => {
      expect(getReadableErrorMessage(null)).toBe(null);
    });

    test('returns user-friendly message for endpoint name conflict', () => {
      const error = new Error('UNIQUE constraint failed: endpoints.name');
      expect(getReadableErrorMessage(error)).toBe(
        'An endpoint with this name already exists. Please choose a different name.',
      );
    });

    test('returns user-friendly message for secret name conflict', () => {
      const error = new Error('UNIQUE constraint failed: secrets.secret_name');
      expect(getReadableErrorMessage(error, { entityType: 'secret' })).toBe(
        'A secret with this name already exists. Please choose a different name or use an existing secret.',
      );
    });

    test('returns generic message for long error messages', () => {
      const longMessage = 'A'.repeat(250);
      const error = new Error(longMessage);
      expect(getReadableErrorMessage(error, { entityType: 'endpoint', action: 'creating' })).toBe(
        'An error occurred while creating the endpoint. Please try again.',
      );
    });

    test('returns original message for short non-matching errors', () => {
      const error = new Error('Invalid provider');
      expect(getReadableErrorMessage(error)).toBe('Invalid provider');
    });
  });
});
