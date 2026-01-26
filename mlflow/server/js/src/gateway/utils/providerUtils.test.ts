import { describe, it, expect } from '@jest/globals';
import {
  formatProviderName,
  formatAuthMethodName,
  formatCredentialFieldName,
  sortFieldsByProvider,
} from './providerUtils';

describe('providerUtils', () => {
  describe('formatProviderName', () => {
    it('returns display name for known providers', () => {
      expect(formatProviderName('openai')).toBe('OpenAI');
      expect(formatProviderName('anthropic')).toBe('Anthropic');
      expect(formatProviderName('bedrock')).toBe('Amazon Bedrock');
      expect(formatProviderName('gemini')).toBe('Google Gemini');
      expect(formatProviderName('vertex_ai')).toBe('Google Vertex AI');
      expect(formatProviderName('azure')).toBe('Azure OpenAI');
      expect(formatProviderName('databricks')).toBe('Databricks');
    });

    it('formats unknown providers with title case', () => {
      expect(formatProviderName('some_unknown_provider')).toBe('Some Unknown Provider');
    });
  });

  describe('formatAuthMethodName', () => {
    it('returns display name for known auth methods', () => {
      expect(formatAuthMethodName('api_key')).toBe('API Key');
      expect(formatAuthMethodName('access_key')).toBe('Access Key');
      expect(formatAuthMethodName('sts')).toBe('STS (Assume Role)');
      expect(formatAuthMethodName('service_account')).toBe('Service Account');
    });

    it('formats unknown auth methods with title case', () => {
      expect(formatAuthMethodName('some_new_auth')).toBe('Some New Auth');
    });
  });

  describe('formatCredentialFieldName', () => {
    it('returns display name for known credential fields', () => {
      expect(formatCredentialFieldName('api_key')).toBe('API Key');
      expect(formatCredentialFieldName('aws_access_key_id')).toBe('AWS Access Key ID');
      expect(formatCredentialFieldName('aws_secret_access_key')).toBe('AWS Secret Access Key');
      expect(formatCredentialFieldName('vertex_project')).toBe('Project ID');
    });

    it('formats unknown credential fields with title case', () => {
      expect(formatCredentialFieldName('some_new_field')).toBe('Some New Field');
    });
  });

  describe('sortFieldsByProvider', () => {
    it('sorts Databricks fields in the defined order', () => {
      const fields = [
        { name: 'api_base', value: '' },
        { name: 'client_id', value: '' },
        { name: 'client_secret', value: '' },
      ];
      const sorted = sortFieldsByProvider(fields, 'databricks');
      expect(sorted.map((f) => f.name)).toEqual(['client_id', 'client_secret', 'api_base']);
    });

    it('returns fields unchanged for providers without custom ordering', () => {
      const fields = [
        { name: 'api_key', value: '' },
        { name: 'api_base', value: '' },
      ];
      const sorted = sortFieldsByProvider(fields, 'openai');
      expect(sorted.map((f) => f.name)).toEqual(['api_key', 'api_base']);
    });

    it('puts unknown fields after known fields for Databricks', () => {
      const fields = [
        { name: 'unknown_field', value: '' },
        { name: 'client_id', value: '' },
        { name: 'api_base', value: '' },
      ];
      const sorted = sortFieldsByProvider(fields, 'databricks');
      expect(sorted.map((f) => f.name)).toEqual(['client_id', 'api_base', 'unknown_field']);
    });

    it('handles empty fields array', () => {
      const sorted = sortFieldsByProvider([], 'databricks');
      expect(sorted).toEqual([]);
    });

    it('preserves additional properties on field objects', () => {
      const fields = [
        { name: 'client_secret', value: 'secret', extra: true },
        { name: 'client_id', value: 'id', extra: false },
      ];
      const sorted = sortFieldsByProvider(fields, 'databricks');
      expect(sorted).toEqual([
        { name: 'client_id', value: 'id', extra: false },
        { name: 'client_secret', value: 'secret', extra: true },
      ]);
    });
  });
});
