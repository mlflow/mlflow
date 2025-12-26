import { describe, it, expect } from '@jest/globals';
import { formatProviderName, formatAuthMethodName, formatCredentialFieldName } from './providerUtils';

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

    it('returns display name for Vertex AI variants', () => {
      expect(formatProviderName('vertex_ai-anthropic')).toBe('Vertex AI (Anthropic)');
      expect(formatProviderName('vertex_ai-llama3')).toBe('Vertex AI (Llama 3)');
      expect(formatProviderName('vertex_ai-mistral')).toBe('Vertex AI (Mistral)');
    });

    it('formats unknown Vertex AI variants', () => {
      expect(formatProviderName('vertex_ai-some-new-model')).toBe('Vertex AI (Some New Model)');
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
});
