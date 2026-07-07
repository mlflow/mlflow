import { describe, expect, it } from '@jest/globals';
import { getStarterCodeProvider } from './EditEndpointFormRenderer';
import type { EndpointModelMapping } from '../../types';

const mapping = (provider: string): EndpointModelMapping =>
  ({
    mapping_id: '1',
    endpoint_id: '1',
    model_definition_id: '1',
    model_definition: { provider },
    weight: 100,
    created_at: 0,
  }) as EndpointModelMapping;

describe('getStarterCodeProvider', () => {
  it('returns the provider when all models share the same provider', () => {
    expect(getStarterCodeProvider([mapping('openai'), mapping('openai')])).toBe('openai');
  });

  it('returns the first provider when openai and azure are mixed', () => {
    expect(getStarterCodeProvider([mapping('openai'), mapping('azure')])).toBe('openai');
    expect(getStarterCodeProvider([mapping('azure'), mapping('openai')])).toBe('azure');
  });

  it('returns undefined when providers differ', () => {
    expect(getStarterCodeProvider([mapping('openai'), mapping('anthropic')])).toBeUndefined();
  });

  it('returns the provider for a single model', () => {
    expect(getStarterCodeProvider([mapping('anthropic')])).toBe('anthropic');
  });

  it('returns undefined for empty mappings', () => {
    expect(getStarterCodeProvider([])).toBeUndefined();
  });

  it('returns undefined when no model_definition exists', () => {
    const m = { mapping_id: '1', endpoint_id: '1', model_definition_id: '1', weight: 100, created_at: 0 };
    expect(getStarterCodeProvider([m as EndpointModelMapping])).toBeUndefined();
  });
});
