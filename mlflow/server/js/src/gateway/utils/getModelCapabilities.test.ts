import { describe, it, expect } from '@jest/globals';
import { getModelCapabilities } from './getModelCapabilities';
import type { ProviderModel } from '../types';

const baseModel: ProviderModel = {
  model: 'test-model',
  provider: 'test-provider',
  supports_function_calling: false,
};

describe('getModelCapabilities', () => {
  it('returns empty array for undefined', () => {
    expect(getModelCapabilities(undefined)).toEqual([]);
  });

  it('returns empty array when no capabilities are supported', () => {
    expect(getModelCapabilities(baseModel)).toEqual([]);
  });

  it('returns all four capabilities when all are supported', () => {
    expect(
      getModelCapabilities({
        ...baseModel,
        supports_function_calling: true,
        supports_reasoning: true,
        supports_prompt_caching: true,
        supports_response_schema: true,
      }),
    ).toEqual(['Tools', 'Reasoning', 'Caching', 'Structured']);
  });

  it('returns only the supported capabilities', () => {
    expect(
      getModelCapabilities({
        ...baseModel,
        supports_function_calling: true,
        supports_response_schema: true,
      }),
    ).toEqual(['Tools', 'Structured']);
  });

  it('preserves order regardless of which capabilities are set', () => {
    expect(
      getModelCapabilities({
        ...baseModel,
        supports_prompt_caching: true,
        supports_reasoning: true,
      }),
    ).toEqual(['Reasoning', 'Caching']);
  });
});
