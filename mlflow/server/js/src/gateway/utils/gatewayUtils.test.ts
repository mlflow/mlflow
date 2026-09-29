import { describe, it, expect } from '@jest/globals';
import type { Endpoint } from '../types';
import {
  endpointHasMixedTypeSafeProviders,
  endpointUsesAnyProvider,
  generateCopyName,
  hasMixedTypeSafeProviders,
} from './gatewayUtils';

const endpointWithProviders = (providers?: string[]): Endpoint =>
  ({
    model_mappings: providers?.map((provider) => ({ model_definition: { provider } })),
  }) as Endpoint;

describe('gatewayUtils', () => {
  describe('hasMixedTypeSafeProviders', () => {
    it.each<[string[], boolean]>([
      [[], false],
      [['typesafe'], false],
      [['typesafe', 'typesafe'], false],
      [['typesafe', ''], false],
      [['openai', 'anthropic'], false],
      [['typesafe', 'openai'], true],
      [['anthropic', 'typesafe'], true],
    ])('checks provider compatibility for %j', (providers, expected) => {
      expect(hasMixedTypeSafeProviders(providers.map((provider) => ({ provider })))).toBe(expected);
    });
  });

  describe('endpointUsesAnyProvider', () => {
    it.each<[string[] | undefined, boolean]>([
      [undefined, false],
      [[], false],
      [['openai'], false],
      [['typesafe'], true],
      [['typesafe', 'openai'], true],
    ])('checks endpoint mappings for %j', (providers, expected) => {
      expect(endpointUsesAnyProvider(endpointWithProviders(providers), ['typesafe'])).toBe(expected);
    });
  });

  describe('endpointHasMixedTypeSafeProviders', () => {
    it.each<[string[] | undefined, boolean]>([
      [undefined, false],
      [[], false],
      [['openai'], false],
      [['typesafe'], false],
      [['typesafe', 'openai'], true],
    ])('checks endpoint mappings for %j', (providers, expected) => {
      expect(endpointHasMixedTypeSafeProviders(endpointWithProviders(providers))).toBe(expected);
    });
  });

  describe('generateCopyName', () => {
    it('generates a basic copy name', () => {
      expect(generateCopyName('my-endpoint', [])).toBe('my-endpoint-copy-1');
    });

    it('avoids conflicts with existing names', () => {
      expect(generateCopyName('my-endpoint', ['my-endpoint-copy-1'])).toBe('my-endpoint-copy-2');
    });

    it('skips multiple conflicting names', () => {
      const existing = ['my-endpoint-copy-1', 'my-endpoint-copy-2', 'my-endpoint-copy-3'];
      expect(generateCopyName('my-endpoint', existing)).toBe('my-endpoint-copy-4');
    });

    it('does not conflict with unrelated names', () => {
      expect(generateCopyName('my-endpoint', ['other-endpoint', 'another-copy-1'])).toBe('my-endpoint-copy-1');
    });

    it('handles the original name being in the list', () => {
      expect(generateCopyName('my-endpoint', ['my-endpoint'])).toBe('my-endpoint-copy-1');
    });
  });
});
