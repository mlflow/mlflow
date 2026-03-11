import { describe, it, expect } from '@jest/globals';

import { rankByKeyImportance, PREFERRED_INPUT_KEYS, PREFERRED_OUTPUT_KEYS } from './SingleChatTurnMessages';

const rankInputByImportance = rankByKeyImportance(PREFERRED_INPUT_KEYS);
const rankOutputByImportance = rankByKeyImportance(PREFERRED_OUTPUT_KEYS);

const keys = (items: { key: string }[]) => items.map((item) => item.key);
const toItems = (...itemKeys: string[]) => itemKeys.map((key) => ({ key, value: 'v' }));

describe('rankByKeyImportance', () => {
  describe('rankInputByImportance', () => {
    it('should prioritize "messages" above other keys', () => {
      const items = toItems('config', 'messages', 'temperature');
      expect(keys(items.sort(rankInputByImportance))).toEqual(['messages', 'config', 'temperature']);
    });

    it('should prioritize "input" above non-preferred keys', () => {
      const items = toItems('temperature', 'input', 'config');
      expect(keys(items.sort(rankInputByImportance))).toEqual(['input', 'temperature', 'config']);
    });

    it('should prioritize "inputs" above non-preferred keys', () => {
      const items = toItems('config', 'inputs');
      expect(keys(items.sort(rankInputByImportance))).toEqual(['inputs', 'config']);
    });

    it('should rank preferred keys in priority order: messages > input > inputs', () => {
      const items = toItems('inputs', 'input', 'messages');
      expect(keys(items.sort(rankInputByImportance))).toEqual(['messages', 'input', 'inputs']);
    });

    it('should be case-insensitive', () => {
      const items = toItems('config', 'Messages', 'INPUT');
      expect(keys(items.sort(rankInputByImportance))).toEqual(['Messages', 'INPUT', 'config']);
    });

    it('should preserve original order for non-preferred keys', () => {
      const items = toItems('alpha', 'beta', 'gamma');
      expect(keys(items.sort(rankInputByImportance))).toEqual(['alpha', 'beta', 'gamma']);
    });

    it('should handle a single preferred key among many non-preferred', () => {
      const items = toItems('a', 'b', 'messages', 'c', 'd');
      expect(keys(items.sort(rankInputByImportance))).toEqual(['messages', 'a', 'b', 'c', 'd']);
    });

    it('should handle an empty list', () => {
      expect(toItems().sort(rankInputByImportance)).toEqual([]);
    });

    it('should handle a single item', () => {
      expect(keys(toItems('messages').sort(rankInputByImportance))).toEqual(['messages']);
    });
  });

  describe('rankOutputByImportance', () => {
    it('should prioritize "response" above other keys', () => {
      const items = toItems('metadata', 'response', 'usage');
      expect(keys(items.sort(rankOutputByImportance))).toEqual(['response', 'metadata', 'usage']);
    });

    it('should prioritize "output" above non-preferred keys', () => {
      const items = toItems('usage', 'output');
      expect(keys(items.sort(rankOutputByImportance))).toEqual(['output', 'usage']);
    });

    it('should prioritize "outputs" above non-preferred keys', () => {
      const items = toItems('metadata', 'outputs', 'usage');
      expect(keys(items.sort(rankOutputByImportance))).toEqual(['outputs', 'metadata', 'usage']);
    });

    it('should prioritize "generations" above non-preferred keys', () => {
      const items = toItems('metadata', 'generations');
      expect(keys(items.sort(rankOutputByImportance))).toEqual(['generations', 'metadata']);
    });

    it('should rank preferred keys in priority order: response > output > outputs > generations', () => {
      const items = toItems('generations', 'outputs', 'output', 'response');
      expect(keys(items.sort(rankOutputByImportance))).toEqual(['response', 'output', 'outputs', 'generations']);
    });

    it('should be case-insensitive', () => {
      const items = toItems('usage', 'Response', 'GENERATIONS');
      expect(keys(items.sort(rankOutputByImportance))).toEqual(['Response', 'GENERATIONS', 'usage']);
    });
  });

  describe('custom preferred keys', () => {
    it('should work with arbitrary preferred key lists', () => {
      const rankByCustom = rankByKeyImportance(['z', 'a']);
      const items = toItems('b', 'a', 'z', 'c');
      expect(keys(items.sort(rankByCustom))).toEqual(['z', 'a', 'b', 'c']);
    });
  });
});
