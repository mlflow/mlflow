import { describe, expect, test } from '@jest/globals';
import { MLFLOW_ISSUE_DETECTION_PRIORITIES_TAG } from '../../constants';
import { ALL_ISSUE_CATEGORIES } from '../../components/experiment-page/components/traces-v3/IssueDetectionCategories';
import {
  defaultIssueDetectionSettings,
  readIssueDetectionSettingsFromTags,
  selectedDetectionCategories,
} from './issueDetectionSettings';

const tag = (value: unknown) => [
  { key: MLFLOW_ISSUE_DETECTION_PRIORITIES_TAG, value: typeof value === 'string' ? value : JSON.stringify(value) },
];

describe('readIssueDetectionSettingsFromTags', () => {
  test('an experiment with no settings detects everything', () => {
    for (const tags of [undefined, null, [], [{ key: 'other.tag', value: 'x' }]]) {
      const settings = readIssueDetectionSettingsFromTags(tags);
      expect(settings.instructions).toBe('');
      expect(selectedDetectionCategories(settings)).toEqual([...ALL_ISSUE_CATEGORIES]);
    }
  });

  test('saved settings round trip', () => {
    const settings = readIssueDetectionSettingsFromTags(
      tag({ instructions: 'Only report high severity.', categories: { correctness: true, cost: false } }),
    );

    expect(settings.instructions).toBe('Only report high severity.');
    expect(settings.categories['correctness']).toBe(true);
    expect(settings.categories['cost']).toBe(false);
  });

  test('a custom focus area is preserved', () => {
    const settings = readIssueDetectionSettingsFromTags(tag({ instructions: '', categories: { 'pii leakage': true } }));

    expect(settings.categories['pii leakage']).toBe(true);
    expect(selectedDetectionCategories(settings)).toContain('pii leakage');
  });

  test('a built-in the saved settings never mentioned stays selected', () => {
    // Otherwise a category added to the product after someone last saved would be silently
    // off for them, which is not a decision they made.
    const settings = readIssueDetectionSettingsFromTags(tag({ instructions: '', categories: { cost: false } }));

    expect(settings.categories['cost']).toBe(false);
    for (const category of ALL_ISSUE_CATEGORIES.filter((id) => id !== 'cost')) {
      expect(settings.categories[category]).toBe(true);
    }
  });

  test('turning everything off is still expressible', () => {
    const allOff = Object.fromEntries(ALL_ISSUE_CATEGORIES.map((category) => [category, false]));

    const settings = readIssueDetectionSettingsFromTags(tag({ instructions: '', categories: allOff }));

    expect(Object.values(settings.categories).some(Boolean)).toBe(false);
  });

  test.each([
    ['not json at all', 'definitely-not-json'],
    ['a JSON array', '[1, 2, 3]'],
    ['a JSON scalar', '"just a string"'],
  ])('%s falls back to the defaults rather than throwing', (_label, raw) => {
    expect(readIssueDetectionSettingsFromTags(tag(raw))).toEqual(defaultIssueDetectionSettings());
  });

  test('fields of the wrong type are ignored, not coerced', () => {
    const settings = readIssueDetectionSettingsFromTags(
      tag({ instructions: 42, categories: { correctness: 'yes', latency: true, '  ': true } }),
    );

    expect(settings.instructions).toBe('');
    // 'yes' is not a decision, so correctness keeps its default; the blank key is dropped.
    expect(settings.categories['correctness']).toBe(true);
    expect(settings.categories['latency']).toBe(true);
    expect(Object.keys(settings.categories)).not.toContain('  ');
  });
});

describe('selectedDetectionCategories', () => {
  test('only the selected areas are sent', () => {
    const settings = readIssueDetectionSettingsFromTags(
      tag({ instructions: '', categories: Object.fromEntries(ALL_ISSUE_CATEGORIES.map((c) => [c, c === 'safety'])) }),
    );

    expect(selectedDetectionCategories(settings)).toEqual(['safety']);
  });

  test('deselecting everything falls back to all of them rather than asking for a run that can report nothing', () => {
    const allOff = Object.fromEntries(ALL_ISSUE_CATEGORIES.map((category) => [category, false]));
    const settings = readIssueDetectionSettingsFromTags(tag({ instructions: '', categories: allOff }));

    expect(selectedDetectionCategories(settings)).toEqual([...ALL_ISSUE_CATEGORIES]);
  });

  test('settings that have not loaded yet still produce a usable run', () => {
    expect(selectedDetectionCategories(undefined)).toEqual([...ALL_ISSUE_CATEGORIES]);
  });
});
