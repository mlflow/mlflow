import { describe, expect, test } from '@jest/globals';
import { MLFLOW_ISSUE_DETECTION_PRIORITIES_TAG } from '../../constants';
import { ALL_ISSUE_CATEGORIES } from '../../components/experiment-page/components/traces-v3/IssueDetectionCategories';
import {
  defaultIssueDetectionSettings,
  noFocusAreaSelected,
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
    expect(selectedDetectionCategories(settings)).toEqual(['pii leakage']);
  });

  test('a focus area the saved settings never mentioned is not selected', () => {
    // Only what the tag selects counts, so a built-in it does not mention is off -- including
    // one added to the product after the owner last saved, which is why the settings modal
    // shows it unticked until they save again.
    const settings = readIssueDetectionSettingsFromTags(tag({ instructions: '', categories: { correctness: true } }));

    expect(settings.categories).toEqual({ correctness: true });
    expect(selectedDetectionCategories(settings)).toEqual(['correctness']);
  });

  test('a deselected built-in is kept alongside the selected ones', () => {
    // The shape the settings modal actually writes: every built-in, with the unticked ones false.
    const settings = readIssueDetectionSettingsFromTags(
      tag({
        instructions: '',
        categories: Object.fromEntries(ALL_ISSUE_CATEGORIES.map((category) => [category, category !== 'cost'])),
      }),
    );

    expect(settings.categories['cost']).toBe(false);
    expect(selectedDetectionCategories(settings)).toEqual(ALL_ISSUE_CATEGORIES.filter((id) => id !== 'cost'));
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
    // 'yes' is not a decision either way, so correctness is dropped rather than read as selected.
    expect(settings.categories).toEqual({ latency: true });
    expect(selectedDetectionCategories(settings)).toEqual(['latency']);
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

describe('noFocusAreaSelected', () => {
  const allOff = Object.fromEntries(ALL_ISSUE_CATEGORIES.map((category) => [category, false]));

  test('is true only when nothing at all is selected', () => {
    expect(noFocusAreaSelected(readIssueDetectionSettingsFromTags(tag({ instructions: '', categories: allOff })))).toBe(
      true,
    );
    expect(noFocusAreaSelected(defaultIssueDetectionSettings())).toBe(false);
    expect(
      noFocusAreaSelected(
        readIssueDetectionSettingsFromTags(tag({ instructions: '', categories: { ...allOff, safety: true } })),
      ),
    ).toBe(false);
  });

  test('settings that have not loaded are not an all-deselected state', () => {
    // Otherwise every run would open claiming the experiment had deselected everything.
    expect(noFocusAreaSelected(undefined)).toBe(false);
  });

  test('is exactly the condition under which stored and sent differ', () => {
    // The relationship the helper exists to let callers disclose: whenever this is true the
    // run is given categories the tag does not list, and whenever it is false it is not.
    const cases = [
      { instructions: '', categories: allOff },
      { instructions: '', categories: { ...allOff, safety: true } },
      { instructions: '', categories: { ...allOff, 'pii leakage': true } },
      { instructions: '', categories: {} },
    ];
    for (const stored of cases) {
      const settings = readIssueDetectionSettingsFromTags(tag(stored));
      const storedSelection = Object.entries(settings.categories)
        .filter(([, selected]) => selected)
        .map(([category]) => category);
      const sent = selectedDetectionCategories(settings);
      expect(noFocusAreaSelected(settings)).toBe(sent.join() !== storedSelection.join());
    }
  });
});
