import { describe, expect, test } from '@jest/globals';
import type { DatasetRecord } from '../hooks/useDatasetsQueries';
import { getDefaultRecord, validateSchemaConsistency } from './datasetSchemaUtils';

const makeRecord = (id: string, inputs: Record<string, unknown>): DatasetRecord =>
  ({ dataset_record_id: id, inputs }) as DatasetRecord;

describe('validateSchemaConsistency', () => {
  test('empty record list is a no-op', () => {
    expect(() => validateSchemaConsistency([])).not.toThrow();
  });

  test('all-singleturn (no goal field anywhere) passes', () => {
    const records = [
      makeRecord('r1', { messages: [{ role: 'user', content: 'hi' }] }),
      makeRecord('r2', { messages: [{ role: 'user', content: 'hello' }] }),
    ];
    expect(() => validateSchemaConsistency(records)).not.toThrow();
  });

  test('all-multiturn with only allowed fields passes', () => {
    const records = [
      makeRecord('r1', { goal: 'g1', persona: 'p1', context: 'c1', simulation_guidelines: 'sg1' }),
      makeRecord('r2', { goal: 'g2', persona: 'p2' }),
      makeRecord('r3', { goal: 'g3' }),
    ];
    expect(() => validateSchemaConsistency(records)).not.toThrow();
  });

  test('mixed singleturn + multiturn throws with both counts', () => {
    const records = [
      makeRecord('r1', { goal: 'g1' }),
      makeRecord('r2', { messages: [] }),
      makeRecord('r3', { goal: 'g3' }),
    ];
    expect(() => validateSchemaConsistency(records)).toThrow(/Mixed schemas:.*2 record\(s\).*1 record\(s\)/);
  });

  test('all-multiturn but one record has an invalid input field throws', () => {
    const records = [makeRecord('r1', { goal: 'g1', persona: 'p1' }), makeRecord('r2', { goal: 'g2', messages: [] })];
    expect(() => validateSchemaConsistency(records)).toThrow(/Invalid field\(s\) in multiturn record:.*messages/);
  });

  test('editedRows overlay turns a singleturn record into multiturn (and reclassifies)', () => {
    const records = [makeRecord('r1', { goal: 'g1' }), makeRecord('r2', { messages: [] })];
    // Without overlay this would be "mixed". The overlay converts r2 to multiturn.
    const edited = { r2: { inputs: { goal: 'g2' } } };
    expect(() => validateSchemaConsistency(records, edited)).not.toThrow();
  });

  test('editedRows overlay turns a multiturn record into singleturn (and reclassifies)', () => {
    const records = [makeRecord('r1', { messages: [] }), makeRecord('r2', { goal: 'g2' })];
    // Overlay strips goal from r2, leaving all-singleturn.
    const edited = { r2: { inputs: { messages: [] } } };
    expect(() => validateSchemaConsistency(records, edited)).not.toThrow();
  });

  test('editedRows overlay introducing an invalid multiturn field is caught', () => {
    const records = [makeRecord('r1', { goal: 'g1' }), makeRecord('r2', { goal: 'g2' })];
    const edited = { r2: { inputs: { goal: 'g2', not_allowed: true } } };
    expect(() => validateSchemaConsistency(records, edited)).toThrow(/Invalid field\(s\) in multiturn record/);
  });

  test('record with undefined inputs treated as singleturn', () => {
    const records = [makeRecord('r1', undefined as unknown as Record<string, unknown>)];
    expect(() => validateSchemaConsistency(records)).not.toThrow();
  });
});

describe('getDefaultRecord', () => {
  test('returns singleturn defaults when records are empty', () => {
    const result = getDefaultRecord([]);
    expect(result.inputs).toEqual({ messages: [{ role: 'user', content: 'Hello' }] });
    expect(result.expectations).toEqual({ guidelines: ['The response must be professional'] });
  });

  test('returns singleturn defaults when no record has the goal field', () => {
    const records = [makeRecord('r1', { messages: [] })];
    const result = getDefaultRecord(records);
    expect(result.inputs).toEqual({ messages: [{ role: 'user', content: 'Hello' }] });
  });

  test('returns multiturn defaults when at least one record has the goal field', () => {
    const records = [makeRecord('r1', { goal: 'g1', persona: 'p1' }), makeRecord('r2', { goal: 'g2' })];
    const result = getDefaultRecord(records);
    expect(result.inputs).toEqual({ goal: 'Complete the user request', persona: 'Helpful assistant' });
    expect(result.expectations).toEqual({ guidelines: ['The response must be professional'] });
  });

  test('editedRows overlay can flip the detected schema', () => {
    const records = [makeRecord('r1', { messages: [] })];
    const edited = { r1: { inputs: { goal: 'g1' } } };
    const result = getDefaultRecord(records, edited);
    expect(result.inputs).toEqual({ goal: 'Complete the user request', persona: 'Helpful assistant' });
  });
});
