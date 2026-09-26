import { describe, expect, test } from '@jest/globals';
import {
  EMPTY_FILTER_MODEL,
  FilterOp,
  countActiveFilters,
  isClauseComplete,
  makeEmptyClause,
  type FilterClause,
  type FilterFieldDef,
} from './filterModel';

const FIELDS: FilterFieldDef[] = [
  {
    id: 'state',
    label: 'State',
    operators: [FilterOp.EQUALS],
    valueInput: 'select',
    options: [{ value: 'OK', label: 'OK' }],
  },
  {
    id: 'duration',
    label: 'Duration',
    operators: [FilterOp.GREATER_THAN, FilterOp.LESS_THAN],
    valueInput: 'number',
  },
];

const KEY_REQUIRING_FIELDS: FilterFieldDef[] = [
  { id: 'tag', label: 'Tag', operators: [FilterOp.EQUALS], valueInput: 'text', requiresKey: true },
];

describe('filterModel', () => {
  test('the empty model has no active filters', () => {
    expect(countActiveFilters(EMPTY_FILTER_MODEL)).toBe(0);
  });

  describe('isClauseComplete', () => {
    test('a clause with a field, operator, and non-blank value is complete', () => {
      expect(isClauseComplete({ field: 'state', operator: FilterOp.EQUALS, value: 'OK' })).toBe(true);
    });

    test('a blank or whitespace-only value is incomplete', () => {
      expect(isClauseComplete({ field: 'state', operator: FilterOp.EQUALS, value: '' })).toBe(false);
      expect(isClauseComplete({ field: 'state', operator: FilterOp.EQUALS, value: '   ' })).toBe(false);
    });

    test('a missing field is incomplete', () => {
      expect(isClauseComplete({ field: '', operator: FilterOp.EQUALS, value: 'OK' })).toBe(false);
    });

    test('a clause with a present but blank key is incomplete', () => {
      expect(isClauseComplete({ field: 'tag', operator: FilterOp.EQUALS, value: 'prod', key: '' })).toBe(false);
      expect(isClauseComplete({ field: 'tag', operator: FilterOp.EQUALS, value: 'prod', key: '  ' })).toBe(false);
    });

    test('a clause with a non-blank key is complete', () => {
      expect(isClauseComplete({ field: 'tag', operator: FilterOp.EQUALS, value: 'prod', key: 'env' })).toBe(true);
    });
  });

  describe('countActiveFilters', () => {
    test('counts only complete clauses', () => {
      const model: FilterClause[] = [
        { field: 'state', operator: FilterOp.EQUALS, value: 'OK' },
        { field: 'duration', operator: FilterOp.GREATER_THAN, value: '' },
      ];
      expect(countActiveFilters(model)).toBe(1);
    });
  });

  describe('makeEmptyClause', () => {
    test('seeds from the first field and its default operator', () => {
      expect(makeEmptyClause(FIELDS)).toEqual({ field: 'state', operator: FilterOp.EQUALS, value: '' });
    });

    test('falls back safely when the field list is empty', () => {
      expect(makeEmptyClause([])).toEqual({ field: '', operator: FilterOp.EQUALS, value: '' });
    });

    test('seeds an empty key for a requiresKey field', () => {
      expect(makeEmptyClause(KEY_REQUIRING_FIELDS)).toEqual({
        field: 'tag',
        operator: FilterOp.EQUALS,
        value: '',
        key: '',
      });
    });
  });
});
