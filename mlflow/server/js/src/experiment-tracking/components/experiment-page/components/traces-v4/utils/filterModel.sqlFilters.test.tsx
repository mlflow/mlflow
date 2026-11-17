import { describe, expect, test } from '@jest/globals';
import { FilterOp } from '@databricks/web-shared/traces-table';
import { compileFilterModel, compileTagFilters } from './filterModel';

describe('trace predicate compilation', () => {
  test.each(['0 OR 1=1', '1.5', '1e3', 'NaN', '9007199254740992'])(
    'does not compile invalid Basic duration %s',
    (value) => {
      expect(compileFilterModel([{ field: 'duration', operator: FilterOp.GREATER_THAN, value }])).toEqual([]);
    },
  );

  test('does not compile an unknown operator', () => {
    expect(compileFilterModel([{ field: 'state', operator: 'OR 1=1' as FilterOp, value: 'OK' }])).toEqual([]);
  });

  test.each(['k`ey', "x' OR 1=1 --", 'a;DROP', 'x"y', 'line\nbreak'])(
    'does not compile an unsafe dynamic key %s',
    (key) => {
      for (const field of ['tag', 'metadata', 'assessment', 'expectation']) {
        expect(compileFilterModel([{ field, key, operator: FilterOp.EQUALS, value: 'v' }])).toEqual([]);
      }
      expect(compileTagFilters([{ key, value: 'v' }])).toEqual([]);
    },
  );

  test('quotes dynamic keys while preserving their values', () => {
    expect(
      compileFilterModel([{ field: 'metadata', key: 'team owner', operator: FilterOp.EQUALS, value: 'a' }]),
    ).toEqual([`request_metadata."team owner" = 'a'`]);
    expect(compileTagFilters([{ key: 'build-id', value: "O'Reilly" }])).toEqual(["tags.`build-id` = 'O''Reilly'"]);
  });
});
