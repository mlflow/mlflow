import { RUNS_VISIBILITY_MODE } from '../models/ExperimentPageUIState';
import { determineIfRowIsHidden } from './experimentPage.common-row-utils';

describe('determineIfRowIsHidden', () => {
  it.each([
    { mode: RUNS_VISIBILITY_MODE.FIRST_10_RUNS, rowIndex: 5, expected: false },
    { mode: RUNS_VISIBILITY_MODE.FIRST_20_RUNS, rowIndex: 5, expected: false },
    { mode: RUNS_VISIBILITY_MODE.FIRST_10_RUNS, rowIndex: 15, expected: true },
    { mode: RUNS_VISIBILITY_MODE.FIRST_20_RUNS, rowIndex: 15, expected: false },
    { mode: RUNS_VISIBILITY_MODE.FIRST_10_RUNS, rowIndex: 25, expected: true },
    { mode: RUNS_VISIBILITY_MODE.FIRST_20_RUNS, rowIndex: 25, expected: true },
    { mode: RUNS_VISIBILITY_MODE.HIDEALL, rowIndex: 5, expected: true },
    { mode: RUNS_VISIBILITY_MODE.CUSTOM, rowIndex: 5, expected: true },
  ])(
    'should return $expected if runs visibility mode is $mode and runsHidden includes runUuid having index $rowIndex',
    ({ expected, mode, rowIndex }) => {
      const runsHidden = ['run1', 'run2'];
      const runUuid = 'run1';
      const result = determineIfRowIsHidden(mode, runsHidden, runUuid, rowIndex);

      expect(result).toBe(expected);
    },
  );

  it.each([
    { mode: RUNS_VISIBILITY_MODE.HIDEALL, rowIndex: 5, expected: true },
    { mode: RUNS_VISIBILITY_MODE.CUSTOM, rowIndex: 5, expected: false },
  ])(
    'should return $expected if runs visibility mode is $mode and runsHidden does not include runUuid having index $rowIndex',
    ({ expected, mode, rowIndex }) => {
      const runsHidden = ['run1', 'run2'];
      const runUuid = 'run3';
      const result = determineIfRowIsHidden(mode, runsHidden, runUuid, rowIndex);

      expect(result).toBe(expected);
    },
  );
});
