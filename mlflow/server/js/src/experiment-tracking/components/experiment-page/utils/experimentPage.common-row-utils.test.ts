import { fromPairs } from 'lodash';
import { shouldUseRunRowsVisibilityMap } from '../../../../common/utils/FeatureUtils';
import { RUNS_VISIBILITY_MODE } from '../models/ExperimentPageUIState';
import { determineIfRowIsHidden } from './experimentPage.common-row-utils';

jest.mock('../../../../common/utils/FeatureUtils', () => ({
  shouldUseRunRowsVisibilityMap: jest.fn(),
}));

describe('determineIfRowIsHidden when using legacy runsHidden UI state', () => {
  beforeEach(() => {
    jest.mocked(shouldUseRunRowsVisibilityMap).mockReturnValue(false);
  });
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
      const result = determineIfRowIsHidden(mode, runsHidden, runUuid, rowIndex, {}, 'RUNNING');

      expect(result).toBe(expected);
    },
  );

  it('hides finished runs when mode is HIDE_FINISHED_RUNS', () => {
    const runsHidden: string[] = [];
    const runUuid = 'run1';
    expect(
      determineIfRowIsHidden(RUNS_VISIBILITY_MODE.HIDE_FINISHED_RUNS, runsHidden, runUuid, 5, {}, 'FINISHED'),
    ).toBe(true);
    expect(determineIfRowIsHidden(RUNS_VISIBILITY_MODE.HIDE_FINISHED_RUNS, runsHidden, runUuid, 5, {}, 'RUNNING')).toBe(
      false,
    );
  });

  it.each([
    { mode: RUNS_VISIBILITY_MODE.HIDEALL, rowIndex: 5, expected: true },
    { mode: RUNS_VISIBILITY_MODE.CUSTOM, rowIndex: 5, expected: false },
  ])(
    'should return $expected if runs visibility mode is $mode and runsHidden does not include runUuid having index $rowIndex',
    ({ expected, mode, rowIndex }) => {
      const runsHidden = ['run1', 'run2'];
      const runUuid = 'run3';
      const result = determineIfRowIsHidden(mode, runsHidden, runUuid, rowIndex, {}, 'RUNNING');

      expect(result).toBe(expected);
    },
  );
});

describe('determineIfRowIsHidden when using runsVisibilityMap UI state', () => {
  // Setup:
  // - runs "run1" and "run2" are hidden in the visibility map
  // - code should use runs visibility map instead of "runsHidden" field
  // - we're testing visibility of "run1" and "run3" at various row indexes

  const runsVisibilityMap = { run1: false, run2: false };

  beforeEach(() => {
    jest.mocked(shouldUseRunRowsVisibilityMap).mockReturnValue(true);
  });
  it.each([
    { mode: RUNS_VISIBILITY_MODE.FIRST_10_RUNS, runUuid: 'run1', rowIndex: 5, expected: true },
    { mode: RUNS_VISIBILITY_MODE.FIRST_10_RUNS, runUuid: 'run3', rowIndex: 5, expected: false },
    { mode: RUNS_VISIBILITY_MODE.FIRST_10_RUNS, runUuid: 'run3', rowIndex: 15, expected: true },
    { mode: RUNS_VISIBILITY_MODE.FIRST_20_RUNS, runUuid: 'run1', rowIndex: 5, expected: true },
    { mode: RUNS_VISIBILITY_MODE.FIRST_20_RUNS, runUuid: 'run3', rowIndex: 5, expected: false },
    { mode: RUNS_VISIBILITY_MODE.FIRST_10_RUNS, runUuid: 'run1', rowIndex: 15, expected: true },
    { mode: RUNS_VISIBILITY_MODE.FIRST_20_RUNS, runUuid: 'run1', rowIndex: 15, expected: true },
    { mode: RUNS_VISIBILITY_MODE.FIRST_10_RUNS, runUuid: 'run1', rowIndex: 25, expected: true },
    { mode: RUNS_VISIBILITY_MODE.FIRST_20_RUNS, runUuid: 'run1', rowIndex: 25, expected: true },
    { mode: RUNS_VISIBILITY_MODE.HIDEALL, runUuid: 'run1', rowIndex: 5, expected: true },
    { mode: RUNS_VISIBILITY_MODE.HIDEALL, runUuid: 'run3', rowIndex: 5, expected: true },
    { mode: RUNS_VISIBILITY_MODE.SHOWALL, runUuid: 'run1', rowIndex: 5, expected: true },
    { mode: RUNS_VISIBILITY_MODE.SHOWALL, runUuid: 'run3', rowIndex: 5, expected: false },
  ])(
    'should return $expected if runs visibility mode is $mode and runsHidden includes runUuid having index $rowIndex',
    ({ expected, runUuid, mode, rowIndex }) => {
      const result = determineIfRowIsHidden(mode, [], runUuid, rowIndex, runsVisibilityMap, 'RUNNING');

      expect(result).toBe(expected);
    },
  );

  it('hides finished runs when mode is HIDE_FINISHED_RUNS using visibility map', () => {
    const runUuid = 'run3';
    expect(
      determineIfRowIsHidden(RUNS_VISIBILITY_MODE.HIDE_FINISHED_RUNS, [], runUuid, 5, runsVisibilityMap, 'FINISHED'),
    ).toBe(true);
    expect(
      determineIfRowIsHidden(RUNS_VISIBILITY_MODE.HIDE_FINISHED_RUNS, [], runUuid, 5, runsVisibilityMap, 'FAILED'),
    ).toBe(true);
    expect(
      determineIfRowIsHidden(RUNS_VISIBILITY_MODE.HIDE_FINISHED_RUNS, [], runUuid, 5, runsVisibilityMap, 'RUNNING'),
    ).toBe(false);
  });
});
