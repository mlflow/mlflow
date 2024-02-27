import { mount } from 'enzyme';
import { MLFLOW_RUN_TYPE_TAG, MLFLOW_RUN_TYPE_VALUE_EVALUATION } from '../../../constants';
import { useAutoExpandRunRows } from './useAutoExpandRunRows';
import type { RunRowType } from '../utils/experimentPage.row-types';
import type { UpdateExperimentSearchFacetsFn } from '../../../types';
import { SingleRunData } from '../utils/experimentPage.row-utils';

jest.mock('../../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual('../../../../common/utils/FeatureUtils'),
  shouldEnableShareExperimentViewByTags: jest.fn(() => false),
}));

describe('useAutoExpandRunRows', () => {
  const updateSearchFacetsMock = jest.fn();
  const TestComponent = ({
    isPristine = () => true,
    runsExpanded = {},
    updateSearchFacets = updateSearchFacetsMock,
    visibleRunRows = [],
    allRunsData = [],
  }: {
    allRunsData?: SingleRunData[];
    visibleRunRows?: RunRowType[];
    isPristine?: () => boolean;
    updateSearchFacets?: UpdateExperimentSearchFacetsFn;
    runsExpanded?: Record<string, boolean>;
  }) => {
    useAutoExpandRunRows(allRunsData, visibleRunRows, isPristine, updateSearchFacets, runsExpanded);
    return null;
  };

  const getEvaluationTags = () => ({
    [MLFLOW_RUN_TYPE_TAG]: {
      value: MLFLOW_RUN_TYPE_VALUE_EVALUATION,
      key: MLFLOW_RUN_TYPE_TAG,
    },
  });

  const createParentRunRow = (runUuid: string, childrenIds: string[] = []): RunRowType =>
    ({
      runUuid,
      runDateAndNestInfo: { hasExpander: true, isParent: true, childrenIds },
    } as any);

  const createRunData = (runUuid: string, type = ''): SingleRunData =>
    ({
      runInfo: { run_uuid: runUuid },
      tags: {
        [MLFLOW_RUN_TYPE_TAG]: { value: type },
      },
    } as any);

  beforeEach(() => {
    updateSearchFacetsMock.mockClear();
  });

  test('should automatically expand single evaluation run row in the given set', () => {
    mount(
      <TestComponent
        allRunsData={[
          createRunData('child_1'),
          createRunData('child_2', 'evaluation'),
          createRunData('child_3'),
          createRunData('child_4', 'evaluation'),
        ]}
        visibleRunRows={[
          createParentRunRow('parent_1', ['child_1']),
          createParentRunRow('parent_2', ['child_2']),
          createParentRunRow('parent_3', ['child_3']),
          createParentRunRow('parent_4'),
        ]}
      />,
    );

    expect(updateSearchFacetsMock).toBeCalledTimes(1);

    const [stateTransformFn, stateTransformParams] = updateSearchFacetsMock.mock.lastCall;

    expect(stateTransformFn({})).toEqual({
      runsExpanded: {
        parent_2: true,
      },
    });
    expect(stateTransformParams).toEqual({
      preservePristine: true,
      replaceHistory: true,
    });
  });

  test('should skip expanding evaluation run if it was contracted before', () => {
    mount(
      <TestComponent
        allRunsData={[
          createRunData('child_1'),
          createRunData('child_2', 'evaluation'),
          createRunData('child_3'),
          createRunData('child_4', 'evaluation'),
        ]}
        visibleRunRows={[
          createParentRunRow('parent_1', ['child_1']),
          createParentRunRow('parent_2', ['child_2']),
          createParentRunRow('parent_3', ['child_3']),
          createParentRunRow('parent_4'),
        ]}
        runsExpanded={{ parent_2: false }}
      />,
    );

    expect(updateSearchFacetsMock).not.toBeCalled();
  });

  test('should skip expanding evaluation run the state is dirty', () => {
    mount(
      <TestComponent
        allRunsData={[
          createRunData('child_1'),
          createRunData('child_2', 'evaluation'),
          createRunData('child_3'),
          createRunData('child_4', 'evaluation'),
        ]}
        visibleRunRows={[
          createParentRunRow('parent_1', ['child_1']),
          createParentRunRow('parent_2', ['child_2']),
          createParentRunRow('parent_3', ['child_3']),
          createParentRunRow('parent_4'),
        ]}
        isPristine={() => false}
      />,
    );

    expect(updateSearchFacetsMock).not.toBeCalled();
  });
});
