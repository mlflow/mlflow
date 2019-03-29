import Fixtures from '../test-utils/Fixtures';
import ExperimentViewUtil from "./ExperimentViewUtil";

test('getRowRenderMetadata identifies parent-child run relationships based on run tags', () => {
  const rowRenderMetadata = ExperimentViewUtil.getRowRenderMetadata(
    {
      runInfos: Fixtures.runInfos,
      sortState: {ascending: false, isMetric: false, isParam: false, key: 'start_time'},
      paramsList: Array(Fixtures.runInfos.length).fill([]),
      metricsList: Array(Fixtures.runInfos.length).fill([]),
      tagsList: Fixtures.tagsList,
      runsExpanded: {}
    });
  const parentRunMetadata = {
    idx: 0,
    isParent: true,
    hasExpander: true,
    expanderOpen: false,
    childrenIds: Fixtures.childRunIds,
    runId: Fixtures.sortedRunIds[0],
    sortValue: Fixtures.runInfos[0].start_time,
  };
  expect(rowRenderMetadata).toContainEqual(parentRunMetadata);
});

test('getRowRenderMetadata sorts top-level and child runs', () => {
  const runsExpanded = {};
  Fixtures.topLevelRunIds.forEach((topLevelRunId) => runsExpanded[topLevelRunId] = true);
  const getRowMetadataWithSort = (({sortAsc}) => (ExperimentViewUtil.getRowRenderMetadata(
    {
      runInfos: Fixtures.runInfos,
      sortState: {ascending: sortAsc, isMetric: false, isParam: false, key: 'start_time'},
      paramsList: Array(Fixtures.runInfos.length).fill([]),
      metricsList: Array(Fixtures.runInfos.length).fill([]),
      tagsList: Fixtures.tagsList,
      runsExpanded,
    })
  ));
  const rowRenderMetadata0 = getRowMetadataWithSort({sortAsc: false});
  const sortedRunIds0 = rowRenderMetadata0.map((rowMetadata) => rowMetadata.runId);
  expect(sortedRunIds0).toEqual(
    ['parent-run-id', 'child-run-id-2', 'child-run-id-1', 'child-run-id-0',
      'top-level-childless-run-2', 'top-level-childless-run-1', 'top-level-childless-run-0']);
  const rowRenderMetadata1 = getRowMetadataWithSort({sortAsc: true});
  const sortedRunIds1 = rowRenderMetadata1.map((rowMetadata) => rowMetadata.runId);
  expect(sortedRunIds1).toEqual(
    ['top-level-childless-run-0', 'top-level-childless-run-1', 'top-level-childless-run-2',
      'parent-run-id', 'child-run-id-0', 'child-run-id-1', 'child-run-id-2']);
});

