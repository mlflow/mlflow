import React from 'react';
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
  // ,
  //         isParent: true,
  //         hasExpander,
  //         expanderOpen: ExperimentViewUtil.isExpanderOpen(runsExpanded, runId),
  //         childrenIds,
  //         runId,
  //         sortValue,
  expect(rowRenderMetadata)
});

