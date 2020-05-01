import React from 'react';
import { shallow } from 'enzyme';
import { ExperimentRunsTableCompactView } from './ExperimentRunsTableCompactView';

describe('ExperimentRunsTableCompactView', () => {
  let wrapper;
  let minimalProps;

  beforeEach(() => {
    minimalProps = {
      runInfos: [],
      paramsList: [],
      metricsList: [],
      tagsList: [],
      onCheckbox: () => {},
      onCheckAll: () => {},
      onExpand: () => {},
      isAllChecked: false,
      onSortBy: () => {},
      orderByAsc: true,
      runsSelected: {},
      runsExpanded: {},
      paramKeyList: [],
      metricKeyList: [],
      metricRanges: {},
      onAddBagged: () => {},
      onRemoveBagged: () => {},
      unbaggedParams: [],
      unbaggedMetrics: [],
      handleLoadMoreRuns: () => {},
      loadingMore: false,
      categorizedUncheckedKeys: {},
    };
    wrapper = shallow(<ExperimentRunsTableCompactView {...minimalProps} />);
  });

  test('should render with minimal props without exploding', () => {
    expect(wrapper.length).toBe(1);
  });
});
