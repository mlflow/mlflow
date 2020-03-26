import React from 'react';
import { shallow } from 'enzyme';
import { ExperimentRunsTableMultiColumnView2 } from './ExperimentRunsTableMultiColumnView2';
import { ColumnTypes } from '../constants';

describe('ExperimentRunsTableMultiColumnView2', () => {
  let wrapper;
  let instance;
  let minimalProps;

  beforeEach(() => {
    minimalProps = {
      runInfos: [],
      paramsList: [],
      metricsList: [],
      paramKeyList: [],
      metricKeyList: [],
      visibleTagKeyList: [],
      tagsList: [],
      onSelectionChange: jest.fn(),
      onExpand: jest.fn(),
      onSortBy: jest.fn(),
      runsSelected: {},
      runsExpanded: {},
      handleLoadMoreRuns: jest.fn(),
      loadingMore: false,
      isLoading: false,
      categorizedUncheckedKeys: {
        [ColumnTypes.ATTRIBUTES]: [],
        [ColumnTypes.PARAMS]: [],
        [ColumnTypes.METRICS]: [],
        [ColumnTypes.TAGS]: [],
      },
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ExperimentRunsTableMultiColumnView2 {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should not refit column size if there is an open column group', () => {
    wrapper = shallow(<ExperimentRunsTableMultiColumnView2 {...minimalProps} />);
    instance = wrapper.instance();
    instance.columnApi = {
      getColumnGroupState: jest.fn(() => [{ open: true }]),
    };
    instance.gridApi = {
      sizeColumnsToFit: jest.fn(),
    };
    instance.handleColumnSizeRefit();
    expect(instance.gridApi.sizeColumnsToFit).not.toBeCalled();

    instance.columnApi.getColumnGroupState = jest.fn(() => [{ open: false }]);
    instance.handleColumnSizeRefit();
    expect(instance.gridApi.sizeColumnsToFit).toBeCalled();
  });
});
