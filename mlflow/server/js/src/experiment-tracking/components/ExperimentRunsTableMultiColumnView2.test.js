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
      onFilter: jest.fn(),
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

  test('on filters change, search is called with good parameters', () => {
    wrapper = shallow(<ExperimentRunsTableMultiColumnView2 {...minimalProps} />);
    instance = wrapper.instance();
    const columns = [
      {
        // Common case colId==coldDef.field
        colId: '$$$param$$$-foo',
        colDef: { field: '$$$param$$$-foo' },
      },
      {
        // colId has a suffix (because array has been regenerated)
        colId: '$$$tag$$$-bar_1',
        colDef: { field: '$$$tag$$$-bar' },
      },
      {
        // A filter that is not a user params, tags or metrics
        colId: 'mlflow.tags.private',
        colDef: { field: 'mlflow.tags.private' },
      },
      {
        // With a space in parameter name
        colId: '$$$param$$$-foo bar',
        colDef: { field: '$$$param$$$-foo bar' },
      },
    ];
    instance.columnApi = { getAllDisplayedColumns: jest.fn(() => columns) };
    const filterInstances = new Map([
      [
        '$$$param$$$-foo',
        { getModel: jest.fn(() => ({ type: 'contains', filter: 'myfilter_1' })) },
      ],
      [
        '$$$tag$$$-bar_1',
        { getModel: jest.fn(() => ({ type: 'contains', filter: 'myfilter_2' })) },
      ],
      ['mlflow.tags.private', { getModel: jest.fn() }],
      [
        '$$$param$$$-foo bar',
        { getModel: jest.fn(() => ({ type: 'contains', filter: 'myfilter_3' })) },
      ],
    ]);
    instance.gridApi = {
      getFilterInstance: jest.fn((columnName) => filterInstances.get(columnName)),
    };
    instance.onFilterChanged();
    expect(minimalProps.onFilter).toHaveBeenCalledWith({
      'params."foo"': ['contains', 'myfilter_1'],
      'tags."bar"': ['contains', 'myfilter_2'],
      'params."foo bar"': ['contains', 'myfilter_3'],
    });
  });
});
