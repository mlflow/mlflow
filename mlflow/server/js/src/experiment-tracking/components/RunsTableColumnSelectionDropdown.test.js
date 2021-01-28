import React from 'react';
import { shallow, mount } from 'enzyme';
import { RunsTableColumnSelectionDropdown } from './RunsTableColumnSelectionDropdown';
import { SearchTree } from '../../common/components/SearchTree';
import { ColumnTypes } from '../constants';

describe('RunsTableColumnSelectionDropdown', () => {
  let wrapper;
  let instance;
  let minimalProps;
  let commonProps;

  beforeEach(() => {
    minimalProps = {
      paramKeyList: [],
      metricKeyList: [],
      visibleTagKeyList: [],
      onCheck: jest.fn(),
      categorizedUncheckedKeys: {
        [ColumnTypes.ATTRIBUTES]: [],
        [ColumnTypes.PARAMS]: [],
        [ColumnTypes.METRICS]: [],
        [ColumnTypes.TAGS]: [],
      },
    };

    commonProps = {
      ...minimalProps,
      paramKeyList: ['p1', 'p2'],
      metricKeyList: ['m1', 'm2'],
      visibleTagKeyList: ['t1', 't2'],
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<RunsTableColumnSelectionDropdown {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should render SearchTree with correct tree data', () => {
    wrapper = mount(<RunsTableColumnSelectionDropdown {...commonProps} />);
    instance = wrapper.instance();
    instance.setState({ menuVisible: true });
    wrapper.update();
    expect(wrapper.find(SearchTree).prop('data')).toEqual([
      { key: 'attributes-Start Time', title: 'Start Time' },
      { key: 'attributes-User', title: 'User' },
      { key: 'attributes-Run Name', title: 'Run Name' },
      { key: 'attributes-Source', title: 'Source' },
      { key: 'attributes-Version', title: 'Version' },
      { key: 'attributes-Models', title: 'Models' },
      {
        title: 'Parameters',
        key: 'params',
        children: [
          { key: 'params-p1', title: 'p1' },
          { key: 'params-p2', title: 'p2' },
        ],
      },
      {
        title: 'Metrics',
        key: 'metrics',
        children: [
          { key: 'metrics-m1', title: 'm1' },
          { key: 'metrics-m2', title: 'm2' },
        ],
      },
      {
        title: 'Tags',
        key: 'tags',
        children: [
          { key: 'tags-t1', title: 't1' },
          { key: 'tags-t2', title: 't2' },
        ],
      },
    ]);
  });

  test('should check all keys by default', () => {
    wrapper = mount(<RunsTableColumnSelectionDropdown {...commonProps} />);
    instance = wrapper.instance();
    instance.setState({ menuVisible: true });
    wrapper.update();
    expect(wrapper.find(SearchTree).prop('checkedKeys')).toEqual([
      'attributes-Start Time',
      'attributes-User',
      'attributes-Run Name',
      'attributes-Source',
      'attributes-Version',
      'attributes-Models',
      'params-p1',
      'params-p2',
      'metrics-m1',
      'metrics-m2',
      'tags-t1',
      'tags-t2',
    ]);
  });

  test('should not check keys marked as unchecked', () => {
    const props = {
      ...commonProps,
      categorizedUncheckedKeys: {
        [ColumnTypes.ATTRIBUTES]: ['User', 'Run Name', 'Source', 'Models', 'Version'],
        [ColumnTypes.PARAMS]: ['p1'],
        [ColumnTypes.METRICS]: ['m1'],
        [ColumnTypes.TAGS]: ['t1'],
      },
    };
    wrapper = mount(<RunsTableColumnSelectionDropdown {...props} />);
    instance = wrapper.instance();
    instance.setState({ menuVisible: true });
    wrapper.update();
    expect(wrapper.find(SearchTree).prop('checkedKeys')).toEqual([
      'attributes-Start Time',
      'params-p2',
      'metrics-m2',
      'tags-t2',
    ]);
  });
});
