import React from 'react';
import {
  RunsTableColumnSelectionDropdown,
  getCategorizedUncheckedKeys,
} from './RunsTableColumnSelectionDropdown';
import { SearchTree } from '../../common/components/SearchTree';
import { COLUMN_TYPES } from '../constants';
import { mountWithIntl, shallowWithIntl } from '../../common/utils/TestUtils';

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
        [COLUMN_TYPES.ATTRIBUTES]: [],
        [COLUMN_TYPES.PARAMS]: [],
        [COLUMN_TYPES.METRICS]: [],
        [COLUMN_TYPES.TAGS]: [],
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
    wrapper = shallowWithIntl(<RunsTableColumnSelectionDropdown {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should render SearchTree with correct tree data', () => {
    wrapper = mountWithIntl(<RunsTableColumnSelectionDropdown {...commonProps} />);
    instance = wrapper.instance();
    instance.setState({ menuVisible: true });
    wrapper.update();
    expect(wrapper.find(SearchTree).prop('data')).toEqual([
      { key: 'attributes-Start Time', title: 'Start Time' },
      { key: 'attributes-Duration', title: 'Duration' },
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
    wrapper = mountWithIntl(<RunsTableColumnSelectionDropdown {...commonProps} />);
    instance = wrapper.instance();
    instance.setState({ menuVisible: true });
    wrapper.update();
    expect(wrapper.find(SearchTree).prop('checkedKeys')).toEqual([
      'attributes-Start Time',
      'attributes-Duration',
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
        [COLUMN_TYPES.ATTRIBUTES]: ['User', 'Run Name', 'Source', 'Models', 'Version'],
        [COLUMN_TYPES.PARAMS]: ['p1'],
        [COLUMN_TYPES.METRICS]: ['m1'],
        [COLUMN_TYPES.TAGS]: ['t1'],
      },
    };
    wrapper = mountWithIntl(<RunsTableColumnSelectionDropdown {...props} />);
    instance = wrapper.instance();
    instance.setState({ menuVisible: true });
    wrapper.update();
    expect(wrapper.find(SearchTree).prop('checkedKeys')).toEqual([
      'attributes-Start Time',
      'attributes-Duration',
      'params-p2',
      'metrics-m2',
      'tags-t2',
    ]);
  });
});

describe('getCategorizedUncheckedKeys', () => {
  test('getCategorizedUncheckedKeys should return correct keys when all checked', () => {
    const allKeys = [
      'attributes-Start Time',
      'attributes-Duration',
      'attributes-User',
      'attributes-Run Name',
      'attributes-Source',
      'attributes-Version',
      'attributes-Models',
      'params',
      'params-p1',
      'params-p2',
      'metrics',
      'metrics-m1',
      'metrics-m2',
      'tags',
      'tags-t1',
      'tags-t2',
    ];
    const checkedKeys = [
      'attributes-Start Time',
      'attributes-Duration',
      'attributes-User',
      'attributes-Run Name',
      'attributes-Source',
      'attributes-Version',
      'attributes-Models',
      'params',
      'params-p1',
      'params-p2',
      'metrics',
      'metrics-m1',
      'metrics-m2',
      'tags',
      'tags-t1',
      'tags-t2',
    ];
    const expectedResult = {
      [COLUMN_TYPES.ATTRIBUTES]: [],
      [COLUMN_TYPES.PARAMS]: [],
      [COLUMN_TYPES.METRICS]: [],
      [COLUMN_TYPES.TAGS]: [],
    };
    expect(getCategorizedUncheckedKeys(checkedKeys, allKeys)).toEqual(expectedResult);
  });
  test('getCategorizedUncheckedKeys should return correct keys when some checked', () => {
    const allKeys = [
      'attributes-Start Time',
      'attributes-Duration',
      'attributes-User',
      'attributes-Run Name',
      'attributes-Source',
      'attributes-Version',
      'attributes-Models',
      'params',
      'params-p1',
      'params-p2',
      'metrics',
      'metrics-m1',
      'metrics-m2',
      'tags',
      'tags-t1',
      'tags-t2',
    ];
    const checkedKeys = [
      'attributes-Start Time',
      'attributes-Duration',
      'attributes-Source',
      'attributes-Version',
      'attributes-Models',
      'params-p1',
      'metrics-m1',
      'tags-t1',
    ];
    const expectedResult = {
      [COLUMN_TYPES.ATTRIBUTES]: ['User', 'Run Name'],
      [COLUMN_TYPES.PARAMS]: ['p2'],
      [COLUMN_TYPES.METRICS]: ['m2'],
      [COLUMN_TYPES.TAGS]: ['t2'],
    };
    expect(getCategorizedUncheckedKeys(checkedKeys, allKeys)).toEqual(expectedResult);
  });
  test('getCategorizedUncheckedKeys should return correct keys when nothing checked', () => {
    const allKeys = [
      'attributes-Start Time',
      'attributes-Duration',
      'attributes-User',
      'attributes-Run Name',
      'attributes-Source',
      'attributes-Version',
      'attributes-Models',
      'params',
      'params-p1',
      'params-p2',
      'metrics',
      'metrics-m1',
      'metrics-m2',
      'tags',
      'tags-t1',
      'tags-t2',
    ];
    const checkedKeys = [];
    const expectedResult = {
      [COLUMN_TYPES.ATTRIBUTES]: [
        'Start Time',
        'Duration',
        'User',
        'Run Name',
        'Source',
        'Version',
        'Models',
      ],
      [COLUMN_TYPES.PARAMS]: ['p1', 'p2'],
      [COLUMN_TYPES.METRICS]: ['m1', 'm2'],
      [COLUMN_TYPES.TAGS]: ['t1', 't2'],
    };
    expect(getCategorizedUncheckedKeys(checkedKeys, allKeys)).toEqual(expectedResult);
  });
});
