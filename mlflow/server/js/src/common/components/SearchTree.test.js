import React from 'react';
import { SearchTree, SearchTreeImpl, styles, getParentKey, flattenDataToList } from './SearchTree';
import { Tree } from 'antd';
import { shallowWithInjectIntl, mountWithIntl } from '../../common/utils/TestUtils';
import { createIntl } from 'react-intl';

const { TreeNode } = Tree;

describe('SearchTree', () => {
  let wrapper;
  let instance;
  let minimalProps;
  let commonProps;

  beforeEach(() => {
    minimalProps = {
      data: [],
      onCheck: jest.fn(),
      checkedKeys: [],
      onSearchInputEscapeKeyPress: jest.fn(),
    };

    commonProps = {
      ...minimalProps,
      data: [
        {
          title: 'Date',
          key: 'attributes-Date',
        },
        {
          title: 'Parameters',
          key: 'params',
          children: [
            {
              title: 'p1',
              key: 'params-p1',
            },
            {
              title: 'p2',
              key: 'params-p2',
            },
          ],
        },
        {
          title: 'Metrics',
          key: 'metrics',
          children: [
            {
              title: 'm1',
              key: 'metrics-m1',
            },
            {
              title: 'm2',
              key: 'metrics-m2',
            },
          ],
        },
      ],
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallowWithInjectIntl(<SearchTree {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should render search tree properly', () => {
    wrapper = shallowWithInjectIntl(<SearchTree {...commonProps} />);
    const treeNodes = wrapper.find(TreeNode);
    expect(treeNodes.length).toBe(7);
  });

  test('should highlight filter matched nodes correctly', () => {
    const intl = createIntl({ locale: 'en' });
    const props = { ...commonProps, intl };
    wrapper = mountWithIntl(<SearchTreeImpl {...props} />);
    instance = wrapper.instance();
    instance.handleSearch({
      target: {
        value: 'p',
      },
    });
    wrapper.update();
    expect(wrapper.find({ style: styles.searchHighlight }).length).toBe(3);

    instance.handleSearch({
      target: {
        value: 'p1',
      },
    });
    wrapper.update();
    expect(wrapper.find({ style: styles.searchHighlight }).length).toBe(1);
  });

  test('flattenDataToList', () => {
    const { data } = commonProps;
    expect(flattenDataToList(data)).toEqual([
      { title: 'Date', key: 'attributes-Date' },
      { title: 'Parameters', key: 'params' },
      { title: 'p1', key: 'params-p1' },
      { title: 'p2', key: 'params-p2' },
      { title: 'Metrics', key: 'metrics' },
      { title: 'm1', key: 'metrics-m1' },
      { title: 'm2', key: 'metrics-m2' },
    ]);
  });

  test('getParentKey', () => {
    const { data } = commonProps;
    expect(getParentKey('params-p1', data)).toBe('params');
    expect(getParentKey('params-p2', data)).toBe('params');
    expect(getParentKey('metrics-m1', data)).toBe('metrics');
    expect(getParentKey('metrics-m2', data)).toBe('metrics');
    expect(getParentKey('attributes-Date', data)).toBe(undefined);
  });
});
