/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { SearchTree, SearchTreeImpl, styles, getParentKey, flattenDataToList } from './SearchTree';
import { shallowWithInjectIntl } from 'common/utils/TestUtils.enzyme';
import { mountWithIntl } from 'common/utils/TestUtils.enzyme';
import { createIntl } from 'react-intl';

describe('SearchTree', () => {
  let wrapper;
  let instance;
  let minimalProps: any;
  let commonProps: any;

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
    wrapper = mountWithIntl(<SearchTree {...commonProps} />);
    const treeNodes = wrapper.find('div[data-testid="tree-node"]');
    expect(treeNodes.length).toBe(3);
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
