import React from 'react';
import { TrimmedText } from './TrimmedText';
import { shallow } from 'enzyme';

const getDefaultTrimmedTextProps = (overrides = {}) => ({
  text: '0123456789',
  maxSize: 10,
  className: 'some class',
  allowShowMore: false,
  ...overrides,
});

describe('TrimmedText', () => {
  test('render normal text if length is less than or equal to max size', () => {
    [true, false].forEach((allowShowMore) => {
      const wrapper = shallow(<TrimmedText {...getDefaultTrimmedTextProps({ allowShowMore: allowShowMore })} />);
      expect(wrapper.text()).toEqual('0123456789');
    });
  });

  test('render trimmed text if length is greater than max size', () => {
    const wrapper = shallow(<TrimmedText {...getDefaultTrimmedTextProps({ maxSize: 5 })} />);
    expect(wrapper.text()).toEqual('01234...');
    expect(wrapper.find('[data-test-id="trimmed-text-button"]').length).toEqual(0);
  });

  test('render show more button if configured', () => {
    const wrapper = shallow(<TrimmedText {...getDefaultTrimmedTextProps({ maxSize: 5, allowShowMore: true })} />);
    expect(wrapper.find('[data-test-id="trimmed-text-button"]').length).toEqual(1);
    expect(wrapper.text().includes('01234...')).toBe(true);
    expect(wrapper.find('[data-test-id="trimmed-text-button"]').children(0).text()).toEqual('expand');
    wrapper.find('[data-test-id="trimmed-text-button"]').simulate('click');
    wrapper.update();
    expect(wrapper.text().includes('0123456789')).toBe(true);
    expect(wrapper.find('[data-test-id="trimmed-text-button"]').children(0).text()).toEqual('collapse');
    wrapper.find('[data-test-id="trimmed-text-button"]').simulate('click');
    wrapper.update();
    expect(wrapper.text().includes('01234...')).toBe(true);
    expect(wrapper.find('[data-test-id="trimmed-text-button"]').children(0).text()).toEqual('expand');
  });
});
