import React from 'react';
import { shallow, mount } from 'enzyme';
import { CollapsibleSection } from './CollapsibleSection';
import { Collapse } from 'antd';

describe('CollapsibleSection', () => {
  let wrapper;
  let minimalProps;

  beforeEach(() => {
    minimalProps = {
      title: 'testTitle',
      children: 'testChild',
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<CollapsibleSection {...minimalProps} />);
    expect(wrapper.find(Collapse)).toHaveLength(1);
    expect(wrapper.find(Collapse.Panel)).toHaveLength(1);
  });

  test('collapse activeKeyProp is passed in when forceOpen is true', () => {
    minimalProps = {
      title: 'testTitle',
      children: 'testChild',
      forceOpen: true,
    };
    wrapper = shallow(<CollapsibleSection {...minimalProps} />);
    expect(wrapper.prop('activeKey')).toEqual(['1']);
  });

  test('should render mounted, and child should exist when clicked', () => {
    wrapper = mount(<CollapsibleSection {...minimalProps} />);
    expect(wrapper.text()).toMatch('testTitle');
    const header = wrapper.find('.collapsible-panel').first();
    expect(header.text()).toMatch('testChild');
    header.simulate('click');
    expect(header.text()).toMatch('testChild');
  });
});
