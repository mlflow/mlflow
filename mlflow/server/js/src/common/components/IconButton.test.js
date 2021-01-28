import React from 'react';
import { shallow, mount } from 'enzyme';
import { Button, Icon } from 'antd';
import { IconButton } from './IconButton';

describe('IconButton', () => {
  let wrapper;
  let mockOnClick;
  let minimalProps;

  beforeEach(() => {
    minimalProps = { icon: <Icon type='edit' /> };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<IconButton {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should not have padding', () => {
    wrapper = shallow(<IconButton {...minimalProps} />);
    expect(wrapper.find(Button).get(0).props.style).toHaveProperty('padding', 0);
  });

  test('should propagate props to Button', () => {
    const props = {
      className: 'class',
      style: { margin: 5 },
    };
    wrapper = shallow(<IconButton {...{ ...minimalProps, ...props }} />);
    expect(wrapper.find(Button).get(0).props).toHaveProperty('className', 'class');
    expect(wrapper.find(Button).get(0).props.style).toHaveProperty('margin', 5);
  });

  test('should trigger onClick when clicked', () => {
    mockOnClick = jest.fn();
    wrapper = mount(<IconButton {...minimalProps} onClick={mockOnClick} />);
    wrapper.find('button').simulate('click');
    expect(mockOnClick).toHaveBeenCalledTimes(1);
  });
});
