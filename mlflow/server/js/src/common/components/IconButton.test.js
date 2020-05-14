import React from 'react';
import { shallow, mount } from 'enzyme';
import { Button, Icon } from 'antd';
import { IconButton } from './IconButton';

describe('IconButton', () => {
  let wrapper;
  let mockOnClick;
  let minimalProps;

  beforeEach(() => {
    minimalProps = { children: <Icon type='edit' /> };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<IconButton {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should not have padding', () => {
    wrapper = shallow(<IconButton {...minimalProps} />);
    expect(wrapper.find(Button).get(0).props.style).toHaveProperty('padding', 0);
  });

  test('should pass style to Button', () => {
    wrapper = shallow(<IconButton {...minimalProps} style={{ margin: 5 }} />);
    expect(wrapper.find(Button).get(0).props.style).toHaveProperty('margin', 5);
  });

  test('should trigger onClick when clicked', () => {
    mockOnClick = jest.fn();
    wrapper = mount(<IconButton {...minimalProps} onClick={mockOnClick} />);
    wrapper.find('button').simulate('click');
    expect(mockOnClick).toHaveBeenCalledTimes(1);
  });
});
