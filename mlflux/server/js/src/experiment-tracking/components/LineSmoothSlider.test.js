import { shallow } from 'enzyme';
import React from 'react';
import { LineSmoothSlider } from './LineSmoothSlider';

describe('LineSmoothSlider', () => {
  let wrapper;
  let minimalProps;

  beforeEach(() => {
    minimalProps = {
      min: 0,
      max: 1,
      handleLineSmoothChange: jest.fn(),
      defaultValue: 0,
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<LineSmoothSlider {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should render Slider and InputNumber with min|max|default value', () => {
    const props = {
      min: 0,
      max: 10,
      handleLineSmoothChange: jest.fn(),
      defaultValue: 5,
    };
    wrapper = shallow(<LineSmoothSlider {...props} />);
    expect(wrapper.state('inputValue')).toBe(5);

    const slider = wrapper.find('Slider').get(0);
    expect(slider.props.min).toBe(0);
    expect(slider.props.max).toBe(10);
    expect(slider.props.value).toBe(5);

    const inputNumber = wrapper.find('InputNumber').get(0);
    expect(inputNumber.props.min).toBe(0);
    expect(inputNumber.props.max).toBe(10);
    expect(inputNumber.props.value).toBe(5);
  });

  test('should invoke handleLineSmoothChange when InputNumber value is changed', () => {
    const props = {
      min: 0,
      max: 10,
      handleLineSmoothChange: jest.fn(),
      defaultValue: 5,
    };
    wrapper = shallow(<LineSmoothSlider {...props} />);
    const inputNumber = wrapper.find('InputNumber');
    inputNumber.simulate('change', 6);
    expect(props.handleLineSmoothChange).toHaveBeenCalledWith(6);
    expect(wrapper.state('inputValue')).toBe(6);
  });

  test('should invoke Slider when InputNumber value is changed', () => {
    const props = {
      min: 0,
      max: 10,
      handleLineSmoothChange: jest.fn(),
      defaultValue: 5,
    };
    wrapper = shallow(<LineSmoothSlider {...props} />);
    const slider = wrapper.find('Slider');
    slider.simulate('change', 1);
    expect(props.handleLineSmoothChange).toHaveBeenCalledWith(1);
    expect(wrapper.state('inputValue')).toBe(1);
  });
});
