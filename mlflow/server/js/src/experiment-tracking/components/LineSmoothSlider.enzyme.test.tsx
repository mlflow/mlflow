/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { shallow } from 'enzyme';
import React from 'react';
import { LineSmoothSlider } from './LineSmoothSlider';
import { Slider } from 'antd';

describe('LineSmoothSlider', () => {
  let wrapper;
  let minimalProps: any;

  beforeEach(() => {
    minimalProps = {
      min: 0,
      max: 1,
      onChange: jest.fn(),
      defaultValue: 0,
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<LineSmoothSlider {...minimalProps} />)
      .find(Slider)
      .dive(1);
    expect(wrapper.length).toBe(1);
  });

  test('should render Slider and InputNumber with min|max|default value', () => {
    const props = {
      min: 0,
      max: 10,
      onChange: jest.fn(),
      defaultValue: 5,
    };
    wrapper = shallow(<LineSmoothSlider {...props} />);

    const slider = wrapper.find('Slider').get(0);
    expect(slider.props.min).toBe(0);
    expect(slider.props.max).toBe(10);
    expect(slider.props.value).toBe(5);

    const inputNumber = wrapper.find('[data-test-id="InputNumber"]').get(0);
    expect(inputNumber.props.min).toBe(0);
    expect(inputNumber.props.max).toBe(10);
    expect(inputNumber.props.value).toBe(5);
  });

  test('should invoke onChange when InputNumber value is changed', () => {
    const props = {
      min: 0,
      max: 10,
      onChange: jest.fn(),
      defaultValue: 5,
    };
    wrapper = shallow(<LineSmoothSlider {...props} />);
    const inputNumber = wrapper.find('[data-test-id="InputNumber"]');
    inputNumber.simulate('change', 6);
    expect(props.onChange).toHaveBeenCalledWith(6);
  });

  test('should invoke Slider when InputNumber value is changed', () => {
    const props = {
      min: 0,
      max: 10,
      onChange: jest.fn(),
      defaultValue: 5,
    };
    wrapper = shallow(<LineSmoothSlider {...props} />);
    const slider = wrapper.find('Slider');
    slider.simulate('change', 1);
    expect(props.onChange).toHaveBeenCalledWith(1);
  });
});
