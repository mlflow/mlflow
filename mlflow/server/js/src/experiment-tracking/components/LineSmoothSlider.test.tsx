/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { shallow } from 'enzyme';
import React from 'react';
import { LineSmoothSlider } from './LineSmoothSlider';

describe('LineSmoothSlider', () => {
  let wrapper;
  let minimalProps: any;

  beforeEach(() => {
    minimalProps = {
      min: 0,
      max: 1,
      handleLineSmoothChange: jest.fn(),
      defaultValue: 0,
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<LineSmoothSlider {...minimalProps} />).dive(1);
    expect(wrapper.length).toBe(1);
  });

  test('should render Slider and InputNumber with min|max|default value', () => {
    const props = {
      min: 0,
      max: 10,
      handleLineSmoothChange: jest.fn(),
      defaultValue: 5,
    };
    wrapper = shallow(<LineSmoothSlider {...props} />).dive(1);

    const slider = wrapper.find('Slider').get(0);
    expect(slider.props.min).toBe(0);
    expect(slider.props.max).toBe(10);
    expect(slider.props.value).toBe(5);

    const inputNumber = wrapper.find('[data-test-id="InputNumber"]').get(0);
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
    wrapper = shallow(<LineSmoothSlider {...props} />).dive(1);
    const inputNumber = wrapper.find('[data-test-id="InputNumber"]');
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
    wrapper = shallow(<LineSmoothSlider {...props} />).dive(1);
    const slider = wrapper.find('Slider');
    slider.simulate('change', 1);
    expect(props.handleLineSmoothChange).toHaveBeenCalledWith(1);
    expect(wrapper.state('inputValue')).toBe(1);
  });
});
