import React from 'react';
import { shallow } from 'enzyme';
import { DirectTransitionFormImpl } from './DirectTransitionForm';
import { ACTIVE_STAGES, Stages } from '../constants';
import { Checkbox } from 'antd';
import _ from 'lodash';

describe('DirectTransitionForm', () => {
  let wrapper;
  let minimalProps;

  beforeEach(() => {
    minimalProps = {
      form: { getFieldDecorator: jest.fn(() => (c) => c) },
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<DirectTransitionFormImpl {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should render checkbox only for active stage', () => {
    const [activeStages, nonActiveStages] = _.partition(Stages, (s) => ACTIVE_STAGES.includes(s));

    activeStages.forEach((toStage) => {
      const props = { ...minimalProps, toStage };
      wrapper = shallow(<DirectTransitionFormImpl {...props} />);
      expect(wrapper.find(Checkbox).length).toBe(1);
    });

    nonActiveStages.forEach((toStage) => {
      const props = { ...minimalProps, toStage };
      wrapper = shallow(<DirectTransitionFormImpl {...props} />);
      expect(wrapper.find(Checkbox).length).toBe(0);
    });
  });
});
