import React from 'react';
import { DirectTransitionForm } from './DirectTransitionForm';
import { ACTIVE_STAGES, Stages } from '../constants';
import { Checkbox } from 'antd';
import _ from 'lodash';
import { mountWithIntl } from '../../common/utils/TestUtils';

describe('DirectTransitionForm', () => {
  let wrapper;
  let minimalProps;

  beforeEach(() => {
    minimalProps = {
      innerRef: React.createRef(),
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = mountWithIntl(<DirectTransitionForm {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should render checkbox only for active stage', () => {
    const [activeStages, nonActiveStages] = _.partition(Stages, (s) => ACTIVE_STAGES.includes(s));

    activeStages.forEach((toStage) => {
      const props = { ...minimalProps, toStage };
      wrapper = mountWithIntl(<DirectTransitionForm {...props} />);
      expect(wrapper.find(Checkbox).length).toBe(1);
    });

    nonActiveStages.forEach((toStage) => {
      const props = { ...minimalProps, toStage };
      wrapper = mountWithIntl(<DirectTransitionForm {...props} />);
      expect(wrapper.find(Checkbox).length).toBe(0);
    });
  });
});
