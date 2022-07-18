import React from 'react';
import { DirectTransitionForm } from './DirectTransitionForm';
import { Checkbox } from 'antd';
import _ from 'lodash';
import { mountWithIntl } from '../../common/utils/TestUtils';
import { Stages, stageTagComponents, modelStageNames } from '../test-utils';

describe('DirectTransitionForm', () => {
  let wrapper;
  let minimalProps;

  beforeEach(() => {
    minimalProps = {
      innerRef: React.createRef(),
      stageTagComponents: stageTagComponents(),
      availableStages: modelStageNames
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = mountWithIntl(<DirectTransitionForm {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should render checkbox only for active stage', () => {
    const [activeStages, nonActiveStages] = _.partition(Stages, (s) => modelStageNames.includes(s));

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
