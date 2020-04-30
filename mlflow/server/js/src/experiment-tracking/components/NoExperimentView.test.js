import React from 'react';
import { shallow } from 'enzyme';
import { NoExperimentView } from './NoExperimentView';

describe('NoExperimentView', () => {
  test('should render without exploding', () => {
    shallow(<NoExperimentView />);
  });
});
