import React from 'react';
import { shallow } from 'enzyme';
import { ExperimentRunsTableEmptyOverlay } from './ExperimentRunsTableEmptyOverlay';
import { LoggingRunsDocUrl } from '../constants';

describe('ExperimentRunsTableEmptyOverlay', () => {
  let wrapper;

  test('should render correct link', () => {
    wrapper = shallow(<ExperimentRunsTableEmptyOverlay />);
    expect(wrapper.find(`a[href="${LoggingRunsDocUrl}"]`)).toHaveLength(1);
  });
});
