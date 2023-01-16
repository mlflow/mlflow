import React from 'react';
import { ExperimentRunsTableEmptyOverlay } from './ExperimentRunsTableEmptyOverlay';
import { LoggingRunsDocUrl } from '../constants';
import { mountWithIntl } from '../../common/utils/TestUtils';

describe('ExperimentRunsTableEmptyOverlay', () => {
  let wrapper;

  test('should render correct link', () => {
    wrapper = mountWithIntl(<ExperimentRunsTableEmptyOverlay />);
    expect(wrapper.find(`a[href="${LoggingRunsDocUrl}"]`)).toHaveLength(1);
  });
});
