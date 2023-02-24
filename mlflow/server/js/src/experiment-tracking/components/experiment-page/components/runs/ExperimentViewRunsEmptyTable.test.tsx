import React from 'react';
import { LoggingRunsDocUrl } from '../../../../../common/constants';
import { ExperimentViewRunsEmptyTable } from './ExperimentViewRunsEmptyTable';
import { mountWithIntl } from '../../../../../common/utils/TestUtils';

describe('ExperimentRunsTableEmptyOverlay', () => {
  let wrapper;

  test('should render button when runs are filtered', () => {
    wrapper = mountWithIntl(<ExperimentViewRunsEmptyTable onClearFilters={() => {}} isFiltered />);
    expect(wrapper.find('Button')).toHaveLength(1);
  });

  test('should render correct link', () => {
    wrapper = mountWithIntl(
      <ExperimentViewRunsEmptyTable onClearFilters={() => {}} isFiltered={false} />,
    );
    expect(wrapper.find(`a[href="${LoggingRunsDocUrl}"]`)).toHaveLength(1);
  });
});
