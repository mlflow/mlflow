import React from 'react';
import { LoggingRunsDocUrl } from '../../../../../common/constants';
import { ExperimentViewRunsEmptyTable } from './ExperimentViewRunsEmptyTable';
import { mountWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.enzyme';

describe('ExperimentRunsTableEmptyOverlay', () => {
  let wrapper;

  test('should render button when runs are filtered', () => {
    wrapper = mountWithIntl(<ExperimentViewRunsEmptyTable onClearFilters={() => {}} isFiltered />);
    expect(wrapper.find('Button')).toHaveLength(1);
  });

  test('should render correct link', () => {
    wrapper = mountWithIntl(<ExperimentViewRunsEmptyTable onClearFilters={() => {}} isFiltered={false} />);
    // eslint-disable-next-line jest/no-standalone-expect
    expect(wrapper.find(`a[href="${LoggingRunsDocUrl}"]`)).toHaveLength(1);
  });
});
