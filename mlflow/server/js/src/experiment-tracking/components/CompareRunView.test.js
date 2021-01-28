import React from 'react';
import { shallow } from 'enzyme';
import Fixtures from '../utils/test-utils/Fixtures';
import Utils from '../../common/utils/Utils';
import { CompareRunView } from './CompareRunView';

const getCompareRunViewMock = () => {
  return shallow(
    <CompareRunView
      runInfos={[Fixtures.createRunInfo(), Fixtures.createRunInfo()]}
      experiment={Fixtures.createExperiment()}
      experimentId={'0'}
      runUuids={['0']}
      metricLists={[]}
      paramLists={[]}
      runNames={['run1']}
      runDisplayNames={['run1DisplayName', 'run2DisplayName']}
    />,
  );
};

test('Page title is set', () => {
  const mockUpdatePageTitle = jest.fn();
  Utils.updatePageTitle = mockUpdatePageTitle;
  getCompareRunViewMock();
  expect(mockUpdatePageTitle.mock.calls[0][0]).toBe('Comparing 2 MLflow Runs');
});
