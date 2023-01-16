import React from 'react';
import { shallow } from 'enzyme';
import Fixtures from '../utils/test-utils/Fixtures';
import Utils from '../../common/utils/Utils';
import { CompareRunView } from './CompareRunView';
import { createIntl } from 'react-intl';

const getCompareRunViewMock = () => {
  return shallow(
    <CompareRunView
      runInfos={[Fixtures.createRunInfo(), Fixtures.createRunInfo()]}
      experiments={[Fixtures.createExperiment()]}
      experimentIds={['0']}
      runUuids={['0']}
      metricLists={[[{ m: 1 }]]}
      paramLists={[[{ p: 'p' }]]}
      tagLists={[[{ t: 't' }]]}
      runNames={['run1']}
      runDisplayNames={['run1DisplayName', 'run2DisplayName']}
      intl={createIntl({ locale: 'en' })}
    />,
  );
};

test('Page title is set', () => {
  const mockUpdatePageTitle = jest.fn();
  Utils.updatePageTitle = mockUpdatePageTitle;
  getCompareRunViewMock();
  expect(mockUpdatePageTitle.mock.calls[0][0]).toBe('Comparing 2 MLflow Runs');
});
