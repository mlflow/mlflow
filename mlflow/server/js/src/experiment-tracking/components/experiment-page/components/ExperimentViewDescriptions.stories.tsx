import React from 'react';
import { IntlProvider } from 'react-intl';
import { MemoryRouter } from '../../../../common/utils/RoutingUtils';
import { ExperimentViewDescriptions } from './ExperimentViewDescriptions';

export default {
  title: 'ExperimentView/ExperimentViewDescriptions',
  component: ExperimentViewDescriptions,
  argTypes: {},
};

const mockExperiments = [
  {
    experimentId: '123456789',
    name: '/Users/john.doe@databricks.com/test-experiment',
    tags: [],
    allowedActions: ['MODIFIY_PERMISSION', 'DELETE', 'RENAME'],
    artifactLocation: 'dbfs://foo/bar/xyz',
  },
] as any;

const Wrapper = ({ children }: React.PropsWithChildren<any>) => (
  <IntlProvider locale="en">
    <MemoryRouter initialEntries={['/']}>{children}</MemoryRouter>
  </IntlProvider>
);

export const Simple = () => (
  <Wrapper>
    <ExperimentViewDescriptions experiment={mockExperiments[0]} />
  </Wrapper>
);
