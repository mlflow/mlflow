import React from 'react';
import { IntlProvider } from 'react-intl';
import { MemoryRouter } from '../../../../../common/utils/RoutingUtils';
import { ExperimentViewHeader } from './ExperimentViewHeader';
import { ExperimentViewHeaderCompare } from './ExperimentViewHeaderCompare';

export default {
  title: 'ExperimentView/ExperimentViewHeader',
  component: ExperimentViewHeader,
  argTypes: {},
};

const mockExperiments = [
  {
    experimentId: '123456789',
    name: '/Users/john.doe@databricks.com/test-experiment',
    tags: [],
    allowedActions: ['MODIFIY_PERMISSION', 'DELETE', 'RENAME'],
  },
  {
    experimentId: '654321',
    name: '/Users/john.doe@databricks.com/another-experiment',
    tags: [],
    allowedActions: [],
  },
] as any;

const Wrapper = ({ children }: React.PropsWithChildren<any>) => (
  <IntlProvider locale="en">
    <MemoryRouter initialEntries={['/']}>{children}</MemoryRouter>
  </IntlProvider>
);

export const SingleExperiment = () => (
  <Wrapper>
    <ExperimentViewHeader experiment={mockExperiments[0]} showAddDescriptionButton setEditing={(b) => null} />
  </Wrapper>
);

export const SingleExperimentWithoutPermissions = () => (
  <Wrapper>
    <ExperimentViewHeader experiment={mockExperiments[1]} showAddDescriptionButton setEditing={(b) => null} />
  </Wrapper>
);

export const CompareExperiments = () => (
  <Wrapper>
    <ExperimentViewHeaderCompare experiments={mockExperiments} />
  </Wrapper>
);
