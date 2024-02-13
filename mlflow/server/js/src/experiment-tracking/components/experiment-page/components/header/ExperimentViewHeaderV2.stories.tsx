import React from 'react';
import { IntlProvider } from 'react-intl';
import { StaticRouter } from '../../../../../common/utils/RoutingUtils';
import { ExperimentViewHeaderV2 } from './ExperimentViewHeaderV2';
import { ExperimentViewHeaderCompare } from './ExperimentViewHeaderCompare';

export default {
  title: 'ExperimentView/ExperimentViewHeaderV2',
  component: ExperimentViewHeaderV2,
  argTypes: {},
};

const mockExperiments = [
  {
    experiment_id: '123456789',
    name: '/Users/john.doe@databricks.com/test-experiment',
    tags: [],
    allowed_actions: ['MODIFIY_PERMISSION', 'DELETE', 'RENAME'],
  },
  {
    experiment_id: '654321',
    name: '/Users/john.doe@databricks.com/another-experiment',
    tags: [],
    allowed_actions: [],
  },
] as any;

const Wrapper = ({ children }: React.PropsWithChildren<any>) => (
  <IntlProvider locale="en">
    <StaticRouter location="/">{children}</StaticRouter>
  </IntlProvider>
);

export const SingleExperiment = () => (
  <Wrapper>
    <ExperimentViewHeaderV2 experiment={mockExperiments[0]} showAddDescriptionButton setEditing={(b) => null} />
  </Wrapper>
);

export const SingleExperimentWithoutPermissions = () => (
  <Wrapper>
    <ExperimentViewHeaderV2 experiment={mockExperiments[1]} showAddDescriptionButton setEditing={(b) => null} />
  </Wrapper>
);

export const CompareExperiments = () => (
  <Wrapper>
    <ExperimentViewHeaderCompare experiments={mockExperiments} />
  </Wrapper>
);
