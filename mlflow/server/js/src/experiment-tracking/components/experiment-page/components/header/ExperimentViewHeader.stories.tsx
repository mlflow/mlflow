import React from 'react';
import { IntlProvider } from 'react-intl';
import { StaticRouter } from '../../../../../common/utils/RoutingUtils';
import { ExperimentViewHeader } from './ExperimentViewHeader';
import { ExperimentViewHeaderCompare } from './ExperimentViewHeaderCompare';

export default {
  title: 'ExperimentView/ExperimentViewHeader',
  component: ExperimentViewHeader,
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
  <IntlProvider locale='en'>
    <StaticRouter location='/'>{children}</StaticRouter>
  </IntlProvider>
);

export const SingleExperiment = () => (
  <Wrapper>
    <ExperimentViewHeader experiment={mockExperiments[0]} />
  </Wrapper>
);

export const SingleExperimentWithoutPermissions = () => (
  <Wrapper>
    <ExperimentViewHeader experiment={mockExperiments[1]} />
  </Wrapper>
);

export const CompareExperiments = () => (
  <Wrapper>
    <ExperimentViewHeaderCompare experiments={mockExperiments} />
  </Wrapper>
);
