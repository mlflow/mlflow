import React from 'react';
import { IntlProvider } from 'react-intl';
import { StaticRouter } from '../../../../common/utils/RoutingUtils';
import { ExperimentViewDescriptions } from './ExperimentViewDescriptions';

export default {
  title: 'ExperimentView/ExperimentViewDescriptions',
  component: ExperimentViewDescriptions,
  argTypes: {},
};

const mockExperiments = [
  {
    experiment_id: '123456789',
    name: '/Users/john.doe@databricks.com/test-experiment',
    tags: [],
    allowed_actions: ['MODIFIY_PERMISSION', 'DELETE', 'RENAME'],
    artifact_location: 'dbfs://foo/bar/xyz',
  },
] as any;

const Wrapper = ({ children }: React.PropsWithChildren<any>) => (
  <IntlProvider locale='en'>
    <StaticRouter location='/'>{children}</StaticRouter>
  </IntlProvider>
);

export const Simple = () => (
  <Wrapper>
    <ExperimentViewDescriptions experiment={mockExperiments[0]} />
  </Wrapper>
);
