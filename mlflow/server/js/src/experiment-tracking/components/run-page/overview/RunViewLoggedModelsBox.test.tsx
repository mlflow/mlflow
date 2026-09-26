import { describe, it, expect } from '@jest/globals';
import { IntlProvider } from 'react-intl';
import { render, screen } from '../../../../common/utils/TestUtils.react18';
import type { LoggedModelProto, RunInfoEntity } from '../../../types';
import { RunViewLoggedModelsBox } from './RunViewLoggedModelsBox';
import { DesignSystemProvider } from '@databricks/design-system';
import { TestRouter, testRoute } from '../../../../common/utils/RoutingTestUtils';
import type { ComponentProps } from 'react';

describe('RunViewLoggedModelsBox', () => {
  const runInfo: RunInfoEntity = {
    runUuid: 'run-id',
    experimentId: 'experiment-id',
    startTime: 0,
    endTime: 1,
    artifactUri: '',
    lifecycleStage: 'active',
    status: 'FINISHED',
    runName: 'run-name',
  };

  const renderTestComponent = (props: Partial<ComponentProps<typeof RunViewLoggedModelsBox>> = {}) =>
    render(<RunViewLoggedModelsBox runInfo={runInfo} loggedModels={[]} loggedModelsV3={[]} {...props} />, {
      wrapper: ({ children }) => (
        <DesignSystemProvider>
          <IntlProvider locale="en">
            <TestRouter routes={[testRoute(<>{children}</>)]} />
          </IntlProvider>
        </DesignSystemProvider>
      ),
    });

  it('renders a "—" placeholder when there are no logged models', () => {
    renderTestComponent();
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
    expect(screen.getByText('—')).toBeInTheDocument();
  });

  it('renders logged model links when models are present', () => {
    renderTestComponent({
      loggedModels: [{ artifactPath: 'model', flavors: ['sklearn'], utcTimeCreated: 0 }],
    });
    expect(screen.queryByText('—')).not.toBeInTheDocument();
    expect(screen.getByText('sklearn')).toBeInTheDocument();
  });

  it('renders logged model V3 links when models are present', () => {
    const loggedModelsV3: LoggedModelProto[] = [{ info: { model_id: 'm-1', name: 'my-model' } }];
    renderTestComponent({ loggedModelsV3 });
    expect(screen.queryByText('—')).not.toBeInTheDocument();
    expect(screen.getByText('my-model')).toBeInTheDocument();
  });
});
