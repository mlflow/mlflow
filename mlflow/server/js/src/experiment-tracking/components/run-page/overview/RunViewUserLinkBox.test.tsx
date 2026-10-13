import { describe, it, expect } from '@jest/globals';
import { IntlProvider } from 'react-intl';
import { render, screen } from '../../../../common/utils/TestUtils.react18';
import type { KeyValueEntity } from '../../../../common/types';
import type { RunInfoEntity } from '../../../types';
import { RunViewUserLinkBox } from './RunViewUserLinkBox';
import { DesignSystemProvider } from '@databricks/design-system';
import { TestRouter, testRoute } from '../../../../common/utils/RoutingTestUtils';

describe('RunViewUserLinkBox', () => {
  const baseRunInfo: RunInfoEntity = {
    runUuid: 'run-id',
    experimentId: 'experiment-id',
    startTime: 0,
    endTime: 1,
    artifactUri: '',
    lifecycleStage: 'active',
    status: 'FINISHED',
    runName: 'run-name',
  };

  const renderTestComponent = (runInfo: RunInfoEntity, tags: Record<string, KeyValueEntity>) =>
    render(<RunViewUserLinkBox runInfo={runInfo} tags={tags} />, {
      wrapper: ({ children }) => (
        <DesignSystemProvider>
          <IntlProvider locale="en">
            <TestRouter routes={[testRoute(<>{children}</>)]} />
          </IntlProvider>
        </DesignSystemProvider>
      ),
    });

  it('renders a link with the user from the mlflow.user tag', () => {
    renderTestComponent(baseRunInfo, {
      'mlflow.user': { key: 'mlflow.user', value: 'alice' } as KeyValueEntity,
    });
    expect(screen.getByRole('link', { name: 'alice' })).toBeInTheDocument();
  });

  it('renders a "—" placeholder when the user is absent', () => {
    renderTestComponent(baseRunInfo, {});
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
    expect(screen.getByText('—')).toBeInTheDocument();
  });
});
