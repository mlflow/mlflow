import React from 'react';
import { IntlProvider } from 'react-intl';
import { render, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { setupTestRouter, testRoute, TestRouter } from '../../common/utils/RoutingTestUtils';
import { RunNotFoundView } from './RunNotFoundView';

describe('RunNotFoundView', () => {
  const { history } = setupTestRouter();
  const mockRunId = 'This is a mock run ID';
  const minimalProps: any = { runId: mockRunId };

  const renderComponent = (props: any) => {
    return render(<RunNotFoundView {...props} />, {
      wrapper: ({ children }) => (
        <TestRouter
          history={history}
          initialEntries={['/test']}
          routes={[testRoute(<IntlProvider locale="en">{children}</IntlProvider>, '/test')]}
        />
      ),
    });
  };

  test('should render with minimal props without exploding', async () => {
    renderComponent(minimalProps);
    await waitFor(() => {
      expect(
        screen.getByRole('heading', {
          name: /run id this is a mock run id does not exist, go back to the home page\./i,
        }),
      ).toBeInTheDocument();
    });
  });
});
