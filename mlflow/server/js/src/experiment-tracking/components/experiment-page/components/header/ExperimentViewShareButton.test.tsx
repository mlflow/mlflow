import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import { render, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';

import { ExperimentViewShareButton } from './ExperimentViewShareButton';
import { MockedReduxStoreProvider } from '../../../../../common/utils/TestUtils';
import { setupTestRouter, testRoute, TestRouter } from '../../../../../common/utils/RoutingTestUtils';
import { createExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import { createExperimentPageUIState } from '../../models/ExperimentPageUIState';

const { history } = setupTestRouter();

const renderButton = () =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <MockedReduxStoreProvider>
          <ExperimentViewShareButton
            experimentId="exp-1"
            searchFacetsState={createExperimentPageSearchFacetsState()}
            uiState={createExperimentPageUIState()}
          />
        </MockedReduxStoreProvider>
      </DesignSystemProvider>
    </IntlProvider>,
    {
      wrapper: ({ children }) => (
        <TestRouter routes={[testRoute(<>{children}</>, '/')]} history={history} initialEntries={['/']} />
      ),
    },
  );

describe('ExperimentViewShareButton', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('opens the Save & share view modal (name-first) rather than copying an anonymous link', async () => {
    renderButton();

    // No modal until the button is clicked.
    expect(screen.queryByTestId('save-view-name-input')).not.toBeInTheDocument();

    await userEvent.click(screen.getByTestId('experiment-share-button'));

    // Sharing routes through the named-view flow: the modal prompts for a view name before saving.
    expect(await screen.findByTestId('save-view-name-input')).toBeInTheDocument();
  });
});
