import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import { render, screen, waitFor, within } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';

import { ExperimentViewSavedViewsButton } from './ExperimentViewSavedViewsButton';
import { MockedReduxStoreProvider } from '../../../../../common/utils/TestUtils';
import { setupTestRouter, testRoute, TestRouter } from '../../../../../common/utils/RoutingTestUtils';
import { useSavedViews } from '../../hooks/useSavedViews';
import type { ExperimentEntity } from '../../../../types';
import type { SavedViewSummary } from '../../utils/savedViewEnvelope';
import { createExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import { createExperimentPageUIState } from '../../models/ExperimentPageUIState';

jest.mock('../../hooks/useSavedViews', () => ({
  useSavedViews: jest.fn(),
}));

const experiment = { experimentId: 'exp-1', tags: [] } as unknown as ExperimentEntity;

const sampleViews: SavedViewSummary[] = [
  { id: 'v1', name: 'GPU runs', createdAt: 1000 },
  { id: 'v2', name: 'Baseline comparison', createdAt: 2000 },
];

const deleteView = jest.fn();
const openView = jest.fn();

const mockUseSavedViews = (overrides: Partial<ReturnType<typeof useSavedViews>> = {}) => {
  jest.mocked(useSavedViews).mockReturnValue({
    views: sampleViews,
    canModify: true,
    deleteView: deleteView as any,
    openView: openView as any,
    activeViewId: null,
    ...overrides,
  });
};

const { history } = setupTestRouter();

const renderButton = () =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <MockedReduxStoreProvider>
          <ExperimentViewSavedViewsButton
            experiment={experiment}
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

describe('ExperimentViewSavedViewsButton', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockUseSavedViews();
  });

  const openDropdown = async () => {
    await userEvent.click(screen.getByTestId('saved-views-trigger'));
  };

  test('renders the saved view list', async () => {
    renderButton();
    await openDropdown();

    expect(screen.getByText('GPU runs')).toBeInTheDocument();
    expect(screen.getByText('Baseline comparison')).toBeInTheDocument();
  });

  test('filters the list by search text', async () => {
    renderButton();
    await openDropdown();

    await userEvent.type(screen.getByTestId('saved-views-search'), 'baseline');
    await waitFor(() => {
      expect(screen.queryByText('GPU runs')).not.toBeInTheDocument();
      expect(screen.getByText('Baseline comparison')).toBeInTheDocument();
    });
  });

  test('clicking a view opens it', async () => {
    renderButton();
    await openDropdown();

    await userEvent.click(screen.getByText('GPU runs'));
    expect(openView).toHaveBeenCalledWith('v1');
  });

  test('deleting a view requires confirmation then dispatches delete', async () => {
    renderButton();
    await openDropdown();

    await userEvent.click(screen.getByTestId('saved-views-delete-v1'));
    // Confirmation modal shows; only on confirm does the delete fire.
    expect(deleteView).not.toHaveBeenCalled();
    const confirmButton = await screen.findByText('Delete');
    // The DangerModal footer sets pointer-events during its enter transition; bypass the check.
    await userEvent.click(confirmButton, { pointerEventsCheck: 0 });
    expect(deleteView).toHaveBeenCalledWith('v1');
  });

  test('hides Save and Delete affordances for a read-only user', async () => {
    mockUseSavedViews({ canModify: false });
    renderButton();
    await openDropdown();

    // The list and copy-link are still available...
    expect(screen.getByText('GPU runs')).toBeInTheDocument();
    // ...but authoring affordances are absent.
    expect(screen.queryByTestId('saved-views-delete-v1')).not.toBeInTheDocument();
    expect(screen.queryByTestId('saved-views-save-current')).not.toBeInTheDocument();
  });

  test('shows an empty state when there are no saved views', async () => {
    mockUseSavedViews({ views: [] });
    renderButton();
    await openDropdown();

    expect(screen.getByText(/No saved views yet/)).toBeInTheDocument();
  });

  test('shows a no-matches state when the search filters everything out', async () => {
    renderButton();
    await openDropdown();

    await userEvent.type(screen.getByTestId('saved-views-search'), 'zzzz-no-match');
    await waitFor(() => expect(screen.getByText(/No views match your search/)).toBeInTheDocument());
  });

  test('labels the trigger with the active view name and checkmarks its row', async () => {
    mockUseSavedViews({ activeViewId: 'v2' });
    renderButton();

    // The trigger reflects the applied view rather than the generic "Views" label.
    const trigger = screen.getByTestId('saved-views-trigger');
    expect(within(trigger).getByText('Baseline comparison')).toBeInTheDocument();
    expect(within(trigger).queryByText('Views')).not.toBeInTheDocument();

    await openDropdown();
    // Only the active view's row carries the checkmark.
    expect(screen.getByTestId('saved-views-active-v2')).toBeInTheDocument();
    expect(screen.queryByTestId('saved-views-active-v1')).not.toBeInTheDocument();
  });

  test('falls back to the generic label when no view is active', async () => {
    renderButton();

    const trigger = screen.getByTestId('saved-views-trigger');
    expect(within(trigger).getByText('Views')).toBeInTheDocument();
  });
});
