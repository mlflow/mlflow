import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import { render, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DropdownMenu, DesignSystemProvider } from '@databricks/design-system';

import { SavedViewsMenu, type SavedViewMenuItem } from './SavedViewsMenu';

const views: SavedViewMenuItem[] = [
  { id: 'v1', name: 'GPU runs', createdAt: 1000 },
  { id: 'v2', name: 'Baseline comparison', createdAt: 2000 },
];

const onOpen = jest.fn();
const onCopyLink = jest.fn();
const onRequestDelete = jest.fn();
const onSaveCurrent = jest.fn();

const renderMenu = (overrides: Partial<React.ComponentProps<typeof SavedViewsMenu>> = {}) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <DropdownMenu.Root open>
          <DropdownMenu.Trigger>trigger</DropdownMenu.Trigger>
          <DropdownMenu.Content>
            <SavedViewsMenu
              componentId="test.saved_views"
              testIdPrefix="test-saved-views"
              views={views}
              canModify
              onOpen={onOpen}
              onCopyLink={onCopyLink}
              onRequestDelete={onRequestDelete}
              onSaveCurrent={onSaveCurrent}
              {...overrides}
            />
          </DropdownMenu.Content>
        </DropdownMenu.Root>
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('SavedViewsMenu', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders the saved view list', () => {
    renderMenu();
    expect(screen.getByText('GPU runs')).toBeInTheDocument();
    expect(screen.getByText('Baseline comparison')).toBeInTheDocument();
  });

  test('filters the list by search text', async () => {
    renderMenu();
    await userEvent.type(screen.getByTestId('test-saved-views-search'), 'baseline');
    await waitFor(() => {
      expect(screen.queryByText('GPU runs')).not.toBeInTheDocument();
      expect(screen.getByText('Baseline comparison')).toBeInTheDocument();
    });
  });

  test('clicking a view invokes onOpen', async () => {
    renderMenu();
    await userEvent.click(screen.getByText('GPU runs'));
    expect(onOpen).toHaveBeenCalledWith('v1');
  });

  test('clicking a row copy-link invokes onCopyLink with the view', async () => {
    renderMenu();
    await userEvent.click(screen.getByTestId('test-saved-views-copy-link-v1'));
    expect(onCopyLink).toHaveBeenCalledWith(views[0]);
    expect(onOpen).not.toHaveBeenCalled();
  });

  test('clicking a row delete invokes onRequestDelete with the view', async () => {
    renderMenu();
    await userEvent.click(screen.getByTestId('test-saved-views-delete-v1'));
    expect(onRequestDelete).toHaveBeenCalledWith(views[0]);
    expect(onOpen).not.toHaveBeenCalled();
  });

  test('clicking save-current invokes onSaveCurrent', async () => {
    renderMenu();
    await userEvent.click(screen.getByTestId('test-saved-views-save-current'));
    expect(onSaveCurrent).toHaveBeenCalledTimes(1);
  });

  test('hides delete and save-current affordances when canModify is false', () => {
    renderMenu({ canModify: false });
    expect(screen.getByText('GPU runs')).toBeInTheDocument();
    expect(screen.queryByTestId('test-saved-views-delete-v1')).not.toBeInTheDocument();
    expect(screen.queryByTestId('test-saved-views-save-current')).not.toBeInTheDocument();
    // Copy-link stays available to read-only users.
    expect(screen.getByTestId('test-saved-views-copy-link-v1')).toBeInTheDocument();
  });

  test('read-only copy-link still stops propagation (does not open the view)', async () => {
    renderMenu({ canModify: false });
    await userEvent.click(screen.getByTestId('test-saved-views-copy-link-v1'));
    expect(onCopyLink).toHaveBeenCalledWith(views[0]);
    expect(onOpen).not.toHaveBeenCalled();
  });

  test('shows an empty state when there are no saved views', () => {
    renderMenu({ views: [] });
    expect(screen.getByText(/No saved views yet/)).toBeInTheDocument();
  });

  test('shows a no-matches state when the search filters everything out', async () => {
    renderMenu();
    await userEvent.type(screen.getByTestId('test-saved-views-search'), 'zzzz-no-match');
    await waitFor(() => expect(screen.getByText(/No views match your search/)).toBeInTheDocument());
  });
});
