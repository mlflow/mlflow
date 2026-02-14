import { describe, test, expect, it, beforeAll, afterAll, jest } from '@jest/globals';
import React from 'react';
import { ErrorView } from './ErrorView';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { MemoryRouter } from '../utils/RoutingUtils';
import { setActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';
import { getWorkspacesEnabledSync } from '../../experiment-tracking/hooks/useServerInfo';

jest.mock('../../experiment-tracking/hooks/useServerInfo', () => ({
  ...jest.requireActual<typeof import('../../experiment-tracking/hooks/useServerInfo')>(
    '../../experiment-tracking/hooks/useServerInfo',
  ),
  getWorkspacesEnabledSync: jest.fn(),
}));

const getWorkspacesEnabledSyncMock = jest.mocked(getWorkspacesEnabledSync);

const TEST_WORKSPACE = 'test-workspace';

describe('ErrorView', () => {
  // With query param routing, workspace is added as a query param
  const workspacePrefixed = (path: string) => `${path}?workspace=${TEST_WORKSPACE}`;

  beforeAll(() => {
    getWorkspacesEnabledSyncMock.mockReturnValue(true);
    setActiveWorkspace(TEST_WORKSPACE);
  });

  afterAll(() => {
    jest.restoreAllMocks();
    setActiveWorkspace(null);
  });

  test('should render 400', () => {
    renderWithIntl(
      <MemoryRouter>
        <ErrorView statusCode={400} fallbackHomePageReactRoute="/path/to" />
      </MemoryRouter>,
    );

    const errorImage = screen.getByRole('img');
    expect(errorImage).toBeInTheDocument();
    expect(errorImage).toHaveAttribute('alt', '400 Bad Request');

    const title = screen.getByRole('heading', { level: 1 });
    expect(title).toBeInTheDocument();
    expect(title).toHaveTextContent('Bad Request');

    const subtitle = screen.getByRole('heading', { level: 2 });
    expect(subtitle).toBeInTheDocument();
    expect(subtitle).toHaveTextContent('Go back to');

    const link = screen.getByRole('link');
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute('href', workspacePrefixed('/path/to'));
  });

  it('should render 404', () => {
    renderWithIntl(
      <MemoryRouter>
        <ErrorView statusCode={404} fallbackHomePageReactRoute="/path/to" />
      </MemoryRouter>,
    );

    const errorImage = screen.getByRole('img');
    expect(errorImage).toBeInTheDocument();
    expect(errorImage).toHaveAttribute('alt', '404 Not Found');

    const title = screen.getByRole('heading', { level: 1 });
    expect(title).toBeInTheDocument();
    expect(title).toHaveTextContent('Page Not Found');

    const subtitle = screen.getByRole('heading', { level: 2 });
    expect(subtitle).toBeInTheDocument();
    expect(subtitle).toHaveTextContent('Go back to');

    const link = screen.getByRole('link');
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute('href', workspacePrefixed('/path/to'));
  });

  test('should render 404 with sub message', () => {
    renderWithIntl(
      <MemoryRouter>
        <ErrorView statusCode={404} fallbackHomePageReactRoute="/path/to" subMessage="sub message" />
      </MemoryRouter>,
    );

    const errorImage = screen.getByRole('img');
    expect(errorImage).toBeInTheDocument();
    expect(errorImage).toHaveAttribute('alt', '404 Not Found');

    const title = screen.getByRole('heading', { level: 1 });
    expect(title).toBeInTheDocument();
    expect(title).toHaveTextContent('Page Not Found');

    const subtitle = screen.getByRole('heading', { level: 2 });
    expect(subtitle).toBeInTheDocument();
    expect(subtitle).toHaveTextContent('sub message, go back to ');

    const link = screen.getByRole('link');
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute('href', workspacePrefixed('/path/to'));
  });
});
