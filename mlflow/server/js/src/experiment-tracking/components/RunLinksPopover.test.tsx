import { describe, beforeEach, jest, test, expect } from '@jest/globals';
import React from 'react';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { renderWithDesignSystem, screen } from '../../common/utils/TestUtils.react18';
import { RunLinksPopover } from './RunLinksPopover';
import Routes from '../routes';
import { prefixRouteWithWorkspace } from '../../workspaces/utils/WorkspaceUtils';

describe('RunLinksPopover', () => {
  let minimalProps: any;

  beforeEach(() => {
    minimalProps = {
      experimentId: '0',
      visible: false,
      x: 0,
      y: 0,
      runItems: [],
      handleClose: jest.fn(),
      handleKeyDown: jest.fn(),
      handleVisibleChange: jest.fn(),
    };
  });

  test('renders a link for each run item pointing at its run page', () => {
    const runItems = [
      { runId: 'runUuid1', name: 'run1', color: 'rgb(1, 1, 1)', y: 0.1 },
      { runId: 'runUuid2', name: 'run2', color: 'rgb(2, 2, 2)', y: 0.2 },
    ];
    renderWithDesignSystem(
      <MemoryRouter>
        <RunLinksPopover {...minimalProps} visible runItems={runItems} />
      </MemoryRouter>,
    );

    runItems.forEach(({ runId, name, y }) => {
      const expectedHref = prefixRouteWithWorkspace(Routes.getRunPageRoute(minimalProps.experimentId, runId));
      const link = screen.getByRole('link', { name: `${name}, ${y}` });
      expect(link).toHaveAttribute('href', expectedHref);
    });
  });
});
