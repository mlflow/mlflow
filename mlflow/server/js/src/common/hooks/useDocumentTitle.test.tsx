import React from 'react';
import { describe, it, expect, beforeEach } from '@jest/globals';
import { render, waitFor } from '@testing-library/react';
import { useDocumentTitle } from './useDocumentTitle';
import type { DocumentTitleHandle } from './useDocumentTitle';
import { createMemoryRouter, RouterProvider, Outlet } from '../utils/RoutingUtils';

/**
 * Test component that uses the useDocumentTitle hook
 */
const DocumentTitleConsumer = ({ children }: { children?: React.ReactNode }) => {
  useDocumentTitle();
  return <div data-testid="content">{children}</div>;
};

/**
 * Helper to create a router with routes and render it
 */
const renderWithRouter = (routes: any[], initialEntries: string[] = ['/']) => {
  const router = createMemoryRouter(
    [
      {
        path: '/',
        element: (
          <DocumentTitleConsumer>
            <Outlet />
          </DocumentTitleConsumer>
        ),
        children: routes,
      },
    ],
    { initialEntries },
  );

  return render(<RouterProvider router={router} />);
};

describe('useDocumentTitle', () => {
  beforeEach(() => {
    // Reset document title before each test
    document.title = '';
  });

  it('sets document title from route handle getPageTitle function', async () => {
    renderWithRouter(
      [
        {
          path: '/',
          element: <div>Home</div>,
          handle: { getPageTitle: () => 'Test Page' } satisfies DocumentTitleHandle,
        },
      ],
      ['/'],
    );

    await waitFor(() => {
      expect(document.title).toBe('Test Page - MLflow');
    });
  });

  it('passes route params to getPageTitle function', async () => {
    renderWithRouter(
      [
        {
          path: 'experiments/:experimentId',
          element: <div>Experiment</div>,
          handle: {
            getPageTitle: (params) => `Experiment ${params['experimentId']}`,
          } satisfies DocumentTitleHandle,
        },
      ],
      ['/experiments/123'],
    );

    await waitFor(() => {
      expect(document.title).toBe('Experiment 123 - MLflow');
    });
  });

  it('uses the last matching route title when multiple routes have getPageTitle', async () => {
    renderWithRouter(
      [
        {
          path: 'parent',
          element: (
            <div>
              Parent
              <Outlet />
            </div>
          ),
          handle: { getPageTitle: () => 'Parent' } satisfies DocumentTitleHandle,
          children: [
            {
              path: 'child',
              element: <div>Child</div>,
              handle: { getPageTitle: () => 'Child' } satisfies DocumentTitleHandle,
            },
          ],
        },
      ],
      ['/parent/child'],
    );

    await waitFor(() => {
      // Should use the most specific (last) route title
      expect(document.title).toBe('Child - MLflow');
    });
  });

  it('falls back to "MLflow" when no route has a getPageTitle function', async () => {
    renderWithRouter(
      [
        {
          path: '/',
          element: <div>No Title</div>,
        },
      ],
      ['/'],
    );

    await waitFor(() => {
      expect(document.title).toBe('MLflow');
    });
  });

  it('falls back to "MLflow" when routes have handles but no getPageTitle function', async () => {
    renderWithRouter(
      [
        {
          path: '/',
          element: <div>Other Handle</div>,
          handle: { someOtherProperty: 'value' },
        },
      ],
      ['/'],
    );

    await waitFor(() => {
      expect(document.title).toBe('MLflow');
    });
  });

  it('handles AI Gateway routes correctly', async () => {
    renderWithRouter(
      [
        {
          path: 'gateway',
          element: <div>Gateway</div>,
          handle: { getPageTitle: () => 'AI Gateway' } satisfies DocumentTitleHandle,
        },
      ],
      ['/gateway'],
    );

    await waitFor(() => {
      expect(document.title).toBe('AI Gateway - MLflow');
    });
  });

  it('handles nested gateway routes correctly', async () => {
    renderWithRouter(
      [
        {
          path: 'gateway',
          element: (
            <div>
              Gateway
              <Outlet />
            </div>
          ),
          handle: { getPageTitle: () => 'AI Gateway' } satisfies DocumentTitleHandle,
          children: [
            {
              path: 'api-keys',
              element: <div>API Keys</div>,
              handle: { getPageTitle: () => 'API Keys' } satisfies DocumentTitleHandle,
            },
          ],
        },
      ],
      ['/gateway/api-keys'],
    );

    await waitFor(() => {
      expect(document.title).toBe('API Keys - MLflow');
    });
  });

  it('handles endpoint details page with params correctly', async () => {
    renderWithRouter(
      [
        {
          path: 'gateway',
          element: (
            <div>
              Gateway
              <Outlet />
            </div>
          ),
          handle: { getPageTitle: () => 'AI Gateway' } satisfies DocumentTitleHandle,
          children: [
            {
              path: 'endpoints/:endpointId',
              element: <div>Endpoint Details</div>,
              handle: {
                getPageTitle: (params) => `Endpoint ${params['endpointId']}`,
              } satisfies DocumentTitleHandle,
            },
          ],
        },
      ],
      ['/gateway/endpoints/test-123'],
    );

    await waitFor(() => {
      expect(document.title).toBe('Endpoint test-123 - MLflow');
    });
  });

  it('handles experiment page with experimentId param', async () => {
    renderWithRouter(
      [
        {
          path: 'experiments/:experimentId',
          element: <div>Experiment</div>,
          handle: {
            getPageTitle: (params) => `Experiment ${params['experimentId']}`,
          } satisfies DocumentTitleHandle,
        },
      ],
      ['/experiments/456'],
    );

    await waitFor(() => {
      expect(document.title).toBe('Experiment 456 - MLflow');
    });
  });

  it('handles run page with multiple params', async () => {
    renderWithRouter(
      [
        {
          path: 'experiments/:experimentId/runs/:runUuid',
          element: <div>Run</div>,
          handle: {
            getPageTitle: (params) => `Run ${params['runUuid']}`,
          } satisfies DocumentTitleHandle,
        },
      ],
      ['/experiments/123/runs/abc-def'],
    );

    await waitFor(() => {
      expect(document.title).toBe('Run abc-def - MLflow');
    });
  });

  it('handles model version page with multiple params', async () => {
    renderWithRouter(
      [
        {
          path: 'models/:modelName/versions/:version',
          element: <div>Model Version</div>,
          handle: {
            getPageTitle: (params) => `${params['modelName']} v${params['version']}`,
          } satisfies DocumentTitleHandle,
        },
      ],
      ['/models/my-model/versions/3'],
    );

    await waitFor(() => {
      expect(document.title).toBe('my-model v3 - MLflow');
    });
  });
});
