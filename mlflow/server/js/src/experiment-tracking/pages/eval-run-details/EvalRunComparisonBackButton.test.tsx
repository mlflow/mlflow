import { describe, expect, test } from '@jest/globals';
import { useEffect } from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { useLocation } from '../../../common/utils/RoutingUtils';
import {
  setupTestRouter,
  testRoute,
  TestRouter,
  waitForRoutesToBeRendered,
} from '../../../common/utils/RoutingTestUtils';
import { EvalRunComparisonBackButton } from './EvalRunComparisonBackButton';

describe('EvalRunComparisonBackButton', () => {
  const { history } = setupTestRouter();
  let lastSearch = '';
  const LocationSpy = () => {
    const search = useLocation().search;
    useEffect(() => {
      lastSearch = search;
    }, [search]);
    return null;
  };

  test('exits comparison when opened from the evaluation runs list', async () => {
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <TestRouter
            history={history}
            routes={[
              testRoute(
                <>
                  <LocationSpy />
                  <EvalRunComparisonBackButton />
                </>,
              ),
            ]}
            initialEntries={['/evaluation-runs/run-1/details?compare=true&compareFrom=runs&compareRuns=run-2']}
          />
        </DesignSystemProvider>
      </IntlProvider>,
    );
    await waitForRoutesToBeRendered();

    await userEvent.click(screen.getByRole('button', { name: 'Back to Reference Run' }));

    await waitFor(() => {
      const params = new URLSearchParams(lastSearch);
      expect(params.has('compare')).toBe(false);
      expect(params.get('compareRuns')).toBe('run-2');
    });
  });
});
