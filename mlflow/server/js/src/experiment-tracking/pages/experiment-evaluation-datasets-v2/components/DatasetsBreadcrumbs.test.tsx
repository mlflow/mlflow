import { describe, expect, test } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { MemoryRouter } from '../../../../common/utils/RoutingUtils';
import { DatasetsBreadcrumbs } from './DatasetsBreadcrumbs';

const dataset = { dataset_id: 'd-123', create_time: '2026-01-01T00:00:00Z', name: 'Friendly dataset' };

const renderBreadcrumbs = (hideTrail?: boolean) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <MemoryRouter>
          <DatasetsBreadcrumbs
            experimentId="1"
            datasetName={dataset.name}
            dataset={dataset}
            rightActions={<button>Dataset actions</button>}
            {...(hideTrail === undefined ? {} : { hideTrail })}
          />
        </MemoryRouter>
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('DatasetsBreadcrumbs', () => {
  test('shows the datasets trail by default', () => {
    renderBreadcrumbs();

    const nav = screen.getByRole('navigation', { name: 'Breadcrumb' });
    expect(nav).toHaveTextContent('Datasets');
    expect(nav).toHaveTextContent('Friendly dataset');
    expect(screen.getByRole('button', { name: 'Dataset actions' })).toBeInTheDocument();
    expect(screen.getByText(/d-123/)).toBeInTheDocument();
  });

  test('hideTrail leaves out the trail but keeps the actions and the dataset ID', () => {
    renderBreadcrumbs(true);

    expect(screen.queryByRole('navigation')).not.toBeInTheDocument();
    expect(screen.queryByText('Friendly dataset')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Dataset actions' })).toBeInTheDocument();
    expect(screen.getByText(/d-123/)).toBeInTheDocument();
  });
});
