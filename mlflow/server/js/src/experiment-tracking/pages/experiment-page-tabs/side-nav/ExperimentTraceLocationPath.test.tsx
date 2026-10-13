import { describe, jest, test, expect } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { ExperimentTraceLocationPath } from './ExperimentTraceLocationPath';
import { MLFLOW_EXPERIMENT_TRACE_STORAGE_UC_SCHEMA_TAG } from '../../../constants';

jest.mock('../../../../common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('../../../../common/utils/RoutingUtils')>(
    '../../../../common/utils/RoutingUtils',
  ),
  useParams: () => ({ experimentId: 'test-experiment-123' }),
}));

const mockUseGetExperimentQuery = jest.fn<() => { data: any; loading: boolean }>();
jest.mock('../../../hooks/useExperimentQuery', () => ({
  useGetExperimentQuery: (...args: any[]) => mockUseGetExperimentQuery(),
}));

const renderComponent = () => {
  return render(
    <DesignSystemProvider>
      <IntlProvider locale="en">
        <ExperimentTraceLocationPath />
      </IntlProvider>
    </DesignSystemProvider>,
  );
};

const makeMockExperimentData = (tags: { key: string; value: string }[]) => {
  return {
    data: {
      tags: tags,
    },
  } as any;
};

describe('ExperimentTraceLocationPath', () => {
  test('renders full path for table prefix experiment (3-part tag)', () => {
    const mockExperimentData = makeMockExperimentData([
      { key: MLFLOW_EXPERIMENT_TRACE_STORAGE_UC_SCHEMA_TAG, value: 'my_catalog.my_schema.my_prefix' },
    ]);
    mockUseGetExperimentQuery.mockReturnValue(mockExperimentData);

    renderComponent();
    expect(screen.getByText('my_catalog.my_schema.my_prefix')).toBeInTheDocument();
  });

  test('renders path for UC schema experiment (2-part tag)', () => {
    const mockExperimentData = makeMockExperimentData([
      { key: MLFLOW_EXPERIMENT_TRACE_STORAGE_UC_SCHEMA_TAG, value: 'my_catalog.my_schema' },
    ]);
    mockUseGetExperimentQuery.mockReturnValue(mockExperimentData);

    renderComponent();
    expect(screen.getByText('my_catalog.my_schema')).toBeInTheDocument();
  });

  test('opens catalog explorer for UC schema path on click', () => {
    const mockOpen = jest.fn<typeof window.open>();
    const originalOpen = window.open;
    window.open = mockOpen;

    try {
      const mockExperimentData = makeMockExperimentData([
        { key: MLFLOW_EXPERIMENT_TRACE_STORAGE_UC_SCHEMA_TAG, value: 'my_catalog.my_schema' },
      ]);
      mockUseGetExperimentQuery.mockReturnValue(mockExperimentData);

      renderComponent();
      screen.getByText('my_catalog.my_schema').click();

      expect(mockOpen).toHaveBeenCalledWith('/explore/data/my_catalog/my_schema', '_blank', 'noopener,noreferrer');
    } finally {
      window.open = originalOpen;
    }
  });

  test('shows tooltip on hover with the full UC schema path', async () => {
    const path = 'my_catalog.my_schema';
    const mockExperimentData = makeMockExperimentData([
      { key: MLFLOW_EXPERIMENT_TRACE_STORAGE_UC_SCHEMA_TAG, value: path },
    ]);
    mockUseGetExperimentQuery.mockReturnValue(mockExperimentData);

    renderComponent();
    await userEvent.hover(screen.getByText(path));

    await waitFor(() => {
      expect(screen.getByRole('tooltip')).toHaveTextContent(path);
    });
  });

  test('renders nothing when no destination path tag', () => {
    const mockExperimentData = makeMockExperimentData([]);
    mockUseGetExperimentQuery.mockReturnValue(mockExperimentData);

    const { container } = renderComponent();
    expect(container.innerHTML).toBe('');
  });

  test('renders nothing when experiment is loading', () => {
    mockUseGetExperimentQuery.mockReturnValue({
      data: undefined,
      loading: true,
    });

    const { container } = renderComponent();
    expect(container.innerHTML).toBe('');
  });

  test('shows tooltip on hover with the full destination path', async () => {
    const path = 'my_catalog.my_schema.my_prefix';
    const mockExperimentData = makeMockExperimentData([
      { key: MLFLOW_EXPERIMENT_TRACE_STORAGE_UC_SCHEMA_TAG, value: path },
    ]);
    mockUseGetExperimentQuery.mockReturnValue(mockExperimentData);

    renderComponent();
    const pathElement = screen.getByText(path);
    await userEvent.hover(pathElement);

    await waitFor(() => {
      expect(screen.getByRole('tooltip')).toHaveTextContent(path);
    });
  });
});
