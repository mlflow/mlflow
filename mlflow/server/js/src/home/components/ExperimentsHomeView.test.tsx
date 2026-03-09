import { describe, expect, it, jest } from '@jest/globals';
import React from 'react';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import type { ExperimentEntity } from '../../experiment-tracking/types';
import { ExperimentsHomeView } from './ExperimentsHomeView';
import { MemoryRouter } from '../../common/utils/RoutingUtils';

const mockShowEditExperimentTagsModal = jest.fn();

jest.mock('../../experiment-tracking/components/ExperimentListTable', () => ({
  ExperimentListTable: ({
    experiments,
    onEditTags,
  }: {
    experiments: ExperimentEntity[];
    onEditTags: (experiment: ExperimentEntity) => void;
  }) => (
    <div data-testid="experiment-list-table">
      {`rows:${experiments.length}`}
      {experiments.length > 0 && (
        <button data-testid="edit-tags-button" onClick={() => onEditTags(experiments[0])}>
          Edit tags
        </button>
      )}
    </div>
  ),
}));

jest.mock('../../experiment-tracking/components/experiment-page/hooks/useUpdateExperimentTags', () => ({
  useUpdateExperimentTags: () => ({
    EditTagsModal: null,
    showEditExperimentTagsModal: mockShowEditExperimentTagsModal,
    isLoading: false,
  }),
}));

const renderWithRouter = (ui: React.ReactElement) => renderWithDesignSystem(<MemoryRouter>{ui}</MemoryRouter>);

describe('ExperimentsHomeView', () => {
  beforeEach(() => {
    mockShowEditExperimentTagsModal.mockClear();
  });

  const sampleExperiment: ExperimentEntity = {
    allowedActions: [],
    artifactLocation: 'dbfs:/experiment',
    creationTime: 1,
    experimentId: '1',
    lastUpdateTime: 2,
    lifecycleStage: 'active',
    name: 'Test experiment',
    tags: [],
  };

  it('shows empty state when there are no experiments', async () => {
    const onCreateExperiment = jest.fn();

    renderWithRouter(
      <ExperimentsHomeView
        experiments={[]}
        isLoading={false}
        error={null}
        onCreateExperiment={onCreateExperiment}
        onRetry={jest.fn()}
      />,
    );

    expect(screen.getByRole('heading', { level: 3, name: 'Recent Experiments' })).toBeInTheDocument();
    expect(screen.getByText('Create your first experiment')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'View all' })).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Create experiment' }));
    expect(onCreateExperiment).toHaveBeenCalled();
  });

  it('renders error state and allows retry', async () => {
    const onRetry = jest.fn();
    const error = new Error('Boom');

    renderWithRouter(
      <ExperimentsHomeView
        experiments={[sampleExperiment]}
        isLoading={false}
        error={error}
        onCreateExperiment={jest.fn()}
        onRetry={onRetry}
      />,
    );

    expect(screen.getByText("We couldn't load your experiments.")).toBeInTheDocument();
    expect(screen.getByText('Boom')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(onRetry).toHaveBeenCalled();
  });

  it('calls showEditExperimentTagsModal when editing tags', async () => {
    renderWithRouter(
      <ExperimentsHomeView
        experiments={[sampleExperiment]}
        isLoading={false}
        error={null}
        onCreateExperiment={jest.fn()}
        onRetry={jest.fn()}
      />,
    );

    await userEvent.click(screen.getByTestId('edit-tags-button'));
    expect(mockShowEditExperimentTagsModal).toHaveBeenCalledWith(sampleExperiment);
  });

  it('renders experiment preview when data is available', () => {
    renderWithRouter(
      <ExperimentsHomeView
        experiments={[sampleExperiment]}
        isLoading={false}
        error={null}
        onCreateExperiment={jest.fn()}
        onRetry={jest.fn()}
      />,
    );

    expect(screen.getByTestId('experiment-list-table')).toHaveTextContent('rows:1');
  });
});
