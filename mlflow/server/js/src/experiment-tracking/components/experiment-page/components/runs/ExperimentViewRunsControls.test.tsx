import { describe, expect, jest, test } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { render, screen } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import type { ReactNode } from 'react';
import { createExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import { createExperimentPageUIState } from '../../models/ExperimentPageUIState';
import { ExperimentPageViewState } from '../../models/ExperimentPageViewState';
import { ExperimentViewRunsControls } from './ExperimentViewRunsControls';

jest.mock('./ExperimentViewRunsControlsActions', () => ({
  ExperimentViewRunsControlsActions: () => <div data-testid="runs-actions" />,
}));

jest.mock('./ExperimentViewRunsControlsFilters', () => ({
  ExperimentViewRunsControlsFilters: ({ additionalControls }: { additionalControls: ReactNode }) => (
    <div data-testid="runs-filters">{additionalControls}</div>
  ),
}));

jest.mock('./ExperimentViewRunsColumnSelector', () => ({
  ExperimentViewRunsColumnSelector: () => <div data-testid="column-selector" />,
}));

jest.mock('./ExperimentViewRunsSortSelectorV2', () => ({
  ExperimentViewRunsSortSelectorV2: () => <div data-testid="sort-selector" />,
}));

jest.mock('./ExperimentViewRunsGroupBySelector', () => ({
  ExperimentViewRunsGroupBySelector: () => <div data-testid="group-by-selector" />,
}));

jest.mock('../../hooks/useExperimentPageViewMode', () => ({
  useExperimentPageViewMode: () => ['TABLE', jest.fn()],
}));

jest.mock('../../contexts/ExperimentPageUIStateContext', () => ({
  useUpdateExperimentViewUIState: () => jest.fn(),
}));

const runsData = {
  paramKeyList: [],
  metricKeyList: [],
  tagsList: [],
  datasetsList: [],
} as any;

const searchFacetsState = createExperimentPageSearchFacetsState();
const uiState = createExperimentPageUIState();

const renderComponent = (viewStateOverride: Partial<ExperimentPageViewState> = {}) => {
  const viewState = Object.assign(new ExperimentPageViewState(), viewStateOverride);

  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <ExperimentViewRunsControls
          runsData={runsData}
          viewState={viewState}
          updateViewState={jest.fn()}
          searchFacetsState={searchFacetsState}
          experimentId="123"
          requestError={null}
          expandRows={false}
          updateExpandRows={jest.fn()}
          refreshRuns={jest.fn()}
          uiState={uiState}
          isLoading={false}
          isComparingExperiments={false}
        />
      </DesignSystemProvider>
    </IntlProvider>,
  );
};

describe('ExperimentViewRunsControls', () => {
  test('shows selection action buttons when runs are selected and column selector is hidden', () => {
    renderComponent({
      runsSelected: { 'run-1': true },
      columnSelectorVisible: false,
    });

    expect(screen.getByTestId('runs-actions')).toBeInTheDocument();
    expect(screen.queryByTestId('runs-filters')).not.toBeInTheDocument();
  });

  test('shows filter controls and column selector when selector is visible, even with selected runs', () => {
    renderComponent({
      runsSelected: { 'run-1': true },
      columnSelectorVisible: true,
    });

    expect(screen.queryByTestId('runs-actions')).not.toBeInTheDocument();
    expect(screen.getByTestId('runs-filters')).toBeInTheDocument();
    expect(screen.getByTestId('column-selector')).toBeInTheDocument();
  });
});
