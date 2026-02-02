import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import React from 'react';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { UsageTrackingConfigurator } from './UsageTrackingConfigurator';

const mockExperiments = [
  { experimentId: 'exp-1', name: 'Experiment 1' },
  { experimentId: 'exp-2', name: 'Experiment 2' },
];

const mockUseExperimentsForSelect = jest.fn();

jest.mock('../../hooks/useExperimentsForSelect', () => ({
  useExperimentsForSelect: () => mockUseExperimentsForSelect(),
}));

describe('UsageTrackingConfigurator', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockUseExperimentsForSelect.mockReturnValue({
      experiments: mockExperiments,
      isLoading: false,
      error: null,
    });
  });

  test('renders toggle in off state by default', () => {
    const onChange = jest.fn();
    const onExperimentIdChange = jest.fn();

    renderWithDesignSystem(
      <UsageTrackingConfigurator
        value={false}
        onChange={onChange}
        experimentId=""
        onExperimentIdChange={onExperimentIdChange}
      />,
    );

    expect(screen.getByText('Enable usage tracking')).toBeInTheDocument();
    expect(screen.getByRole('switch')).not.toBeChecked();
    expect(screen.queryByText('Experiment')).not.toBeInTheDocument();
  });

  test('shows experiment selector when toggle is on', () => {
    const onChange = jest.fn();
    const onExperimentIdChange = jest.fn();

    renderWithDesignSystem(
      <UsageTrackingConfigurator
        value
        onChange={onChange}
        experimentId=""
        onExperimentIdChange={onExperimentIdChange}
      />,
    );

    expect(screen.getByRole('switch')).toBeChecked();
    expect(screen.getByText('Experiment')).toBeInTheDocument();
    expect(screen.getByRole('combobox')).toBeInTheDocument();
    expect(
      screen.getByText(/Select an existing experiment or leave blank to auto-create one named/),
    ).toBeInTheDocument();
  });

  test('calls onChange when toggle is clicked', async () => {
    const onChange = jest.fn();
    const onExperimentIdChange = jest.fn();

    renderWithDesignSystem(
      <UsageTrackingConfigurator
        value={false}
        onChange={onChange}
        experimentId=""
        onExperimentIdChange={onExperimentIdChange}
      />,
    );

    await userEvent.click(screen.getByRole('switch'));
    expect(onChange).toHaveBeenCalledWith(true);
  });

  test('displays description text', () => {
    const onChange = jest.fn();
    const onExperimentIdChange = jest.fn();

    renderWithDesignSystem(
      <UsageTrackingConfigurator
        value={false}
        onChange={onChange}
        experimentId=""
        onExperimentIdChange={onExperimentIdChange}
      />,
    );

    expect(screen.getByText(/all requests to this endpoint will be logged as traces/)).toBeInTheDocument();
  });
});
