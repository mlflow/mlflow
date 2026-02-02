import { describe, test, expect, jest } from '@jest/globals';
import React from 'react';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { UsageTrackingConfigurator } from './UsageTrackingConfigurator';

jest.mock('../shared/ExperimentSelect', () => ({
  ExperimentSelect: ({
    value,
    onChange,
    componentIdPrefix,
  }: {
    value: string;
    onChange: (value: string) => void;
    componentIdPrefix: string;
  }) => (
    <select
      data-testid={`${componentIdPrefix}-mock-select`}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    >
      <option value="">Auto-create</option>
      <option value="exp-1">Experiment 1</option>
      <option value="exp-2">Experiment 2</option>
    </select>
  ),
}));

describe('UsageTrackingConfigurator', () => {
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

  test('calls onExperimentIdChange when experiment is selected', async () => {
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

    const select = screen.getByTestId('mlflow.gateway.usage-tracking.experiment-mock-select');
    await userEvent.selectOptions(select, 'exp-1');
    expect(onExperimentIdChange).toHaveBeenCalledWith('exp-1');
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

    expect(
      screen.getByText(/all requests to this endpoint will be logged as traces/),
    ).toBeInTheDocument();
  });
});
