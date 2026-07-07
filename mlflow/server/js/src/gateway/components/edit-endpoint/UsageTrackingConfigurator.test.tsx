import { describe, test, expect, jest } from '@jest/globals';
import React from 'react';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { UsageTrackingConfigurator } from './UsageTrackingConfigurator';

describe('UsageTrackingConfigurator', () => {
  test('renders toggle in off state by default', () => {
    const onChange = jest.fn();

    renderWithDesignSystem(<UsageTrackingConfigurator value={false} onChange={onChange} />);

    expect(screen.getByText('Enable usage tracking')).toBeInTheDocument();
    expect(screen.getByRole('switch')).not.toBeChecked();
  });

  test('renders toggle in on state when value is true', () => {
    const onChange = jest.fn();

    renderWithDesignSystem(<UsageTrackingConfigurator value onChange={onChange} />);

    expect(screen.getByRole('switch')).toBeChecked();
  });

  test('calls onChange when toggle is clicked', async () => {
    const onChange = jest.fn();

    renderWithDesignSystem(<UsageTrackingConfigurator value={false} onChange={onChange} />);

    await userEvent.click(screen.getByRole('switch'));
    expect(onChange).toHaveBeenCalledWith(true);
  });

  test('displays description text', () => {
    const onChange = jest.fn();

    renderWithDesignSystem(<UsageTrackingConfigurator value={false} onChange={onChange} />);

    expect(screen.getByText(/all requests to this endpoint will be logged as traces/)).toBeInTheDocument();
  });
});
