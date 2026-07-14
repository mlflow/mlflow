import { describe, test, expect, jest } from '@jest/globals';
import React from 'react';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { UsageTrackingConfigurator, getUsageTrackingMode } from './UsageTrackingConfigurator';

describe('getUsageTrackingMode', () => {
  test.each([
    [false, false, 'off'],
    [false, true, 'off'],
    [true, true, 'metadata_only'],
    [true, false, 'full'],
  ] as const)('usageTracking=%s excludeContent=%s -> %s', (usageTracking, excludeContent, expected) => {
    expect(getUsageTrackingMode(usageTracking, excludeContent)).toBe(expected);
  });
});

describe('UsageTrackingConfigurator', () => {
  test('renders all three mode options with descriptions', () => {
    const onChange = jest.fn();

    renderWithDesignSystem(<UsageTrackingConfigurator mode="off" onChange={onChange} />);

    expect(screen.getByRole('radio', { name: 'Tracing off' })).toBeChecked();
    expect(screen.getByRole('radio', { name: 'Redact message content' })).not.toBeChecked();
    expect(screen.getByRole('radio', { name: 'Full tracing' })).not.toBeChecked();
    expect(screen.getByText(/prompts, messages, and model responses are redacted/i)).toBeInTheDocument();
    expect(screen.getByText(/complete request and response content/i)).toBeInTheDocument();
  });

  test('reflects the selected mode', () => {
    const onChange = jest.fn();

    renderWithDesignSystem(<UsageTrackingConfigurator mode="metadata_only" onChange={onChange} />);

    expect(screen.getByRole('radio', { name: 'Redact message content' })).toBeChecked();
  });

  test('calls onChange with the selected mode', async () => {
    const onChange = jest.fn();

    renderWithDesignSystem(<UsageTrackingConfigurator mode="off" onChange={onChange} />);

    await userEvent.click(screen.getByRole('radio', { name: 'Redact message content' }));
    expect(onChange).toHaveBeenCalledWith('metadata_only');

    await userEvent.click(screen.getByRole('radio', { name: 'Full tracing' }));
    expect(onChange).toHaveBeenCalledWith('full');
  });

  test('hides descriptions in compact mode', () => {
    const onChange = jest.fn();

    renderWithDesignSystem(<UsageTrackingConfigurator mode="full" onChange={onChange} compact />);

    expect(screen.getByRole('radio', { name: 'Full tracing' })).toBeChecked();
    expect(screen.queryByText(/complete request and response content/i)).not.toBeInTheDocument();
  });
});
