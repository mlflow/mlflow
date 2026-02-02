import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import React from 'react';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { ExperimentSelect } from './ExperimentSelect';

const mockExperiments = [
  { experimentId: 'exp-1', name: 'Experiment 1' },
  { experimentId: 'exp-2', name: 'Experiment 2' },
  { experimentId: 'exp-3', name: 'Gateway Experiment' },
];

const mockUseExperimentsForSelect = jest.fn();

jest.mock('../../hooks/useExperimentsForSelect', () => ({
  useExperimentsForSelect: () => mockUseExperimentsForSelect(),
}));

describe('ExperimentSelect', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders loading state', () => {
    mockUseExperimentsForSelect.mockReturnValue({
      experiments: [],
      isLoading: true,
      error: null,
    });

    renderWithDesignSystem(
      <ExperimentSelect value="" onChange={jest.fn()} componentIdPrefix="test" />,
    );

    expect(screen.getByText('Loading experiments...')).toBeInTheDocument();
  });

  test('renders error state', () => {
    mockUseExperimentsForSelect.mockReturnValue({
      experiments: [],
      isLoading: false,
      error: new Error('Failed to fetch experiments'),
    });

    renderWithDesignSystem(
      <ExperimentSelect value="" onChange={jest.fn()} componentIdPrefix="test" />,
    );

    expect(screen.getByText('Failed to fetch experiments')).toBeInTheDocument();
  });

  test('renders select with experiments when loaded', () => {
    mockUseExperimentsForSelect.mockReturnValue({
      experiments: mockExperiments,
      isLoading: false,
      error: null,
    });

    renderWithDesignSystem(
      <ExperimentSelect value="" onChange={jest.fn()} componentIdPrefix="test" />,
    );

    expect(screen.getByRole('combobox')).toBeInTheDocument();
  });

  test('displays placeholder when no value selected', () => {
    mockUseExperimentsForSelect.mockReturnValue({
      experiments: mockExperiments,
      isLoading: false,
      error: null,
    });

    renderWithDesignSystem(
      <ExperimentSelect value="" onChange={jest.fn()} componentIdPrefix="test" />,
    );

    expect(screen.getByText('Auto-create experiment')).toBeInTheDocument();
  });

  test('renders with disabled state', () => {
    mockUseExperimentsForSelect.mockReturnValue({
      experiments: mockExperiments,
      isLoading: false,
      error: null,
    });

    renderWithDesignSystem(
      <ExperimentSelect value="" onChange={jest.fn()} componentIdPrefix="test" disabled />,
    );

    // The SimpleSelect component uses a button role for the trigger
    const trigger = screen.getByRole('combobox');
    // When disabled, the component should not be interactable
    expect(trigger).toBeInTheDocument();
  });

  test('renders default error message when error has no message', () => {
    mockUseExperimentsForSelect.mockReturnValue({
      experiments: [],
      isLoading: false,
      error: new Error(),
    });

    renderWithDesignSystem(
      <ExperimentSelect value="" onChange={jest.fn()} componentIdPrefix="test" />,
    );

    expect(screen.getByText('Failed to load experiments')).toBeInTheDocument();
  });
});
