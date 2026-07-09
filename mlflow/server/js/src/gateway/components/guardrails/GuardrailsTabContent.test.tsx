import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import React from 'react';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { GuardrailsTabContent } from './GuardrailsTabContent';

jest.mock('../../hooks/useGuardrailsQuery');
jest.mock('../../hooks/useRemoveGuardrail');
jest.mock('./AddGuardrailModal', () => ({ AddGuardrailModal: () => null }));
jest.mock('./GuardrailDetailModal', () => ({ GuardrailDetailModal: () => null }));
jest.mock('@mlflow/mlflow/src/common/utils/reactQueryHooks', () => ({
  useQueryClient: () => ({ invalidateQueries: jest.fn() }),
}));

const { useGuardrailsQuery } = jest.requireMock<typeof import('../../hooks/useGuardrailsQuery')>(
  '../../hooks/useGuardrailsQuery',
);
const { useRemoveGuardrail } = jest.requireMock<typeof import('../../hooks/useRemoveGuardrail')>(
  '../../hooks/useRemoveGuardrail',
);

const mockGuardrails = [
  {
    endpoint_id: 'ep-1',
    guardrail_id: 'g-1',
    execution_order: 0,
    created_at: 0,
    guardrail: {
      guardrail_id: 'g-1',
      name: 'Safety',
      stage: 'BEFORE' as const,
      action: 'VALIDATION' as const,
      created_at: 0,
      last_updated_at: 0,
    },
  },
  {
    endpoint_id: 'ep-1',
    guardrail_id: 'g-2',
    execution_order: 1,
    created_at: 0,
    guardrail: {
      guardrail_id: 'g-2',
      name: 'PII Filter',
      stage: 'AFTER' as const,
      action: 'SANITIZATION' as const,
      created_at: 0,
      last_updated_at: 0,
    },
  },
];

const defaultProps = {
  endpointName: 'test-endpoint',
  endpointId: 'ep-1',
  experimentId: '0',
};

function setup(guardrails = mockGuardrails, isLoading = false) {
  const refetch = jest.fn() as any;
  jest.mocked(useGuardrailsQuery).mockReturnValue({
    data: guardrails,
    isLoading,
    error: undefined,
    refetch,
  });
  jest.mocked(useRemoveGuardrail).mockReturnValue({ mutateAsync: jest.fn() as any, isLoading: false } as any);
  return { refetch };
}

describe('GuardrailsTabContent', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('shows loading spinner while loading', () => {
    setup([], true);
    renderWithDesignSystem(<GuardrailsTabContent {...defaultProps} />);
    expect(screen.getByText('Loading guardrails...')).toBeInTheDocument();
  });

  test('shows empty state when no guardrails', () => {
    setup([]);
    renderWithDesignSystem(<GuardrailsTabContent {...defaultProps} />);
    expect(screen.getByText(/No guardrails configured/)).toBeInTheDocument();
  });

  test('renders guardrail rows with name, placement, and action', () => {
    setup();
    renderWithDesignSystem(<GuardrailsTabContent {...defaultProps} />);
    expect(screen.getByText('Safety')).toBeInTheDocument();
    expect(screen.getByText('PII Filter')).toBeInTheDocument();
    expect(screen.getByText('Pre-LLM Guardrails')).toBeInTheDocument();
    expect(screen.getByText('Post-LLM Guardrails')).toBeInTheDocument();
    expect(screen.getByText('Block')).toBeInTheDocument();
    expect(screen.getByText('Sanitize')).toBeInTheDocument();
  });

  test('filters guardrails by search', async () => {
    setup();
    renderWithDesignSystem(<GuardrailsTabContent {...defaultProps} />);
    await userEvent.type(screen.getByPlaceholderText('Search guardrails'), 'safety');
    expect(screen.getByText('Safety')).toBeInTheDocument();
    expect(screen.queryByText('PII Filter')).not.toBeInTheDocument();
  });

  test('shows empty search state when no match', async () => {
    setup();
    renderWithDesignSystem(<GuardrailsTabContent {...defaultProps} />);
    await userEvent.type(screen.getByPlaceholderText('Search guardrails'), 'nonexistent');
    expect(screen.getByText(/No guardrails match your search/)).toBeInTheDocument();
  });

  test('clicking delete opens confirmation modal after selecting a row', async () => {
    setup();
    renderWithDesignSystem(<GuardrailsTabContent {...defaultProps} />);

    // Select the first row via its checkbox, then click the toolbar Delete button
    const checkboxes = screen.getAllByRole('checkbox');
    await userEvent.click(checkboxes[1]); // index 0 is "select all", index 1 is first row
    await userEvent.click(screen.getByRole('button', { name: /^Delete/ }));

    expect(screen.getByText(/Delete.*guardrail/i)).toBeInTheDocument();
  });

  test('clicking a guardrail name renders the row', async () => {
    setup();
    renderWithDesignSystem(<GuardrailsTabContent {...defaultProps} />);

    const nameButton = screen.getByRole('button', { name: 'Safety' });
    expect(nameButton).toBeInTheDocument();
    await userEvent.click(nameButton);
    // Row remains visible after click (modal is mocked out)
    expect(screen.getByRole('button', { name: 'Safety' })).toBeInTheDocument();
  });
});
