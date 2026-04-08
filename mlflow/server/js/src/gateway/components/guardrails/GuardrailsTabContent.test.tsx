import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import React from 'react';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { GuardrailsTabContent } from './GuardrailsTabContent';

jest.mock('../../hooks/useGuardrailsQuery');
jest.mock('../../hooks/useRemoveGuardrail');
jest.mock('./AddGuardrailModal', () => ({ AddGuardrailModal: () => null }));

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
  jest.mocked(useRemoveGuardrail).mockReturnValue({ mutateAsync: jest.fn() as any, isPending: false } as any);
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
    expect(screen.getByText('Before LLM')).toBeInTheDocument();
    expect(screen.getByText('After LLM')).toBeInTheDocument();
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

  test('clicking delete opens confirmation modal', async () => {
    setup();
    renderWithDesignSystem(<GuardrailsTabContent {...defaultProps} />);
    const deleteButtons = screen.getAllByRole('button', { name: /Remove guardrail/i });
    await userEvent.click(deleteButtons[0]);
    expect(screen.getByText('Remove Guardrail')).toBeInTheDocument();
  });

  test('view buttons are disabled', () => {
    setup();
    renderWithDesignSystem(<GuardrailsTabContent {...defaultProps} />);
    const viewButtons = screen.getAllByRole('button', { name: /View and edit guardrail/i });
    viewButtons.forEach((btn) => expect(btn).toBeDisabled());
  });
});
