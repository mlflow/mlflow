import { describe, it, expect, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';

import type { Assessment } from './ModelTrace.types';
import { MOCK_ASSESSMENT, MOCK_SPAN_ASSESSMENT, MOCK_TRACE, MOCK_V3_TRACE } from './ModelTraceExplorer.test-utils';
import { SimplifiedModelTraceExplorerImpl } from './SimplifiedModelTraceExplorer';

jest.mock('./hooks/useGetModelTraceInfo', () => ({
  useGetModelTraceInfo: jest.fn().mockReturnValue({
    refetch: jest.fn(),
  }),
}));

const Wrapper = ({ children }: { children: React.ReactNode }) => {
  const queryClient = new QueryClient();
  return (
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
      </DesignSystemProvider>
    </IntlProvider>
  );
};

describe('SimplifiedModelTraceExplorer', () => {
  it('renders the component with trace data and assessments', () => {
    const assessments: Assessment[] = [MOCK_ASSESSMENT, MOCK_SPAN_ASSESSMENT];

    render(<SimplifiedModelTraceExplorerImpl modelTrace={MOCK_TRACE} assessments={assessments} />, {
      wrapper: Wrapper,
    });

    // Assert that inputs/outputs sections are rendered (these are always visible)
    expect(screen.getByText('Inputs')).toBeInTheDocument();
    expect(screen.getByText('Outputs')).toBeInTheDocument();

    // Assert that assessment cards are rendered
    expect(screen.getByText('Relevance')).toBeInTheDocument();
    expect(screen.getByText('Thumbs')).toBeInTheDocument();
  });

  it('displays trace breakdown in left pane', () => {
    render(<SimplifiedModelTraceExplorerImpl modelTrace={MOCK_V3_TRACE} assessments={[]} />, {
      wrapper: Wrapper,
    });

    // Check that inputs/outputs sections are present (these are always visible)
    expect(screen.getByText('Inputs')).toBeInTheDocument();
    expect(screen.getByText('Outputs')).toBeInTheDocument();
  });

  it('displays assessments in right pane', () => {
    const assessments: Assessment[] = [MOCK_ASSESSMENT];

    render(<SimplifiedModelTraceExplorerImpl modelTrace={MOCK_TRACE} assessments={assessments} />, {
      wrapper: Wrapper,
    });

    // Check that assessment is rendered
    expect(screen.getByText('Relevance')).toBeInTheDocument();
    expect(screen.getByText('The thought process is sound and follows from the request')).toBeInTheDocument();
  });

  it('displays "No assessments available" when assessments array is empty', () => {
    render(<SimplifiedModelTraceExplorerImpl modelTrace={MOCK_TRACE} assessments={[]} />, {
      wrapper: Wrapper,
    });

    // Check that empty state message is shown
    expect(screen.getByText('No assessments available')).toBeInTheDocument();
  });

  it('does not render render mode selector in summary view', () => {
    render(<SimplifiedModelTraceExplorerImpl modelTrace={MOCK_TRACE} assessments={[]} />, {
      wrapper: Wrapper,
    });

    // Check that render mode selector is not present
    expect(screen.queryByText('Default')).not.toBeInTheDocument();
    expect(screen.queryByText('JSON')).not.toBeInTheDocument();
  });

  it('renders intermediate nodes correctly', () => {
    render(<SimplifiedModelTraceExplorerImpl modelTrace={MOCK_TRACE} assessments={[]} />, {
      wrapper: Wrapper,
    });

    // Check that important intermediate nodes are rendered
    // The MOCK_TRACE has CHAT_MODEL and LLM type spans which are important
    expect(screen.getByText('_generate_response')).toBeInTheDocument();
    expect(screen.getByText('rephrase_chat_to_queue')).toBeInTheDocument();
  });
});
