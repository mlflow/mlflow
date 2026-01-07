import { describe, it, expect } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { ModelTraceExplorerLinkedPromptsView } from './ModelTraceExplorerLinkedPromptsView';
import { BrowserRouter } from '../RoutingUtils';

describe('ModelTraceExplorerLinkedPromptsView', () => {
  const mockExperimentId = 'exp-123';

  const renderWithProviders = (component: React.ReactElement) => {
    return render(
      <BrowserRouter>
        <IntlProvider locale="en" messages={{}}>
          <DesignSystemProvider>{component}</DesignSystemProvider>
        </IntlProvider>
      </BrowserRouter>,
    );
  };

  const createMockModelTrace = (linkedPromptsTag?: string) => ({
    info: {
      version: '3' as const,
      trace_metadata: {}, // Required for isV3ModelTraceInfo check
      trace_location: {
        type: 'MLFLOW_EXPERIMENT' as const,
        mlflow_experiment: {
          experiment_id: mockExperimentId,
        },
      },
      tags: linkedPromptsTag
        ? {
            'mlflow.linkedPrompts': linkedPromptsTag,
          }
        : {},
    },
    data: {
      spans: [],
      exceptions: [],
    },
  });

  it('should render table with three parsed prompts', async () => {
    const linkedPrompts = [
      { name: 'prompt-1', version: 'v1.0.0' },
      { name: 'prompt-2', version: 'v2.0.0' },
      { name: 'prompt-3', version: 'v3.0.0' },
    ];
    const modelTrace = createMockModelTrace(JSON.stringify(linkedPrompts));

    renderWithProviders(<ModelTraceExplorerLinkedPromptsView modelTraceInfo={modelTrace.info as any} />);

    // Wait for the table to render
    await waitFor(() => {
      expect(screen.getByRole('table')).toBeInTheDocument();
    });

    // Verify the three prompts are displayed in the table
    expect(screen.getByText('prompt-1')).toBeInTheDocument();
    expect(screen.getByText('prompt-2')).toBeInTheDocument();
    expect(screen.getByText('prompt-3')).toBeInTheDocument();
  });

  it('should show empty state when no linked prompts exist', () => {
    const modelTrace = createMockModelTrace();

    renderWithProviders(<ModelTraceExplorerLinkedPromptsView modelTraceInfo={modelTrace.info as any} />);

    // Verify empty state is shown
    expect(screen.getByText('Link prompts to traces')).toBeInTheDocument();
  });

  it('should handle invalid JSON in tags gracefully', () => {
    const modelTrace = createMockModelTrace('invalid-json-{]');

    renderWithProviders(<ModelTraceExplorerLinkedPromptsView modelTraceInfo={modelTrace.info as any} />);

    // Verify empty state is shown when JSON is invalid
    expect(screen.getByText('Link prompts to traces')).toBeInTheDocument();
  });
});
