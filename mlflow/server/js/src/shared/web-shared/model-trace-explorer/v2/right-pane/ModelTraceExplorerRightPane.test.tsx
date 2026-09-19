import { describe, it, expect } from '@jest/globals';
import { screen, within } from '@testing-library/react';
import { render } from '@databricks/web-shared/test-utils/render';
import userEvent from '@testing-library/user-event';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { ModelTraceExplorerContentTab } from './ModelTraceExplorerContentTab';
import type { ModelTraceSpan } from '../ModelTrace.types';
import { mockSpans, MOCK_RETRIEVER_SPAN, MOCK_CHAT_SPAN } from '../../ModelTraceExplorer.test-utils';
import { ModelTraceExplorerPreferencesProvider } from '../ModelTraceExplorerPreferencesContext';

const DEFAULT_SPAN: ModelTraceSpan = mockSpans[0];

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>
      <ModelTraceExplorerPreferencesProvider initialRenderMode="default">
        {children}
      </ModelTraceExplorerPreferencesProvider>
    </DesignSystemProvider>
  </IntlProvider>
);

describe('ModelTraceExplorerRightPane', () => {
  it('renders selected span payloads with the pretty field renderers by default', () => {
    const { rerender } = render(
      <ModelTraceExplorerContentTab
        activeSpan={{
          ...DEFAULT_SPAN,
          start: DEFAULT_SPAN.start_time,
          end: DEFAULT_SPAN.end_time,
          key: DEFAULT_SPAN.context.span_id,
          assessments: [],
          traceId: DEFAULT_SPAN.context.trace_id,
        }}
        searchFilter=""
        activeMatch={null}
      />,
      { wrapper: Wrapper },
    );

    expect(screen.queryByTestId('model-trace-explorer-retriever-field-renderer')).not.toBeInTheDocument();

    rerender(<ModelTraceExplorerContentTab activeSpan={MOCK_RETRIEVER_SPAN} searchFilter="" activeMatch={null} />);

    expect(screen.queryByTestId('model-trace-explorer-retriever-field-renderer')).toBeInTheDocument();
    expect(screen.getByTestId('model-trace-explorer-content-tab')).toHaveTextContent('Content with metadata');
  });

  it('renders chat-shaped inputs in pretty mode', async () => {
    render(<ModelTraceExplorerContentTab activeSpan={MOCK_CHAT_SPAN} searchFilter="" activeMatch={null} />, {
      wrapper: Wrapper,
    });

    // check that the user text renders
    expect(screen.queryByText('User')).toBeInTheDocument();
    expect(screen.queryByText("What's the weather in Singapore and New York?")).toBeInTheDocument();

    // Outputs render as message cards, without the raw LangChain wrapper fields in pretty mode.
    expect(screen.queryByText('The weather in Singapore is hot, while in New York, it is cold.')).toBeInTheDocument();
    const contentTab = screen.getByTestId('model-trace-explorer-content-tab');
    expect(contentTab).not.toHaveTextContent('generations');
    expect(contentTab).not.toHaveTextContent('llm_output');

    // Tool definitions start collapsed and show their count in the section title.
    expect(screen.queryByText('Tools')).toBeInTheDocument();
    expect(screen.queryByText('(1)')).toBeInTheDocument();
    expect(screen.queryAllByTestId('model-trace-explorer-chat-tool')).toHaveLength(0);
    await userEvent.click(screen.getByTestId('model-trace-explorer-tools-section-toggle'));
    expect(screen.queryAllByTestId('model-trace-explorer-chat-tool')).toHaveLength(1);
    expect(screen.queryByText('Tells a joke')).not.toBeInTheDocument();
    // Expand tool definition detail
    const toolDefinitionToggle = within(screen.getByTestId('model-trace-explorer-chat-tool')).getByTestId(
      'model-trace-explorer-chat-tool-toggle',
    );
    await userEvent.click(toolDefinitionToggle);
    expect(screen.queryByText('Tells a joke')).toBeInTheDocument();
  });

  it('shows raw input and output fields after switching render mode', async () => {
    render(<ModelTraceExplorerContentTab activeSpan={MOCK_CHAT_SPAN} searchFilter="" activeMatch={null} />, {
      wrapper: Wrapper,
    });

    expect(screen.queryByText('Inputs')).toBeInTheDocument();
    expect(screen.queryByText('Outputs')).toBeInTheDocument();
    expect(screen.getAllByText('Pretty')).toHaveLength(2);
    expect(screen.queryByText('Table')).not.toBeInTheDocument();
    await userEvent.click(screen.getAllByText('Pretty')[1]);
    await userEvent.click(screen.getByText('JSON'));

    const contentTab = screen.getByTestId('model-trace-explorer-content-tab');
    expect(contentTab).toHaveTextContent('generations');
    expect(contentTab).toHaveTextContent('llm_output');
  });

  it('switches a section to YAML rendering', async () => {
    render(<ModelTraceExplorerContentTab activeSpan={MOCK_CHAT_SPAN} searchFilter="" activeMatch={null} />, {
      wrapper: Wrapper,
    });

    await userEvent.click(screen.getAllByText('Pretty')[1]);
    await userEvent.click(screen.getByText('YAML'));

    expect(screen.getByRole('button', { name: 'YAML' })).toBeInTheDocument();
    expect(screen.getByTestId('model-trace-explorer-content-tab')).toHaveTextContent('generations:');
  });
});
