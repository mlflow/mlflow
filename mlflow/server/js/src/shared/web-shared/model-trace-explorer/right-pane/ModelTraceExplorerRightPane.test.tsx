import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { ModelTraceExplorerChatTab } from './ModelTraceExplorerChatTab';
import { ModelTraceExplorerContentTab } from './ModelTraceExplorerContentTab';
import type { ModelTraceSpan } from '../ModelTrace.types';
import {
  mockSpans,
  MOCK_RETRIEVER_SPAN,
  MOCK_CHAT_SPAN,
  MOCK_CHAT_MESSAGES,
  MOCK_CHAT_TOOLS,
} from '../ModelTraceExplorer.test-utils';

const DEFAULT_SPAN: ModelTraceSpan = mockSpans[0];

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

describe('ModelTraceExplorerRightPane', () => {
  it('switches between span renderers appropriately', () => {
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

    expect(screen.queryByTestId('model-trace-explorer-default-span-view')).toBeInTheDocument();

    rerender(<ModelTraceExplorerContentTab activeSpan={MOCK_RETRIEVER_SPAN} searchFilter="" activeMatch={null} />);

    expect(screen.queryByTestId('model-trace-explorer-retriever-span-view')).toBeInTheDocument();
  });

  it('should render conversations if possible', async () => {
    render(<ModelTraceExplorerChatTab chatMessages={MOCK_CHAT_MESSAGES} chatTools={MOCK_CHAT_TOOLS} />, {
      wrapper: Wrapper,
    });

    // check that the user text renders
    expect(screen.queryByText('User')).toBeInTheDocument();
    expect(screen.queryByText('tell me a joke in 50 words')).toBeInTheDocument();

    // check that the tool calls render
    expect(screen.queryByText('Assistant')).toBeInTheDocument();
    expect(screen.queryAllByText('tell_joke')).toHaveLength(2); // one in input, one in tool definition

    // check that the tool result render
    expect(screen.queryByText('Tool')).toBeInTheDocument();
    expect(
      screen.queryByText('Why did the scarecrow win an award? Because he was outstanding in his field!'),
    ).toBeInTheDocument();

    // check that the tool definition render
    expect(screen.queryAllByTestId('model-trace-explorer-chat-tool')).toHaveLength(1);
    expect(screen.queryByText('Tells a joke')).not.toBeInTheDocument();
    // Expand tool definition detail
    const toolDefinitionToggle = screen.queryAllByTestId('model-trace-explorer-chat-tool-toggle')[0];
    await userEvent.click(toolDefinitionToggle);
    expect(screen.queryByText('Tells a joke')).toBeInTheDocument();
  });

  it('shows raw input and output of spans', async () => {
    render(<ModelTraceExplorerContentTab activeSpan={MOCK_CHAT_SPAN} searchFilter="" activeMatch={null} />, {
      wrapper: Wrapper,
    });

    expect(screen.queryByText('Inputs')).toBeInTheDocument();
    expect(screen.queryByText('Outputs')).toBeInTheDocument();
    expect(screen.queryByText('generations')).toBeInTheDocument();
    expect(screen.queryByText('llm_output')).toBeInTheDocument();
    expect(screen.queryAllByText('See more')).toHaveLength(3);
  });
});
