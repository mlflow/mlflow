import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { DesignSystemProvider } from '@databricks/design-system';

import { CollapsibleJsonViewer } from './CollapsibleJsonViewer';
import { IntlProvider } from '@databricks/i18n';

function renderWithProviders(ui: React.ReactElement) {
  return render(
    <IntlProvider locale="en">
      <DesignSystemProvider>{ui}</DesignSystemProvider>
    </IntlProvider>,
  );
}

describe('CollapsibleJsonViewer', () => {
  describe('Primitive values', () => {
    it('should render string primitives', () => {
      renderWithProviders(<CollapsibleJsonViewer data={JSON.stringify('hello')} />);
      expect(screen.getByText('"hello"')).toBeInTheDocument();
    });

    it('should render number primitives', () => {
      renderWithProviders(<CollapsibleJsonViewer data={JSON.stringify(42)} />);
      expect(screen.getByText('42')).toBeInTheDocument();
    });

    it('should render boolean primitives', () => {
      renderWithProviders(<CollapsibleJsonViewer data={JSON.stringify(true)} />);
      expect(screen.getByText('true')).toBeInTheDocument();
    });

    it('should render null primitives', () => {
      renderWithProviders(<CollapsibleJsonViewer data="null" />);
      expect(screen.getByText('null')).toBeInTheDocument();
    });

    it('should truncate very long primitive strings', () => {
      const longString = 'a'.repeat(2000);
      renderWithProviders(<CollapsibleJsonViewer data={JSON.stringify(longString)} />);

      const displayedText = screen.getByText(/"a+\.\.\."/, { exact: false });
      expect(displayedText.textContent).toContain('...');
    });
  });

  describe('Render modes', () => {
    it('should default to IDE view for objects and arrays', () => {
      renderWithProviders(<CollapsibleJsonViewer data={JSON.stringify({ test: 'value' })} />);

      expect(screen.queryByText('Path')).not.toBeInTheDocument();
      expect(screen.queryByText('Value')).not.toBeInTheDocument();
    });

    it('should render in table mode when specified', () => {
      renderWithProviders(<CollapsibleJsonViewer data={JSON.stringify({ test: 'value' })} renderMode="table" />);

      expect(screen.getByText('Path')).toBeInTheDocument();
      expect(screen.getByText('Value')).toBeInTheDocument();
      expect(screen.getByText('test')).toBeInTheDocument();
    });

    it('should render in IDE mode when specified', () => {
      renderWithProviders(
        <CollapsibleJsonViewer data={JSON.stringify({ name: 'Alice' })} renderMode="json" initialExpanded />,
      );

      expect(screen.getByText('"name"')).toBeInTheDocument();
      expect(screen.queryByText('Path')).not.toBeInTheDocument();
    });
  });

  describe('Table view rendering', () => {
    it('should render object properties in table format', () => {
      const data = JSON.stringify({ name: 'John', age: 42, active: true, metadata: null });
      renderWithProviders(<CollapsibleJsonViewer data={data} renderMode="table" initialExpanded />);

      expect(screen.getByText('name')).toBeInTheDocument();
      expect(screen.getByText('"John"')).toBeInTheDocument();
      expect(screen.getByText('age')).toBeInTheDocument();
      expect(screen.getByText('42')).toBeInTheDocument();
      expect(screen.getByText('active')).toBeInTheDocument();
      expect(screen.getByText('true')).toBeInTheDocument();
      expect(screen.getByText('metadata')).toBeInTheDocument();
      expect(screen.getByText('null')).toBeInTheDocument();
    });

    it('should render nested objects with hierarchical paths', () => {
      const data = JSON.stringify({
        user: {
          profile: {
            name: 'Alice',
            location: 'SF',
          },
        },
      });
      renderWithProviders(<CollapsibleJsonViewer data={data} renderMode="table" initialExpanded />);

      expect(screen.getByText('user')).toBeInTheDocument();
      expect(screen.getByText('profile')).toBeInTheDocument();
      expect(screen.getByText('name')).toBeInTheDocument();
      expect(screen.getByText('"Alice"')).toBeInTheDocument();
      expect(screen.getByText('location')).toBeInTheDocument();
      expect(screen.getByText('"SF"')).toBeInTheDocument();
    });

    it('should render arrays with numeric indices', () => {
      const data = JSON.stringify({ items: ['apple', 'banana', 'cherry'] });
      renderWithProviders(<CollapsibleJsonViewer data={data} renderMode="table" initialExpanded />);

      expect(screen.getByText('items')).toBeInTheDocument();
      expect(screen.getByText('"apple"')).toBeInTheDocument();
      expect(screen.getByText('"banana"')).toBeInTheDocument();
      expect(screen.getByText('"cherry"')).toBeInTheDocument();
    });

    it('should render arrays of objects', () => {
      const data = JSON.stringify({
        users: [
          { id: 1, name: 'Alice' },
          { id: 2, name: 'Bob' },
        ],
      });
      renderWithProviders(<CollapsibleJsonViewer data={data} renderMode="table" initialExpanded />);

      expect(screen.getByText('users')).toBeInTheDocument();
      expect(screen.getAllByText('id')).toHaveLength(2);
      expect(screen.getAllByText('name')).toHaveLength(2);
      expect(screen.getByText('"Alice"')).toBeInTheDocument();
      expect(screen.getByText('"Bob"')).toBeInTheDocument();
    });

    it('should render root-level arrays', () => {
      const data = JSON.stringify([1, 2, 3]);
      renderWithProviders(<CollapsibleJsonViewer data={data} renderMode="table" initialExpanded />);

      expect(screen.getAllByText('1').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('2').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('3').length).toBeGreaterThanOrEqual(1);
    });
  });

  describe('Collapsible behavior', () => {
    it('should collapse nested structures by default in table mode', () => {
      const data = JSON.stringify({
        user: { name: 'Alice', age: 30 },
        settings: { theme: 'dark', language: 'en' },
      });
      renderWithProviders(<CollapsibleJsonViewer data={data} renderMode="table" />);

      expect(screen.getByText('user')).toBeInTheDocument();
      expect(screen.queryByText('name')).not.toBeInTheDocument();
      expect(screen.queryByText('"Alice"')).not.toBeInTheDocument();

      expect(screen.getAllByText(/2 properties/).length).toBe(2);
    });

    it('should expand all when initialExpanded is true in table mode', () => {
      const data = JSON.stringify({
        user: { name: 'Alice', age: 30 },
      });
      renderWithProviders(<CollapsibleJsonViewer data={data} renderMode="table" initialExpanded />);

      expect(screen.getByText('user')).toBeInTheDocument();
      expect(screen.getByText('name')).toBeInTheDocument();
      expect(screen.getByText('"Alice"')).toBeInTheDocument();
    });

    it('should toggle collapse state on click in table mode', async () => {
      const user = userEvent.setup();
      const data = JSON.stringify({ items: ['a', 'b', 'c'] });
      renderWithProviders(<CollapsibleJsonViewer data={data} renderMode="table" />);

      expect(screen.queryByText('"a"')).not.toBeInTheDocument();

      await user.click(screen.getByRole('button'));

      expect(screen.getByText('"a"')).toBeInTheDocument();
      expect(screen.getByText('"b"')).toBeInTheDocument();
      expect(screen.getByText('"c"')).toBeInTheDocument();

      await user.click(screen.getByRole('button'));

      expect(screen.queryByText('"a"')).not.toBeInTheDocument();
    });

    it('should show preview for collapsed array with multiple items', () => {
      renderWithProviders(
        <CollapsibleJsonViewer data={JSON.stringify({ items: [1, 2, 3, 4, 5] })} renderMode="table" />,
      );
      expect(screen.getByText(/5 items/)).toBeInTheDocument();
    });

    it('should show preview for collapsed array with single item', () => {
      renderWithProviders(<CollapsibleJsonViewer data={JSON.stringify({ single: [1] })} renderMode="table" />);
      expect(screen.getByText(/1 item/)).toBeInTheDocument();
    });

    it('should show preview for collapsed object with multiple properties', () => {
      renderWithProviders(
        <CollapsibleJsonViewer data={JSON.stringify({ obj: { a: 1, b: 2, c: 3 } })} renderMode="table" />,
      );
      expect(screen.getByText(/3 properties/)).toBeInTheDocument();
    });

    it('should show preview for collapsed object with single property', () => {
      renderWithProviders(<CollapsibleJsonViewer data={JSON.stringify({ single: { a: 1 } })} renderMode="table" />);
      expect(screen.getByText(/1 property/)).toBeInTheDocument();
    });
  });

  describe('IDE view rendering', () => {
    it('should render with JSON syntax in IDE view', () => {
      const data = JSON.stringify({ name: 'Alice', age: 30 });
      renderWithProviders(<CollapsibleJsonViewer data={data} renderMode="json" initialExpanded />);

      expect(screen.getByText('"name"')).toBeInTheDocument();
      expect(screen.getByText('"Alice"')).toBeInTheDocument();
      expect(screen.getByText('"age"')).toBeInTheDocument();
      expect(screen.getByText('30')).toBeInTheDocument();
    });

    it('should support collapsing in IDE view', async () => {
      const user = userEvent.setup();
      const data = JSON.stringify({ items: ['a', 'b', 'c'] });
      renderWithProviders(<CollapsibleJsonViewer data={data} renderMode="json" />);

      expect(screen.getByText('"a"')).toBeInTheDocument();

      const itemsKey = screen.getByText('"items"');
      await user.click(itemsKey.closest('div')!);

      expect(screen.queryByText('"a"')).not.toBeInTheDocument();
    });
  });

  describe('Error handling', () => {
    it('should display error message for invalid JSON', () => {
      const { container } = renderWithProviders(<CollapsibleJsonViewer data="{ invalid json }" />);
      expect(container).toHaveTextContent('[Invalid JSON]');
    });

    it('should display error message for malformed JSON', () => {
      const { container } = renderWithProviders(<CollapsibleJsonViewer data='{"key": "value"' />);
      expect(container).toHaveTextContent('[Invalid JSON]');
    });

    it('should display error message for empty string', () => {
      const { container } = renderWithProviders(<CollapsibleJsonViewer data="" />);
      expect(container).toHaveTextContent('[Invalid JSON]');
    });
  });

  describe('Edge cases', () => {
    it('should handle empty object', () => {
      renderWithProviders(<CollapsibleJsonViewer data={JSON.stringify({})} renderMode="table" />);
      expect(screen.getByText('Path')).toBeInTheDocument();
      expect(screen.getByText('Value')).toBeInTheDocument();
    });

    it('should handle empty array', () => {
      renderWithProviders(
        <CollapsibleJsonViewer data={JSON.stringify({ items: [] })} renderMode="table" initialExpanded />,
      );
      expect(screen.getByText('items')).toBeInTheDocument();
      expect(screen.getByText('[]')).toBeInTheDocument();
    });

    it('should handle empty string value', () => {
      renderWithProviders(
        <CollapsibleJsonViewer data={JSON.stringify({ text: '' })} renderMode="table" initialExpanded />,
      );
      expect(screen.getByText('text')).toBeInTheDocument();
      expect(screen.getByText('""')).toBeInTheDocument();
    });

    it('should handle quotes in strings', () => {
      const data = JSON.stringify({
        quotes: 'He said "hello"',
      });
      renderWithProviders(<CollapsibleJsonViewer data={data} renderMode="table" initialExpanded />);

      expect(screen.getByText('"He said "hello""')).toBeInTheDocument();
    });

    it('should handle newlines in strings', () => {
      const data = JSON.stringify({
        newlines: 'Line 1\nLine 2',
      });
      renderWithProviders(<CollapsibleJsonViewer data={data} renderMode="table" initialExpanded />);

      expect(screen.getByText(/"Line 1\s+Line 2"/)).toBeInTheDocument();
    });

    it('should handle unicode characters', () => {
      const data = JSON.stringify({
        unicode: '🚀 你好',
      });
      renderWithProviders(<CollapsibleJsonViewer data={data} renderMode="table" initialExpanded />);

      expect(screen.getByText('"🚀 你好"')).toBeInTheDocument();
    });

    it('should handle numeric keys', () => {
      const data = JSON.stringify({
        123: 'numeric',
      });
      renderWithProviders(<CollapsibleJsonViewer data={data} renderMode="table" initialExpanded />);

      expect(screen.getByText('123')).toBeInTheDocument();
      expect(screen.getByText('"numeric"')).toBeInTheDocument();
    });

    it('should handle keys with dashes', () => {
      const data = JSON.stringify({
        'key-with-dashes': 'value1',
      });
      renderWithProviders(<CollapsibleJsonViewer data={data} renderMode="table" initialExpanded />);

      expect(screen.getByText('key-with-dashes')).toBeInTheDocument();
      expect(screen.getByText('"value1"')).toBeInTheDocument();
    });

    it('should handle keys with dots', () => {
      const data = JSON.stringify({
        'key.with.dots': 'value2',
      });
      renderWithProviders(<CollapsibleJsonViewer data={data} renderMode="table" initialExpanded />);

      expect(screen.getByText('key.with.dots')).toBeInTheDocument();
      expect(screen.getByText('"value2"')).toBeInTheDocument();
    });

    it('should handle keys with spaces', () => {
      const data = JSON.stringify({
        'key with spaces': 'value3',
      });
      renderWithProviders(<CollapsibleJsonViewer data={data} renderMode="table" initialExpanded />);

      expect(screen.getByText('key with spaces')).toBeInTheDocument();
      expect(screen.getByText('"value3"')).toBeInTheDocument();
    });

    it('should truncate very long strings', () => {
      const longString = 'a'.repeat(2000);
      const data = JSON.stringify({ text: longString });
      renderWithProviders(<CollapsibleJsonViewer data={data} renderMode="table" initialExpanded />);

      const displayedText = screen.getByText(/"a+\.\.\."/, { exact: false });
      expect(displayedText.textContent).toContain('...');
    });

    it('should prevent infinite recursion with depth limit', () => {
      let deepData: any = { value: 'end' };
      for (let i = 0; i < 150; i++) {
        deepData = { nested: deepData };
      }
      renderWithProviders(<CollapsibleJsonViewer data={JSON.stringify(deepData)} renderMode="table" initialExpanded />);

      expect(screen.queryByText('value')).not.toBeInTheDocument();
    });
  });

  describe('Model trace data formats', () => {
    it('should render LLM chat messages', () => {
      const data = JSON.stringify({
        role: 'user',
        content: 'Hello, how are you?',
      });
      renderWithProviders(<CollapsibleJsonViewer data={data} renderMode="table" initialExpanded />);

      expect(screen.getByText('role')).toBeInTheDocument();
      expect(screen.getByText('"user"')).toBeInTheDocument();
      expect(screen.getByText('content')).toBeInTheDocument();
      expect(screen.getByText('"Hello, how are you?"')).toBeInTheDocument();
    });

    it('should render LLM tool calls', () => {
      const data = JSON.stringify({
        role: 'assistant',
        content: null,
        tool_calls: [
          {
            id: 'call_123',
            type: 'function',
            function: {
              name: 'get_weather',
              arguments: '{"location": "San Francisco"}',
            },
          },
        ],
      });
      renderWithProviders(<CollapsibleJsonViewer data={data} renderMode="table" initialExpanded />);

      expect(screen.getByText('"assistant"')).toBeInTheDocument();
      expect(screen.getByText('tool_calls')).toBeInTheDocument();
      expect(screen.getByText('function')).toBeInTheDocument();
      expect(screen.getByText('"get_weather"')).toBeInTheDocument();
    });

    it('should render conversation history', () => {
      const data = JSON.stringify([
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'What is the capital of France?' },
        { role: 'assistant', content: 'The capital of France is Paris.' },
      ]);
      renderWithProviders(<CollapsibleJsonViewer data={data} renderMode="table" initialExpanded />);

      expect(screen.getByText('"system"')).toBeInTheDocument();
      expect(screen.getByText('"user"')).toBeInTheDocument();
      expect(screen.getByText('"assistant"')).toBeInTheDocument();
      expect(screen.getByText('"You are a helpful assistant."')).toBeInTheDocument();
      expect(screen.getByText('"What is the capital of France?"')).toBeInTheDocument();
      expect(screen.getByText('"The capital of France is Paris."')).toBeInTheDocument();
    });

    it('should render model trace metadata', () => {
      const data = JSON.stringify({
        request_id: 'req_abc123',
        model: 'gpt-4',
        temperature: 0.7,
        timestamp: 1234567890,
        usage: {
          prompt_tokens: 10,
          completion_tokens: 20,
          total_tokens: 30,
        },
      });
      renderWithProviders(<CollapsibleJsonViewer data={data} renderMode="table" initialExpanded />);

      expect(screen.getByText('model')).toBeInTheDocument();
      expect(screen.getByText('"gpt-4"')).toBeInTheDocument();
      expect(screen.getByText('temperature')).toBeInTheDocument();
      expect(screen.getByText('0.7')).toBeInTheDocument();
      expect(screen.getByText('usage')).toBeInTheDocument();
      expect(screen.getByText('total_tokens')).toBeInTheDocument();
      expect(screen.getByText('30')).toBeInTheDocument();
    });
  });
});
