import { describe, it, expect } from '@jest/globals';
import { screen } from '@testing-library/react';
import { render } from '@databricks/web-shared/test-utils/render';
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

    it('should render expanded IDE mode', () => {
      renderWithProviders(<CollapsibleJsonViewer data={JSON.stringify({ name: 'Alice' })} initialExpanded />);

      expect(screen.getByText('"name"')).toBeInTheDocument();
      expect(screen.queryByText('Path')).not.toBeInTheDocument();
    });
  });

  describe('IDE view rendering', () => {
    it('should render with JSON syntax in IDE view', () => {
      const data = JSON.stringify({ name: 'Alice', age: 30 });
      renderWithProviders(<CollapsibleJsonViewer data={data} initialExpanded />);

      expect(screen.getByText('"name"')).toBeInTheDocument();
      expect(screen.getByText('"Alice"')).toBeInTheDocument();
      expect(screen.getByText('"age"')).toBeInTheDocument();
      expect(screen.getByText('30')).toBeInTheDocument();
    });

    it('should support collapsing in IDE view', async () => {
      const user = userEvent.setup();
      const data = JSON.stringify({ items: ['a', 'b', 'c'] });
      renderWithProviders(<CollapsibleJsonViewer data={data} />);

      expect(screen.getByText('"a"')).toBeInTheDocument();

      const itemsKey = screen.getByText('"items"');
      // eslint-disable-next-line testing-library/no-node-access -- FEINF-5819: migrate direct DOM-node access to RTL queries; remove this disable when migrated
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
});
