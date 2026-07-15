import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { MCPServerTags } from './MCPServerTags';

const renderTags = (tags: Record<string, string>) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <MCPServerTags tags={tags} />
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('MCPServerTags', () => {
  it('renders em-dash when tags are empty', () => {
    const { container } = renderTags({});
    expect(container.textContent).toBe('—');
  });

  it('renders key-value tags', () => {
    const { container } = renderTags({ env: 'production' });
    expect(container.textContent).toContain('env');
    expect(container.textContent).toContain('production');
  });

  it('renders key-only tag when value is empty', () => {
    renderTags({ featured: '' });
    expect(screen.getByText('featured')).toBeInTheDocument();
  });
});
