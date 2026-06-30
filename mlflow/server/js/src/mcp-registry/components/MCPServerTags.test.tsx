import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { DesignSystemProvider } from '@databricks/design-system';
import { MCPServerTags } from './MCPServerTags';

const renderTags = (tags: Record<string, string>) =>
  render(
    <DesignSystemProvider>
      <MCPServerTags tags={tags} />
    </DesignSystemProvider>,
  );

describe('MCPServerTags', () => {
  it('renders em-dash when tags are empty', () => {
    const { container } = renderTags({});
    expect(container.textContent).toBe('—');
  });

  it('renders key-value tags', () => {
    renderTags({ env: 'production' });
    expect(screen.getByText('env: production')).toBeInTheDocument();
  });

  it('renders key-only tag when value is empty', () => {
    renderTags({ featured: '' });
    expect(screen.getByText('featured')).toBeInTheDocument();
  });
});
