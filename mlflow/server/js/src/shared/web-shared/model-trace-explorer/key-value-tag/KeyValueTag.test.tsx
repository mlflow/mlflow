import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';

import { DesignSystemProvider } from '@databricks/design-system';

import { KeyValueTag, getKeyAndValueComplexTruncation } from './KeyValueTag';

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <DesignSystemProvider>{children}</DesignSystemProvider>
);

describe('KeyValueTag', () => {
  it('renders key and value as plain text for non-URL values', () => {
    render(<KeyValueTag itemKey="page" itemValue="5" />, { wrapper: Wrapper });
    expect(screen.getByText('page')).toBeInTheDocument();
    expect(screen.getByText('5')).toBeInTheDocument();
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
  });

  it('renders a clickable link for http URL values', () => {
    render(<KeyValueTag itemKey="source" itemValue='"http://example.com/doc.pdf"' />, { wrapper: Wrapper });
    expect(screen.getByText('source')).toBeInTheDocument();
    const link = screen.getByRole('link');
    expect(link).toHaveAttribute('href', 'http://example.com/doc.pdf');
    expect(link).toHaveAttribute('target', '_blank');
  });

  it('renders a clickable link for https URL values', () => {
    render(<KeyValueTag itemKey="source" itemValue='"https://example.com/doc.pdf"' />, { wrapper: Wrapper });
    const link = screen.getByRole('link');
    expect(link).toHaveAttribute('href', 'https://example.com/doc.pdf');
  });

  it('renders a clickable link for URL values without surrounding quotes', () => {
    render(<KeyValueTag itemKey="source" itemValue="https://example.com/doc.pdf" />, { wrapper: Wrapper });
    const link = screen.getByRole('link');
    expect(link).toHaveAttribute('href', 'https://example.com/doc.pdf');
  });

  it('does not render a link for non-URL string values', () => {
    render(<KeyValueTag itemKey="format" itemValue='"pdf"' />, { wrapper: Wrapper });
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
    expect(screen.getByText('"pdf"')).toBeInTheDocument();
  });
});

describe('getKeyAndValueComplexTruncation', () => {
  it('does not truncate when total length is within limit', () => {
    expect(getKeyAndValueComplexTruncation('key', 'val', 18)).toEqual({
      shouldTruncateKey: false,
      shouldTruncateValue: false,
    });
  });

  it('truncates the longer string when shorter is within half limit', () => {
    expect(getKeyAndValueComplexTruncation('k', 'a-very-long-value-string', 18)).toEqual({
      shouldTruncateKey: false,
      shouldTruncateValue: true,
    });
  });

  it('truncates both when shorter string exceeds half the limit', () => {
    expect(getKeyAndValueComplexTruncation('a-long-key-name', 'a-long-value-name', 18)).toEqual({
      shouldTruncateKey: true,
      shouldTruncateValue: true,
    });
  });
});
