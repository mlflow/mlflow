import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { TraceInfoCellValue } from './TraceInfoCellValue';

const renderValue = (value: string) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <TraceInfoCellValue value={value} />
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('TraceInfoCellValue', () => {
  it('renders an https value as a clickable link that opens in a new tab', () => {
    const url = 'https://example.com/traces/abc/rendering.html';
    renderValue(url);

    const link = screen.getByRole('link', { name: url });
    expect(link).toHaveAttribute('href', url);
    expect(link).toHaveAttribute('target', '_blank');
    // openInNewTab links should be safe against reverse tabnabbing
    expect(link).toHaveAttribute('rel', expect.stringContaining('noopener'));
  });

  it('renders an http value as a clickable link', () => {
    const url = 'http://localhost:8080/artifacts/trace.html';
    renderValue(url);

    expect(screen.getByRole('link', { name: url })).toHaveAttribute('href', url);
  });

  it('renders a non-URL string as plain text (no link)', () => {
    renderValue('production');

    expect(screen.queryByRole('link')).not.toBeInTheDocument();
    expect(screen.getByText('production')).toBeInTheDocument();
  });

  it('does not treat non-http(s) schemes such as javascript: as a link', () => {
    // eslint-disable-next-line no-script-url
    const value = 'javascript:alert(1)';
    renderValue(value);

    expect(screen.queryByRole('link')).not.toBeInTheDocument();
    expect(screen.getByText(value)).toBeInTheDocument();
  });
});
