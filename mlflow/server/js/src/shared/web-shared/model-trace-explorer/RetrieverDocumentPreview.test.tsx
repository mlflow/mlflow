import { describe, expect, it } from '@jest/globals';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from '@databricks/web-shared/test-utils/render';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { ModelTraceExplorerRetrieverDocument as LegacyDocument } from './right-pane/ModelTraceExplorerRetrieverDocument';
import { ModelTraceExplorerRetrieverDocument as RedesignedDocument } from './v2/right-pane/ModelTraceExplorerRetrieverDocument';
import { getRetrieverDocumentPreviewText } from './getRetrieverDocumentPreviewText';

const text = `DOCUMENT-START café 🌳 ${'document body '.repeat(5000)}DOCUMENT-END`;
const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <DesignSystemProvider>
    <IntlProvider locale="en">{children}</IntlProvider>
  </DesignSystemProvider>
);

describe('getRetrieverDocumentPreviewText', () => {
  it('preserves the exact limit and truncates the first unit above it', () => {
    expect(getRetrieverDocumentPreviewText('x'.repeat(1000))).toBe('x'.repeat(1000));
    expect(getRetrieverDocumentPreviewText('x'.repeat(1001))).toBe(`${'x'.repeat(999)}…`);
  });
});

describe.each([
  ['legacy', LegacyDocument],
  ['redesigned', RedesignedDocument],
] as const)('%s retriever document', (_, Document) => {
  it('bounds the collapsed DOM text instead of laying out the complete document', () => {
    render(<Document text={text} metadata={{ source: 'fixture-source' }} />, { wrapper: Wrapper });
    const preview = screen.getByText(/^DOCUMENT-START/);
    expect(preview.textContent?.length).toBeLessThanOrEqual(1000);
    expect(preview.textContent).toMatch(/^DOCUMENT-START café 🌳 /);
    expect(preview.textContent).not.toContain('DOCUMENT-END');
    expect(screen.getByText(/fixture-source/)).toBeInTheDocument();
  });

  it('preserves the complete expanded Unicode document and supports collapse', async () => {
    render(<Document text={text} metadata={{ source: 'fixture-source' }} />, { wrapper: Wrapper });
    await userEvent.click(screen.getByText(/^DOCUMENT-START/));
    expect(screen.getByText(/^DOCUMENT-START/).textContent).toBe(text);
    expect(screen.getByText(/fixture-source/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'See less' }));
    expect(screen.queryByRole('button', { name: 'See less' })).not.toBeInTheDocument();
  });

  it('keeps short text unchanged', () => {
    render(<Document text="short café 🌳 document" metadata={{}} />, { wrapper: Wrapper });
    expect(screen.getByText('short café 🌳 document').textContent).toBe('short café 🌳 document');
  });

  it.each([null, undefined, false, 0])('preserves malformed document content %s without crashing', (value) => {
    render(<Document text={value as unknown as string} metadata={{ source: 'fixture-source' }} />, {
      wrapper: Wrapper,
    });
    expect(screen.getByText(/fixture-source/)).toBeInTheDocument();
    if (value === 0) expect(screen.getByText('0')).toBeInTheDocument();
  });

  it('does not cut an emoji surrogate pair at the preview boundary', () => {
    render(<Document text={`${'x'.repeat(998)}🌳tail`} metadata={{}} />, { wrapper: Wrapper });
    expect(screen.getByText(/^x+/).textContent).toBe(`${'x'.repeat(998)}…`);
  });
});
