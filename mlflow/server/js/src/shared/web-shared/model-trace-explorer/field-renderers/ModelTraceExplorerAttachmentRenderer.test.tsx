import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { QueryClient, QueryClientProvider } from '../../query-client/queryClient';

import { ModelTraceExplorerAttachmentRenderer } from './ModelTraceExplorerAttachmentRenderer';

// Mock the useTraceAttachment hook
const mockUseTraceAttachment = jest.fn<() => { objectUrl: string | null; isLoading: boolean; error: unknown }>();
jest.mock('../hooks/useTraceAttachment', () => ({
  useTraceAttachment: () => mockUseTraceAttachment(),
}));

const renderWithProviders = (ui: React.ReactElement) => {
  const queryClient = new QueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <DesignSystemProvider>
        <IntlProvider locale="en">{ui}</IntlProvider>
      </DesignSystemProvider>
    </QueryClientProvider>,
  );
};

describe('ModelTraceExplorerAttachmentRenderer', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('shows spinner while loading', () => {
    mockUseTraceAttachment.mockReturnValue({ objectUrl: null, isLoading: true, error: null });
    renderWithProviders(
      <ModelTraceExplorerAttachmentRenderer
        title="Test"
        attachmentId="abc-123"
        traceId="tr-456"
        contentType="image/png"
      />,
    );
    expect(screen.getByRole('heading', { hidden: true })).toBeInTheDocument(); // LegacySkeleton renders as heading
  });

  it('renders an image for image content types', () => {
    mockUseTraceAttachment.mockReturnValue({ objectUrl: 'blob:mock-url', isLoading: false, error: null });
    renderWithProviders(
      <ModelTraceExplorerAttachmentRenderer
        title="Test Image"
        attachmentId="abc-123"
        traceId="tr-456"
        contentType="image/png"
      />,
    );
    expect(screen.getByAltText('Attachment abc-123')).toBeInTheDocument();
    expect(screen.getByText('Test Image')).toBeInTheDocument();
  });

  it('renders audio element for audio content types', () => {
    mockUseTraceAttachment.mockReturnValue({ objectUrl: 'blob:mock-url', isLoading: false, error: null });
    renderWithProviders(
      <ModelTraceExplorerAttachmentRenderer
        title="Test Audio"
        attachmentId="abc-123"
        traceId="tr-456"
        contentType="audio/wav"
      />,
    );
    expect(screen.getByText('Test Audio')).toBeInTheDocument();
  });

  it('renders download link for unknown content types', () => {
    mockUseTraceAttachment.mockReturnValue({ objectUrl: 'blob:mock-url', isLoading: false, error: null });
    renderWithProviders(
      <ModelTraceExplorerAttachmentRenderer
        title="Test File"
        attachmentId="abc-123"
        traceId="tr-456"
        contentType="application/octet-stream"
      />,
    );
    expect(screen.getByRole('link')).toBeInTheDocument();
  });

  it('shows error message when fetch fails', () => {
    mockUseTraceAttachment.mockReturnValue({ objectUrl: null, isLoading: false, error: new Error('fail') });
    renderWithProviders(
      <ModelTraceExplorerAttachmentRenderer
        title="Test"
        attachmentId="abc-123"
        traceId="tr-456"
        contentType="image/png"
      />,
    );
    expect(screen.getByText('Failed to load attachment')).toBeInTheDocument();
  });
});
