import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { ModelTraceExplorerAttachmentRenderer } from './ModelTraceExplorerAttachmentRenderer';

// Mock the fetch utility
const mockGetTraceAttachment = jest.fn();
jest.mock('../oss-notebook-renderer/mlflow-fetch-utils', () => ({
  getTraceAttachment: (...args) => mockGetTraceAttachment(...args),
}));

const renderWithProviders = (ui: React.ReactElement) =>
  render(
    <DesignSystemProvider>
      <IntlProvider locale="en">{ui}</IntlProvider>
    </DesignSystemProvider>,
  );

describe('ModelTraceExplorerAttachmentRenderer', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Mock URL.createObjectURL / revokeObjectURL
    global.URL.createObjectURL = jest.fn(() => 'blob:mock-url');
    global.URL.revokeObjectURL = jest.fn();
  });

  it('shows spinner while loading', () => {
    mockGetTraceAttachment.mockReturnValue(new Promise(() => {})); // never resolves
    renderWithProviders(
      <ModelTraceExplorerAttachmentRenderer
        title="Test"
        attachmentId="abc-123"
        traceId="tr-456"
        contentType="image/png"
      />,
    );
    expect(screen.getByRole('img', { hidden: true })).toBeInTheDocument(); // Spinner renders as img
  });

  it('renders an image for image content types', async () => {
    const mockData = new ArrayBuffer(8);
    mockGetTraceAttachment.mockResolvedValue(mockData);
    renderWithProviders(
      <ModelTraceExplorerAttachmentRenderer
        title="Test Image"
        attachmentId="abc-123"
        traceId="tr-456"
        contentType="image/png"
      />,
    );
    await waitFor(() => {
      expect(screen.getByAltText('Attachment abc-123')).toBeInTheDocument();
    });
    expect(screen.getByText('Test Image')).toBeInTheDocument();
  });

  it('renders audio element for audio content types', async () => {
    const mockData = new ArrayBuffer(8);
    mockGetTraceAttachment.mockResolvedValue(mockData);
    renderWithProviders(
      <ModelTraceExplorerAttachmentRenderer
        title="Test Audio"
        attachmentId="abc-123"
        traceId="tr-456"
        contentType="audio/wav"
      />,
    );
    await waitFor(() => {
      expect(screen.getByText('Test Audio')).toBeInTheDocument();
    });
  });

  it('renders download link for unknown content types', async () => {
    const mockData = new ArrayBuffer(8);
    mockGetTraceAttachment.mockResolvedValue(mockData);
    renderWithProviders(
      <ModelTraceExplorerAttachmentRenderer
        title="Test File"
        attachmentId="abc-123"
        traceId="tr-456"
        contentType="application/octet-stream"
      />,
    );
    await waitFor(() => {
      expect(screen.getByRole('link')).toBeInTheDocument();
    });
  });

  it('shows error message when fetch returns undefined', async () => {
    mockGetTraceAttachment.mockResolvedValue(undefined);
    renderWithProviders(
      <ModelTraceExplorerAttachmentRenderer
        title="Test"
        attachmentId="abc-123"
        traceId="tr-456"
        contentType="image/png"
      />,
    );
    await waitFor(() => {
      expect(screen.getByText('Failed to load attachment')).toBeInTheDocument();
    });
  });

  // Note: getTraceAttachment catches errors internally and returns undefined,
  // so rejection is handled by the "fetch returns undefined" test above.
});
