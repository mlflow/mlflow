import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithIntl } from '../common/utils/TestUtils.react18';
import WebhooksSettings from './WebhooksSettings';
import { DesignSystemProvider } from '@databricks/design-system';

jest.mock('./webhooksApi', () => ({
  WebhooksApi: {
    listWebhooks: jest.fn(() => Promise.resolve({ webhooks: [] })),
    createWebhook: jest.fn(() => Promise.resolve({})),
    updateWebhook: jest.fn(() => Promise.resolve({})),
    deleteWebhook: jest.fn(() => Promise.resolve()),
    testWebhook: jest.fn(() => Promise.resolve({ result: { success: true, response_status: 200 } })),
  },
}));

import { WebhooksApi } from './webhooksApi';

const mockListWebhooks = WebhooksApi.listWebhooks as jest.MockedFunction<typeof WebhooksApi.listWebhooks>;
const mockCreateWebhook = WebhooksApi.createWebhook as jest.MockedFunction<typeof WebhooksApi.createWebhook>;
const mockDeleteWebhook = WebhooksApi.deleteWebhook as jest.MockedFunction<typeof WebhooksApi.deleteWebhook>;
const mockTestWebhook = WebhooksApi.testWebhook as jest.MockedFunction<typeof WebhooksApi.testWebhook>;

const SAMPLE_WEBHOOK = {
  webhook_id: 'wh-1',
  name: 'Test Webhook',
  url: 'https://example.com/hook',
  events: [{ entity: 'PROMPT', action: 'CREATED' }],
  status: 'ACTIVE',
  creation_timestamp: '1700000000000',
  last_updated_timestamp: '1700000000000',
  description: 'A test webhook',
};

describe('WebhooksSettings', () => {
  const renderComponent = () =>
    renderWithIntl(
      <DesignSystemProvider>
        <WebhooksSettings />
      </DesignSystemProvider>,
    );

  beforeEach(() => {
    jest.clearAllMocks();
    mockListWebhooks.mockResolvedValue({ webhooks: [] });
  });

  it('renders empty state when no webhooks exist', async () => {
    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('No webhooks configured. Create one to get started.')).toBeInTheDocument();
    });
  });

  it('renders webhook list', async () => {
    mockListWebhooks.mockResolvedValue({ webhooks: [SAMPLE_WEBHOOK] });

    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('Test Webhook')).toBeInTheDocument();
    });
    expect(screen.getByText('https://example.com/hook')).toBeInTheDocument();
    expect(screen.getByText('Active')).toBeInTheDocument();
  });

  it('shows validation errors when creating webhook with empty fields', async () => {
    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('Create webhook')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByText('Create webhook'));

    // Click Create in modal without filling fields
    const createButtons = screen.getAllByText('Create');
    await userEvent.click(createButtons[createButtons.length - 1]);

    await waitFor(() => {
      expect(screen.getByText('Name is required')).toBeInTheDocument();
    });
  });

  it('creates a webhook successfully', async () => {
    mockCreateWebhook.mockResolvedValue({} as any);

    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('Create webhook')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByText('Create webhook'));

    // Fill in name and URL
    const nameInput = screen.getByPlaceholderText('My webhook');
    const urlInput = screen.getByPlaceholderText('https://example.com/webhook');

    await userEvent.type(nameInput, 'New Webhook');
    await userEvent.type(urlInput, 'https://test.com/hook');

    // Select an event
    await userEvent.click(screen.getByText('Prompt created'));

    // Submit
    const createButtons = screen.getAllByText('Create');
    const modalCreateButton = createButtons.find((btn) => btn.closest('button[type]') || btn.closest('.ant-modal'));
    await userEvent.click(modalCreateButton ?? createButtons[createButtons.length - 1]);

    await waitFor(() => {
      expect(mockCreateWebhook).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'New Webhook',
          url: 'https://test.com/hook',
          events: [{ entity: 'PROMPT', action: 'CREATED' }],
          status: 'ACTIVE',
        }),
      );
    });
  });

  it('deletes a webhook with confirmation', async () => {
    mockListWebhooks.mockResolvedValue({ webhooks: [SAMPLE_WEBHOOK] });
    mockDeleteWebhook.mockResolvedValue(undefined);

    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('Test Webhook')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByText('Delete'));

    // Confirmation modal should appear
    await waitFor(() => {
      expect(screen.getByText(/Are you sure you want to delete the webhook "Test Webhook"\?/)).toBeInTheDocument();
    });

    // Confirm deletion
    const deleteButtons = screen.getAllByText('Delete');
    const confirmButton = deleteButtons[deleteButtons.length - 1];
    await userEvent.click(confirmButton);

    await waitFor(() => {
      expect(mockDeleteWebhook).toHaveBeenCalledWith('wh-1');
    });
  });

  it('tests a webhook and shows result', async () => {
    mockListWebhooks.mockResolvedValue({ webhooks: [SAMPLE_WEBHOOK] });
    mockTestWebhook.mockResolvedValue({ result: { success: true, response_status: 200 } });

    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('Test Webhook')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByText('Test'));

    await waitFor(() => {
      expect(mockTestWebhook).toHaveBeenCalledWith('wh-1');
    });

    await waitFor(() => {
      expect(screen.getByText('Test succeeded (HTTP 200)')).toBeInTheDocument();
    });
  });

  it('shows error when loading webhooks fails', async () => {
    mockListWebhooks.mockRejectedValue(new Error('Network error'));

    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('Network error')).toBeInTheDocument();
    });
  });
});
