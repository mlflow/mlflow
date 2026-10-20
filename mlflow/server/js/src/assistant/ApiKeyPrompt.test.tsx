import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import { screen } from '@testing-library/react';
import { renderWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { DesignSystemProvider } from '@databricks/design-system';

import { ApiKeyPrompt } from './ApiKeyPrompt';

let mockIsLocalServer = true;
let mockCanEditServerSettings = true;

jest.mock('./AssistantContext', () => ({
  useAssistant: () => ({ isLocalServer: mockIsLocalServer, canEditServerSettings: mockCanEditServerSettings }),
}));

jest.mock('./AssistantService', () => ({
  updateConfig: jest.fn(),
}));

const renderPrompt = () =>
  renderWithIntl(
    <DesignSystemProvider>
      <ApiKeyPrompt providerId="mlflow_gateway" providerName="OpenAI" gatewayVendor="openai" onSaved={jest.fn()} />
    </DesignSystemProvider>,
  );

describe('ApiKeyPrompt', () => {
  beforeEach(() => {
    mockIsLocalServer = true;
    mockCanEditServerSettings = true;
  });

  test('asks for the key when the caller can add server-wide connections', () => {
    renderPrompt();

    expect(screen.getByPlaceholderText('sk-...')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Continue' })).toBeInTheDocument();
  });

  test('tells a non-admin on the server host to ask an administrator', () => {
    mockCanEditServerSettings = false;
    renderPrompt();

    expect(screen.getByText(/Only an administrator can add the OpenAI API key/)).toBeInTheDocument();
    expect(screen.queryByPlaceholderText('sk-...')).not.toBeInTheDocument();
  });

  test('tells a remote user that keys are added from the server host', () => {
    mockIsLocalServer = false;
    mockCanEditServerSettings = false;
    renderPrompt();

    expect(screen.getByText(/can only be added from the MLflow server host/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Continue' })).not.toBeInTheDocument();
  });
});
