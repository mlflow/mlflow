import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import { renderWithDesignSystem, screen } from '../../common/utils/TestUtils.react18';
import { MemoryRouter, Route, Routes } from '../../common/utils/RoutingUtils';
import { useEndpointsQuery } from '../hooks/useEndpointsQuery';
import { useSecretsConfigQuery } from '../hooks/useSecretsConfigQuery';
import GatewayLayout from './GatewayLayout';

jest.mock('../hooks/useEndpointsQuery');
jest.mock('../hooks/useSecretsConfigQuery');

const mockSecretsConfigAvailable = () => {
  jest.mocked(useSecretsConfigQuery).mockReturnValue({
    secretsAvailable: true,
    isLoading: false,
    error: undefined,
  });
};

const mockSecretsConfigLoading = () => {
  jest.mocked(useSecretsConfigQuery).mockReturnValue({
    secretsAvailable: true,
    isLoading: true,
    error: undefined,
  });
};

const mockSecretsConfigUnavailable = () => {
  jest.mocked(useSecretsConfigQuery).mockReturnValue({
    secretsAvailable: false,
    isLoading: false,
    error: undefined,
  });
};

describe('GatewayLayout', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockSecretsConfigAvailable();
  });

  test('renders loading state while checking backend support', () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: undefined,
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(
      <MemoryRouter initialEntries={['/gateway']}>
        <Routes>
          <Route path="/gateway" element={<GatewayLayout />} />
        </Routes>
      </MemoryRouter>,
    );

    // Header should be visible during loading
    expect(screen.getByText('AI Gateway')).toBeInTheDocument();
    // Side nav should NOT be visible during loading (only header + spinner)
    expect(screen.queryByText('Endpoints')).not.toBeInTheDocument();
  });

  test('renders loading state while checking secrets config', () => {
    mockSecretsConfigLoading();
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: [],
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(
      <MemoryRouter initialEntries={['/gateway']}>
        <Routes>
          <Route path="/gateway" element={<GatewayLayout />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(screen.getByText('AI Gateway')).toBeInTheDocument();
    expect(screen.queryByText('Endpoints')).not.toBeInTheDocument();
  });

  test('renders gateway layout with child content when backend supports gateway', () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: [],
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(
      <MemoryRouter initialEntries={['/gateway']}>
        <Routes>
          <Route path="/gateway" element={<GatewayLayout />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(screen.getByText('AI Gateway')).toBeInTheDocument();
  });

  test('shows unsupported backend message for NotImplementedError', () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error('NotImplementedError: FileStore'),
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(
      <MemoryRouter initialEntries={['/gateway']}>
        <Routes>
          <Route path="/gateway" element={<GatewayLayout />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(screen.getByText('AI Gateway requires a SQL backend')).toBeInTheDocument();
    expect(screen.getByText(/SQL-based tracking store/)).toBeInTheDocument();
  });

  test('shows unsupported backend message for FileStore error', () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error('FileStore does not support this operation'),
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(
      <MemoryRouter initialEntries={['/gateway']}>
        <Routes>
          <Route path="/gateway" element={<GatewayLayout />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(screen.getByText('AI Gateway requires a SQL backend')).toBeInTheDocument();
  });

  test('shows unsupported backend message for Internal Server Error', () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error('Internal Server Error'),
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(
      <MemoryRouter initialEntries={['/gateway']}>
        <Routes>
          <Route path="/gateway" element={<GatewayLayout />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(screen.getByText('AI Gateway requires a SQL backend')).toBeInTheDocument();
  });

  test('does not show unsupported message for other errors', () => {
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error('Network error'),
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(
      <MemoryRouter initialEntries={['/gateway']}>
        <Routes>
          <Route path="/gateway" element={<GatewayLayout />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(screen.queryByText('AI Gateway requires a SQL backend')).not.toBeInTheDocument();
    expect(screen.getByText('AI Gateway')).toBeInTheDocument();
  });

  test('shows secrets setup guide when encryption is not configured', () => {
    mockSecretsConfigUnavailable();
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: [],
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(
      <MemoryRouter initialEntries={['/gateway']}>
        <Routes>
          <Route path="/gateway" element={<GatewayLayout />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(screen.getByText('Encryption Required')).toBeInTheDocument();
    expect(screen.getByText(/MLFLOW_CRYPTO_KEK_PASSPHRASE/)).toBeInTheDocument();
    expect(screen.getByText('Security Requirements')).toBeInTheDocument();
  });

  test('shows normal layout when secrets are available', () => {
    mockSecretsConfigAvailable();
    jest.mocked(useEndpointsQuery).mockReturnValue({
      data: [],
      isLoading: false,
      error: undefined,
      refetch: jest.fn(),
    } as any);

    renderWithDesignSystem(
      <MemoryRouter initialEntries={['/gateway']}>
        <Routes>
          <Route path="/gateway" element={<GatewayLayout />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(screen.queryByText('Encryption Required')).not.toBeInTheDocument();
    expect(screen.getByText('AI Gateway')).toBeInTheDocument();
  });
});
