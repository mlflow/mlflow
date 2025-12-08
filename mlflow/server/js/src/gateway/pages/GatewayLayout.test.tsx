import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import { renderWithDesignSystem, screen } from '../../common/utils/TestUtils.react18';
import { MemoryRouter, Route, Routes } from '../../common/utils/RoutingUtils';
import { useEndpointsQuery } from '../hooks/useEndpointsQuery';
import GatewayLayout from './GatewayLayout';

jest.mock('../hooks/useEndpointsQuery');

describe('GatewayLayout', () => {
  beforeEach(() => {
    jest.clearAllMocks();
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
    expect(screen.getByText('Gateway')).toBeInTheDocument();
    // Side nav should NOT be visible during loading (only header + spinner)
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

    expect(screen.getByText('Gateway')).toBeInTheDocument();
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

    expect(screen.getByText('Gateway requires a SQL backend')).toBeInTheDocument();
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

    expect(screen.getByText('Gateway requires a SQL backend')).toBeInTheDocument();
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

    expect(screen.getByText('Gateway requires a SQL backend')).toBeInTheDocument();
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

    expect(screen.queryByText('Gateway requires a SQL backend')).not.toBeInTheDocument();
    expect(screen.getByText('Gateway')).toBeInTheDocument();
  });
});
