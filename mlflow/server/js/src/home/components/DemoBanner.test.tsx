import { describe, expect, it, jest, beforeEach } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { DemoBanner } from './DemoBanner';

const mockNavigate = jest.fn();

jest.mock('../../common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('../../common/utils/RoutingUtils')>('../../common/utils/RoutingUtils'),
  useNavigate: () => mockNavigate,
}));

jest.mock('../../common/utils/FetchUtils', () => ({
  getAjaxUrl: (url: string) => `/${url}`,
}));

describe('DemoBanner', () => {
  beforeEach(() => {
    mockNavigate.mockClear();
    localStorage.clear();
    jest.restoreAllMocks();
  });

  it('renders banner with title and launch button', () => {
    renderWithDesignSystem(<DemoBanner />);

    expect(screen.getByText('New to MLflow?')).toBeInTheDocument();
    expect(screen.getByText('Launch Demo')).toBeInTheDocument();
  });

  it('dismisses banner on close and persists in localStorage', async () => {
    renderWithDesignSystem(<DemoBanner />);

    expect(screen.getByText('New to MLflow?')).toBeInTheDocument();

    await userEvent.click(screen.getByLabelText('Dismiss'));

    expect(screen.queryByText('New to MLflow?')).not.toBeInTheDocument();
    expect(localStorage.getItem('mlflow.demo.banner.dismissed')).toBe('true');
  });

  it('does not render when previously dismissed', () => {
    localStorage.setItem('mlflow.demo.banner.dismissed', 'true');

    renderWithDesignSystem(<DemoBanner />);

    expect(screen.queryByText('New to MLflow?')).not.toBeInTheDocument();
  });

  it('navigates to returned URL on successful demo generation', async () => {
    // @ts-expect-error -- partial Response mock
    jest.spyOn(global, 'fetch').mockResolvedValueOnce({
      json: () => Promise.resolve({ navigation_url: '/experiments/123/traces' }),
    });

    renderWithDesignSystem(<DemoBanner />);

    await userEvent.click(screen.getByText('Launch Demo'));

    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/experiments/123/traces');
    });
  });

  it('does not navigate on failed demo generation', async () => {
    jest.spyOn(global, 'fetch').mockRejectedValueOnce(new Error('Network error'));

    renderWithDesignSystem(<DemoBanner />);

    await userEvent.click(screen.getByText('Launch Demo'));

    await waitFor(() => {
      expect(mockNavigate).not.toHaveBeenCalled();
    });
  });
});
