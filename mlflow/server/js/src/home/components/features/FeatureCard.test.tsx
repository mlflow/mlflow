import { describe, expect, it, jest, beforeEach } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { FeatureCard } from './FeatureCard';
import { featureDefinitions } from './feature-definitions';

const mockNavigate = jest.fn();

jest.mock('../../../common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/RoutingUtils')>('../../../common/utils/RoutingUtils'),
  useNavigate: () => mockNavigate,
}));

jest.mock('../../../common/utils/FetchUtils', () => ({
  getAjaxUrl: (url: string) => `/${url}`,
}));

describe('FeatureCard', () => {
  beforeEach(() => {
    mockNavigate.mockClear();
    jest.restoreAllMocks();
  });

  const tracingFeature = featureDefinitions.find((f) => f.id === 'tracing')!;
  const experimentsFeature = featureDefinitions.find((f) => f.id === 'experiments')!;

  it('renders feature title and summary', () => {
    renderWithDesignSystem(<FeatureCard feature={tracingFeature} />);

    expect(screen.getByRole('heading', { level: 3, name: 'Tracing' })).toBeInTheDocument();
    expect(screen.getByText('Read the docs')).toBeInTheDocument();
  });

  it('shows "Explore demo" button for features with demoFeatureId', () => {
    renderWithDesignSystem(<FeatureCard feature={tracingFeature} />);

    expect(screen.getByText('Explore demo')).toBeInTheDocument();
  });

  it('shows "Go to" button for features without demoFeatureId', () => {
    renderWithDesignSystem(<FeatureCard feature={experimentsFeature} />);

    expect(screen.queryByText('Explore demo')).not.toBeInTheDocument();
  });

  it('navigates directly for features without demoFeatureId', async () => {
    renderWithDesignSystem(<FeatureCard feature={experimentsFeature} />);

    await userEvent.click(screen.getByRole('button', { name: /go to/i }));

    expect(mockNavigate).toHaveBeenCalledWith(experimentsFeature.navigationPath);
  });

  it('generates demo and navigates to traces page for tracing feature', async () => {
    // @ts-expect-error -- partial Response mock
    jest.spyOn(global, 'fetch').mockResolvedValueOnce({
      json: () => Promise.resolve({ experiment_id: '42' }),
    });

    renderWithDesignSystem(<FeatureCard feature={tracingFeature} />);

    await userEvent.click(screen.getByText('Explore demo'));

    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/experiments/42/traces');
    });
  });

  it('does not navigate on failed demo generation', async () => {
    jest.spyOn(global, 'fetch').mockRejectedValueOnce(new Error('Network error'));

    renderWithDesignSystem(<FeatureCard feature={tracingFeature} />);

    await userEvent.click(screen.getByText('Explore demo'));

    await waitFor(() => {
      expect(mockNavigate).not.toHaveBeenCalled();
    });
  });

  it('renders docs link with correct URL', () => {
    renderWithDesignSystem(<FeatureCard feature={tracingFeature} />);

    const docsLink = screen.getByText('Read the docs').closest('a');
    expect(docsLink).toHaveAttribute('href', tracingFeature.docsLink);
  });
});
