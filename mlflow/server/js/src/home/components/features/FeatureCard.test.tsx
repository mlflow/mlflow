import { describe, expect, it, jest } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';
import { FeatureCard } from './FeatureCard';
import { featureDefinitions } from './feature-definitions';

const mockOpenLogTracesDrawer = jest.fn();

jest.mock('../../HomePageViewStateContext', () => ({
  useHomePageViewState: () => ({
    openLogTracesDrawer: mockOpenLogTracesDrawer,
  }),
}));

const renderWithRouter = (ui: React.ReactElement) => renderWithDesignSystem(<MemoryRouter>{ui}</MemoryRouter>);

describe('FeatureCard', () => {
  const tracingFeature = featureDefinitions.find((f) => f.id === 'tracing')!;
  const evaluationFeature = featureDefinitions.find((f) => f.id === 'evaluation')!;

  it('renders feature title and summary', () => {
    renderWithRouter(<FeatureCard feature={tracingFeature} componentId="" />);

    expect(screen.getByRole('heading', { level: 2, name: 'Tracing' })).toBeInTheDocument();
  });

  it('opens drawer on click for features with hasDrawer', async () => {
    renderWithRouter(<FeatureCard feature={tracingFeature} componentId="" />);

    await userEvent.click(screen.getByRole('link'));

    expect(mockOpenLogTracesDrawer).toHaveBeenCalled();
  });

  it('renders as a link for features without hasDrawer', () => {
    renderWithRouter(<FeatureCard feature={evaluationFeature} componentId="" />);

    const link = screen.getByRole('link');
    expect(link).toHaveAttribute('href', evaluationFeature.docsLink);
    expect(link).toHaveAttribute('target', '_blank');
  });
});
