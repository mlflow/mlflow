import { describe, expect, it, jest } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { FeatureCard } from './FeatureCard';
import { featureDefinitions } from './feature-definitions';

const mockOpenLogTracesDrawer = jest.fn();

jest.mock('../../HomePageViewStateContext', () => ({
  useHomePageViewState: () => ({
    openLogTracesDrawer: mockOpenLogTracesDrawer,
  }),
}));

describe('FeatureCard', () => {
  const tracingFeature = featureDefinitions.find((f) => f.id === 'tracing')!;
  const evaluationFeature = featureDefinitions.find((f) => f.id === 'evaluation')!;

  it('renders feature title and summary', () => {
    renderWithDesignSystem(<FeatureCard feature={tracingFeature} />);

    expect(screen.getByRole('heading', { level: 2, name: 'Tracing' })).toBeInTheDocument();
  });

  it('renders as a button and opens drawer for features with hasDrawer', async () => {
    renderWithDesignSystem(<FeatureCard feature={tracingFeature} />);

    await userEvent.click(screen.getByRole('button'));

    expect(mockOpenLogTracesDrawer).toHaveBeenCalled();
  });

  it('renders as a link for features without hasDrawer', () => {
    renderWithDesignSystem(<FeatureCard feature={evaluationFeature} />);

    const link = screen.getByRole('link');
    expect(link).toHaveAttribute('href', evaluationFeature.docsLink);
    expect(link).toHaveAttribute('target', '_blank');
  });
});
