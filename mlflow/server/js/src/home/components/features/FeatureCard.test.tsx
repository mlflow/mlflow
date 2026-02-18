import { describe, expect, it } from '@jest/globals';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';
import { FeatureCard } from './FeatureCard';
import { featureDefinitions } from './feature-definitions';

describe('FeatureCard', () => {
  const tracingFeature = featureDefinitions.find((f) => f.id === 'tracing')!;
  const evaluationFeature = featureDefinitions.find((f) => f.id === 'evaluation')!;

  it('renders feature title and summary', () => {
    renderWithDesignSystem(
      <MemoryRouter>
        <FeatureCard feature={tracingFeature} />
      </MemoryRouter>,
    );

    expect(screen.getByRole('heading', { level: 3, name: 'Tracing' })).toBeInTheDocument();
  });

  it('renders explore demo button for features with demoFeatureId', () => {
    renderWithDesignSystem(
      <MemoryRouter>
        <FeatureCard feature={tracingFeature} />
      </MemoryRouter>,
    );

    expect(screen.getByRole('button', { name: /explore demo/i })).toBeInTheDocument();
  });

  it('renders go to button for features without demoFeatureId', () => {
    const featureWithoutDemo = { ...evaluationFeature, demoFeatureId: undefined };
    renderWithDesignSystem(
      <MemoryRouter>
        <FeatureCard feature={featureWithoutDemo} />
      </MemoryRouter>,
    );

    expect(screen.getByRole('button', { name: /go to/i })).toBeInTheDocument();
  });

  it('renders docs link', () => {
    renderWithDesignSystem(
      <MemoryRouter>
        <FeatureCard feature={tracingFeature} />
      </MemoryRouter>,
    );

    expect(screen.getByRole('link', { name: /read the docs/i })).toHaveAttribute('href', tracingFeature.docsLink);
  });
});
