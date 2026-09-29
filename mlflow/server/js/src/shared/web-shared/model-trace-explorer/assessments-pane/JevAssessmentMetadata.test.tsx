import { describe, expect, it } from '@jest/globals';
import { render, screen } from '@databricks/web-shared/test-utils/render';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { JevAssessmentMetadata } from './JevAssessmentMetadata';

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

describe('JevAssessmentMetadata', () => {
  it('shows the original noul probability even when the decision is thresholded', () => {
    render(<JevAssessmentMetadata metadata={{ 'jev.model': 'jev-latest', 'jev.probability': '0.65' }} />, {
      wrapper: Wrapper,
    });

    expect(screen.getByText('Probability of true')).toBeInTheDocument();
    expect(screen.getByText('65%')).toBeInTheDocument();
    expect(screen.queryByText('Confidence')).not.toBeInTheDocument();
  });

  it('shows choice probabilities and confidence', () => {
    render(
      <JevAssessmentMetadata
        metadata={{
          'jev.model': 'jev-latest',
          'jev.probabilities': '{"yes":0.8,"no":0.2}',
          'jev.confidence': '0.6',
        }}
      />,
      { wrapper: Wrapper },
    );

    expect(screen.getByText('yes')).toBeInTheDocument();
    expect(screen.getByText('80%')).toBeInTheDocument();
    expect(screen.getByText('no')).toBeInTheDocument();
    expect(screen.getByText('20%')).toBeInTheDocument();
    expect(screen.getByText('Confidence')).toBeInTheDocument();
    expect(screen.getByText('60%')).toBeInTheDocument();
  });

  it('shows score probabilities with their legend', () => {
    render(
      <JevAssessmentMetadata
        metadata={{
          'jev.model': 'jev-latest',
          'jev.probabilities': '{"0":0.2,"1":0.8}',
          'jev.legend': '{"0":"Incorrect","1":"Correct"}',
          'jev.confidence': '0.6',
        }}
      />,
      { wrapper: Wrapper },
    );

    expect(screen.getByText('0: Incorrect')).toBeInTheDocument();
    expect(screen.getByText('1: Correct')).toBeInTheDocument();
    expect(screen.getByText('20%')).toBeInTheDocument();
    expect(screen.getByText('80%')).toBeInTheDocument();
  });

  it('ignores malformed values and non-Jev metadata', () => {
    const { container, rerender } = render(
      <JevAssessmentMetadata
        metadata={{
          'jev.model': 'jev-latest',
          'jev.probabilities': 'invalid',
          'jev.probability': 'NaN',
          'jev.confidence': '1.2',
        }}
      />,
      { wrapper: Wrapper },
    );
    expect(container.querySelector('dl')).toBeNull();

    rerender(<JevAssessmentMetadata metadata={{ 'jev.probability': '0.65' }} />);
    expect(container.querySelector('dl')).toBeNull();
  });
});
