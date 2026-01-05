import { render, screen } from '@testing-library/react';
import { IntlProvider } from '@databricks/i18n';
import { DesignSystemProvider } from '@databricks/design-system';
import { useForm } from 'react-hook-form';
import EvaluateTracesSectionRenderer from './EvaluateTracesSectionRenderer';
import { SCORER_FORM_MODE, ScorerEvaluationScope } from './constants';

describe('EvaluateTracesSectionRenderer', () => {
  const TestWrapper = ({
    defaultValues = {},
    mode = SCORER_FORM_MODE.CREATE,
  }: {
    defaultValues?: any;
    mode?: any;
  }) => {
    const { control } = useForm({ defaultValues });
    return (
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <EvaluateTracesSectionRenderer control={control} mode={mode} />
        </DesignSystemProvider>
      </IntlProvider>
    );
  };

  describe('Section visibility', () => {
    it('should hide entire section when disableMonitoring is true', () => {
      const { container } = render(<TestWrapper defaultValues={{ disableMonitoring: true }} />);
      expect(container.firstChild).toBeNull();
    });

    it('should show section when disableMonitoring is false', () => {
      render(<TestWrapper defaultValues={{ disableMonitoring: false, sampleRate: 0 }} />);
      expect(screen.getByText(/Automatically evaluate future traces/i)).toBeInTheDocument();
    });
  });

  describe('Filter box visibility for session-level scorers', () => {
    it('should hide filter input when evaluationScope is SESSIONS', () => {
      render(
        <TestWrapper
          defaultValues={{
            evaluationScope: ScorerEvaluationScope.SESSIONS,
            sampleRate: 100,
          }}
        />,
      );

      expect(screen.queryByText(/Filter string/i)).not.toBeInTheDocument();
    });

    it('should show filter input when evaluationScope is TRACES', () => {
      render(
        <TestWrapper
          defaultValues={{
            evaluationScope: ScorerEvaluationScope.TRACES,
            sampleRate: 100,
          }}
        />,
      );

      expect(screen.getByText(/Filter string/i)).toBeInTheDocument();
    });
  });

  describe('Automatic evaluation controls', () => {
    it('should hide sample rate and filter when sampleRate is 0', () => {
      render(<TestWrapper defaultValues={{ sampleRate: 0 }} />);

      expect(screen.queryByText(/Sample rate/i)).not.toBeInTheDocument();
      expect(screen.queryByText(/Filter string/i)).not.toBeInTheDocument();
    });

    it('should show sample rate and filter when sampleRate > 0', () => {
      render(
        <TestWrapper
          defaultValues={{
            sampleRate: 50,
            evaluationScope: ScorerEvaluationScope.TRACES,
          }}
        />,
      );

      expect(screen.getByText(/Sample rate/i)).toBeInTheDocument();
      expect(screen.getByText(/Filter string/i)).toBeInTheDocument();
    });
  });
});
