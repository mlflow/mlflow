import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from '@databricks/i18n';
import { DesignSystemProvider } from '@databricks/design-system';
import { useForm } from 'react-hook-form';
import EvaluateTracesSection from './EvaluateTracesSection';
import { SCORER_FORM_MODE, ScorerEvaluationScope } from './constants';
import { LLM_TEMPLATE } from './types';

describe('EvaluateTracesSection', () => {
  const TestWrapper = ({ defaultValues = {}, mode = SCORER_FORM_MODE.CREATE }: { defaultValues?: any; mode?: any }) => {
    const { control, setValue } = useForm({ defaultValues });
    return (
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <EvaluateTracesSection control={control} mode={mode} setValue={setValue} />
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
      expect(screen.getByText(/Automatically evaluate new traces using this scorer/i)).toBeInTheDocument();
    });
  });

  describe('Filter box for different evaluation scopes', () => {
    it('should show filter input with session-specific help text when evaluationScope is SESSIONS', async () => {
      render(
        <TestWrapper
          defaultValues={{
            evaluationScope: ScorerEvaluationScope.SESSIONS,
            sampleRate: 100,
          }}
        />,
      );

      // Click the Advanced settings toggle to reveal filter fields
      await userEvent.click(screen.getByText(/Advanced settings/i));

      expect(screen.getByText(/Filter string/i)).toBeInTheDocument();
      expect(screen.getByText(/Filter applies to the first trace in each session/i)).toBeInTheDocument();
    });

    it('should show filter input with trace-specific help text when evaluationScope is TRACES', async () => {
      render(
        <TestWrapper
          defaultValues={{
            evaluationScope: ScorerEvaluationScope.TRACES,
            sampleRate: 100,
          }}
        />,
      );

      // Click the Advanced settings toggle to reveal filter fields
      await userEvent.click(screen.getByText(/Advanced settings/i));

      expect(screen.getByText(/Filter string/i)).toBeInTheDocument();
      expect(screen.getByText(/Only run on traces matching this filter/i)).toBeInTheDocument();
    });
  });

  describe('Automatic evaluation controls', () => {
    it('should hide sample rate and filter when sampleRate is 0', () => {
      render(<TestWrapper defaultValues={{ sampleRate: 0 }} />);

      expect(screen.queryByText(/Sample rate/i)).not.toBeInTheDocument();
      expect(screen.queryByText(/Filter string/i)).not.toBeInTheDocument();
    });

    it('should show sample rate and filter when sampleRate > 0', async () => {
      render(
        <TestWrapper
          defaultValues={{
            sampleRate: 50,
            evaluationScope: ScorerEvaluationScope.TRACES,
          }}
        />,
      );

      // Click the Advanced settings toggle to reveal sample rate and filter fields
      await userEvent.click(screen.getByText(/Advanced settings/i));

      expect(screen.getByText(/Sample rate/i)).toBeInTheDocument();
      expect(screen.getByText(/Filter string/i)).toBeInTheDocument();
    });

    it('should disable and uncheck automatic evaluation when instructions contain expectations', () => {
      render(
        <TestWrapper
          defaultValues={{
            sampleRate: 100,
            instructions: 'Compare {{ outputs }} against {{ expectations }}',
          }}
        />,
      );

      expect(screen.getByText(/not available for judges that use expectations/i)).toBeInTheDocument();
      expect(screen.getByRole('switch')).toBeDisabled();
      // sampleRate is set to 0, so sample rate and filter controls should be hidden
      expect(screen.queryByText(/Sample rate/i)).not.toBeInTheDocument();
      expect(screen.queryByText(/Filter string/i)).not.toBeInTheDocument();
    });

    it('should handle expectations with variable whitespace in template', () => {
      render(
        <TestWrapper
          defaultValues={{
            sampleRate: 100,
            instructions: 'Compare {{ outputs }} against {{   expectations   }}',
          }}
        />,
      );

      expect(screen.getByText(/not available for judges that use expectations/i)).toBeInTheDocument();
      expect(screen.getByRole('switch')).toBeDisabled();
    });

    it('should disable automatic evaluation for built-in templates that require expectations', () => {
      render(
        <TestWrapper
          defaultValues={{
            sampleRate: 100,
            llmTemplate: LLM_TEMPLATE.CORRECTNESS,
          }}
        />,
      );

      expect(screen.getByText(/not available for judges that use expectations/i)).toBeInTheDocument();
      expect(screen.getByRole('switch')).toBeDisabled();
      expect(screen.queryByText(/Sample rate/i)).not.toBeInTheDocument();
    });

    it('should re-enable automatic evaluation when switching from expectations template to non-expectations template', async () => {
      const ReenableWrapper = () => {
        const { control, setValue } = useForm({
          defaultValues: {
            sampleRate: 100,
            llmTemplate: LLM_TEMPLATE.CORRECTNESS,
          },
        });
        return (
          <IntlProvider locale="en">
            <DesignSystemProvider>
              <EvaluateTracesSection control={control} mode={SCORER_FORM_MODE.CREATE} setValue={setValue} />
              <button onClick={() => setValue('llmTemplate', LLM_TEMPLATE.RELEVANCE_TO_QUERY)}>
                Switch to Relevance
              </button>
            </DesignSystemProvider>
          </IntlProvider>
        );
      };

      render(<ReenableWrapper />);

      // Initially disabled because Correctness requires expectations
      expect(screen.getByRole('switch')).toBeDisabled();
      expect(screen.getByText(/not available for judges that use expectations/i)).toBeInTheDocument();

      // Switch to a template that doesn't require expectations
      await userEvent.click(screen.getByText('Switch to Relevance'));

      // Should re-enable automatic evaluation
      expect(screen.getByRole('switch')).not.toBeDisabled();
      expect(screen.queryByText(/not available for judges that use expectations/i)).not.toBeInTheDocument();
    });

    it('should disable automatic evaluation when using a non-gateway model', () => {
      render(
        <TestWrapper
          defaultValues={{
            sampleRate: 0,
            model: 'openai:/gpt-4.1',
          }}
        />,
      );

      expect(screen.getByText(/only available for judges that use gateway endpoints/i)).toBeInTheDocument();
      expect(screen.getByRole('switch')).toBeDisabled();
    });
  });
});
