import { useEffect, useRef, useState } from 'react';
import { keyframes } from '@emotion/react';
import { Alert, ArrowRightIcon, BarChartIcon, Button, Tag, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import type { GenAIOverviewChartContext } from '../genAIOverview.types';

const placeholderSlideIn = keyframes({
  from: { opacity: 0, transform: 'translateY(8px)' },
  to: { opacity: 1, transform: 'translateY(0)' },
});

const contextBorderPulse = keyframes({
  '0%, 100%': { opacity: 0 },
  '30%': { opacity: 1 },
});

export interface GenAIOverviewPromptBoxProps {
  chartContext?: GenAIOverviewChartContext;
  onClearChartContext: () => void;
  onSubmit: (
    prompt: string,
    chartContext?: GenAIOverviewChartContext,
  ) => Promise<'submitted' | 'unavailable' | 'failed'>;
}

export const GenAIOverviewPromptBox = ({
  chartContext,
  onClearChartContext,
  onSubmit,
}: GenAIOverviewPromptBoxProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const submissionInFlightRef = useRef(false);
  const [draft, setDraft] = useState('');
  const [placeholderIndex, setPlaceholderIndex] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submissionError, setSubmissionError] = useState<'unavailable' | 'failed'>();
  const placeholderCandidates = [
    intl.formatMessage({
      defaultMessage: 'How is my agent quality trending?',
      description: 'Overview prompt placeholder',
    }),
    intl.formatMessage({
      defaultMessage: 'Ask about recent trace failures',
      description: 'Overview prompt placeholder',
    }),
    intl.formatMessage({
      defaultMessage: 'Help me create an evaluation',
      description: 'Overview prompt placeholder',
    }),
    intl.formatMessage({
      defaultMessage: 'Which online scorer should I use?',
      description: 'Overview prompt placeholder',
    }),
  ];

  useEffect(() => {
    const interval = window.setInterval(
      () => setPlaceholderIndex((index) => (index + 1) % placeholderCandidates.length),
      3200,
    );
    return () => window.clearInterval(interval);
  }, [placeholderCandidates.length]);

  useEffect(() => {
    if (chartContext) textareaRef.current?.focus();
  }, [chartContext]);

  const submit = async () => {
    const prompt = draft.trim();
    if (!prompt || submissionInFlightRef.current) return;
    submissionInFlightRef.current = true;
    setIsSubmitting(true);
    setSubmissionError(undefined);
    try {
      const result = await onSubmit(prompt, chartContext);
      if (result === 'submitted') {
        setDraft('');
        onClearChartContext();
      } else {
        setSubmissionError(result);
      }
    } catch {
      setSubmissionError('failed');
    } finally {
      submissionInFlightRef.current = false;
      setIsSubmitting(false);
    }
  };

  const submissionErrorMessage =
    submissionError === 'unavailable'
      ? intl.formatMessage({
          defaultMessage: 'Assistant is unavailable right now. Refresh the page and try again.',
          description: 'Error shown when the Overview page cannot reach Assistant',
        })
      : intl.formatMessage({
          defaultMessage: "We couldn't send your message. Try again.",
          description: 'Error shown when the Overview Assistant prompt fails to submit',
        });
  const contextualComposerLabel = intl.formatMessage({
    defaultMessage: 'Ask Assistant about the selected chart',
    description: 'Accessible label for the contextual Assistant composer on the Overview page',
  });

  return (
    <div
      css={{
        position: 'relative',
        width: '100%',
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusXl,
        backgroundColor: theme.colors.backgroundPrimary,
        boxShadow: theme.shadows.sm,
        overflow: 'hidden',
      }}
      role={chartContext ? 'region' : undefined}
      aria-label={chartContext ? contextualComposerLabel : undefined}
      aria-busy={isSubmitting}
    >
      {chartContext && (
        <>
          <span
            key={`${chartContext.stage}-${chartContext.startTimeMs}-${chartContext.endTimeMsExclusive}-pulse`}
            aria-hidden="true"
            css={{
              position: 'absolute',
              zIndex: 1,
              inset: 0,
              opacity: 0,
              boxSizing: 'border-box',
              border: `2px solid ${theme.colors.actionDefaultBorderFocus}`,
              borderRadius: 'inherit',
              pointerEvents: 'none',
              animation: `${contextBorderPulse} ${theme.animation.transitionDuration * 4}ms ease-out`,
              '@media (prefers-reduced-motion: reduce)': {
                animation: 'none',
              },
            }}
          />
          <div
            key={`${chartContext.stage}-${chartContext.startTimeMs}-${chartContext.endTimeMsExclusive}-chip`}
            css={{
              display: 'flex',
              minWidth: 0,
              padding: `${theme.spacing.sm}px ${theme.spacing.lg}px 0`,
              animation: `${placeholderSlideIn} ${theme.animation.transitionDuration}ms ease-out`,
              '@media (prefers-reduced-motion: reduce)': {
                animation: 'none',
              },
            }}
          >
            <Tag
              componentId="mlflow.genai-overview.chart-context"
              color="default"
              icon={<BarChartIcon />}
              closable
              onClose={onClearChartContext}
              closeButtonProps={{
                'aria-label': intl.formatMessage({
                  defaultMessage: 'Remove chart context',
                  description: 'Accessible label for removing selected chart context from the Overview prompt',
                }),
              }}
              title={chartContext.label}
              css={{ minWidth: 0, maxWidth: '100%' }}
            >
              <span css={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {chartContext.label}
              </span>
            </Tag>
          </div>
        </>
      )}
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.sm,
          minHeight: theme.spacing.xl + theme.spacing.md,
          padding: `0 ${theme.spacing.lg}px`,
        }}
      >
        <div css={{ position: 'relative', minWidth: 0, flex: 1, height: theme.typography.lineHeightBase }}>
          {!draft && (
            <span
              key={placeholderIndex}
              css={{
                position: 'absolute',
                inset: 0,
                color: theme.colors.textPlaceholder,
                pointerEvents: 'none',
                animation: `${placeholderSlideIn} ${theme.animation.transitionDuration}ms ease-out`,
              }}
            >
              {placeholderCandidates[placeholderIndex]}
            </span>
          )}
          <textarea
            ref={textareaRef}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            disabled={isSubmitting}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                void submit();
              }
            }}
            aria-label={
              chartContext
                ? contextualComposerLabel
                : intl.formatMessage({
                    defaultMessage: 'Ask MLflow about this agent',
                    description: 'Overview assistant input label',
                  })
            }
            rows={1}
            css={{
              position: 'relative',
              width: '100%',
              height: theme.typography.lineHeightBase,
              boxSizing: 'border-box',
              border: 0,
              outline: 0,
              overflow: 'hidden',
              padding: 0,
              resize: 'none',
              backgroundColor: 'transparent',
              font: 'inherit',
              lineHeight: theme.typography.lineHeightBase,
            }}
          />
        </div>
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flex: `0 0 ${theme.spacing.lg}px`,
            width: theme.spacing.lg,
            height: theme.spacing.lg,
            borderRadius: '50%',
            backgroundColor: draft.trim()
              ? theme.colors.actionPrimaryBackgroundDefault
              : theme.colors.actionDisabledBackground,
          }}
        >
          <Button
            componentId="mlflow.genai-overview.submit-prompt"
            type="link"
            icon={<ArrowRightIcon />}
            disabled={!draft.trim()}
            loading={isSubmitting}
            loadingDescription="Sending prompt to Assistant"
            onClick={() => {
              void submit();
            }}
            aria-label={
              chartContext
                ? intl.formatMessage({
                    defaultMessage: 'Submit',
                    description: 'Submit button for the contextual Overview assistant prompt',
                  })
                : intl.formatMessage({
                    defaultMessage: 'Submit prompt',
                    description: 'Overview assistant submit label',
                  })
            }
            size="small"
          />
        </div>
      </div>
      {submissionError && (
        <div css={{ padding: `0 ${theme.spacing.lg}px ${theme.spacing.sm}px` }}>
          <Alert
            componentId="mlflow.genai-overview.prompt-box.submission-error"
            type="error"
            message={submissionErrorMessage}
            size="small"
            closable={false}
          />
        </div>
      )}
    </div>
  );
};
