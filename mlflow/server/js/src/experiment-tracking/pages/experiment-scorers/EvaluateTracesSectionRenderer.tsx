import React from 'react';
import {
  useDesignSystemTheme,
  Typography,
  FormUI,
  Slider,
  Input,
  Checkbox,
  Accordion,
  Tooltip,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { Controller, type Control, useWatch } from 'react-hook-form';
import { COMPONENT_ID_PREFIX, type ScorerFormMode, SCORER_FORM_MODE } from './constants';

interface EvaluateTracesSectionRendererProps {
  control: Control<any>;
  mode: ScorerFormMode;
  // TODO(ML-60517): Support automatic evaluation with conversation variable (session-level scorers)
  isSessionLevelScorer?: boolean;
}

const EvaluateTracesSectionRenderer: React.FC<EvaluateTracesSectionRendererProps> = ({
  control,
  mode,
  isSessionLevelScorer,
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  // Watch the sample rate to determine if automatic evaluation is enabled
  const sampleRate = useWatch({
    control,
    name: 'sampleRate',
  });
  const disableMonitoring = useWatch({
    control,
    name: 'disableMonitoring',
  });

  const isAutomaticEvaluationEnabled = sampleRate > 0;

  const stopPropagationClick = (e: React.MouseEvent) => {
    e.stopPropagation();
  };

  const sectionStyles = {
    display: 'flex' as const,
    flexDirection: 'column' as const,
  };

  if (disableMonitoring) {
    return null;
  }

  return (
    <>
      {/* Evaluation settings section header */}
      {(mode === SCORER_FORM_MODE.EDIT || mode === SCORER_FORM_MODE.CREATE) && (
        <div css={sectionStyles}>
          <FormUI.Label>
            <FormattedMessage
              defaultMessage="Evaluation settings"
              description="Section header for evaluation settings"
            />
          </FormUI.Label>
        </div>
      )}
      {/* Automatic evaluation checkbox */}
      <div>
        <Controller
          name="sampleRate"
          control={control}
          render={({ field }) => {
            const isCheckboxDisabled = mode === SCORER_FORM_MODE.DISPLAY || isSessionLevelScorer;
            const checkbox = (
              <Checkbox
                componentId={`${COMPONENT_ID_PREFIX}.automatic-evaluation-checkbox`}
                isChecked={isAutomaticEvaluationEnabled}
                onChange={(checked) => {
                  // If unchecked, set sample rate to 0; if checked and currently 0, set to 100
                  field.onChange(checked ? 100 : 0);
                }}
                disabled={isCheckboxDisabled}
                onClick={stopPropagationClick}
              >
                <FormattedMessage
                  defaultMessage="Automatically evaluate future traces using this judge"
                  description="Checkbox label for enabling automatic evaluation"
                />
              </Checkbox>
            );
            // Wrap in tooltip if disabled due to session-level scorer
            if (isSessionLevelScorer) {
              return (
                <Tooltip
                  content={intl.formatMessage({
                    defaultMessage: 'Automatic evaluation is not supported for session-level scorers',
                    description: 'Tooltip message when auto-evaluation is disabled for session-level scorer',
                  })}
                >
                  <span css={{ display: 'inline-block' }}>{checkbox}</span>
                </Tooltip>
              );
            }
            return checkbox;
          }}
        />
      </div>
      {/* Sample Rate and Filter String - stacked vertically (hidden when automatic evaluation is disabled) */}
      {isAutomaticEvaluationEnabled && (
        <Accordion componentId={`${COMPONENT_ID_PREFIX}.advanced-accordion`} chevronAlignment="left">
          <Accordion.Panel
            key="advanced"
            header={<FormattedMessage defaultMessage="Advanced" description="Advanced settings accordion header" />}
          >
            {/* Sample Rate Slider */}
            <div css={sectionStyles}>
              <FormUI.Label htmlFor="mlflow-experiment-scorers-sample-rate">
                <FormattedMessage defaultMessage="Sample rate" description="Section header for sample rate" />
              </FormUI.Label>
              <FormUI.Hint>
                <FormattedMessage
                  defaultMessage="Percentage of traces evaluated by this judge."
                  description="Hint text for sample rate slider"
                />
              </FormUI.Hint>
              <Controller
                name="sampleRate"
                control={control}
                render={({ field }) => (
                  <div onClick={stopPropagationClick}>
                    <Slider.Root
                      value={[field.value || 0]}
                      onValueChange={(value) => field.onChange(value[0])}
                      min={0}
                      max={100}
                      step={1}
                      disabled={mode === SCORER_FORM_MODE.DISPLAY}
                      id="mlflow-experiment-scorers-sample-rate"
                      aria-label={intl.formatMessage({
                        defaultMessage: 'Sample rate',
                        description: 'Aria label for sample rate slider',
                      })}
                      css={{
                        width: '100% !important',
                        maxWidth: '400px !important',
                        '&[data-orientation="horizontal"]': {
                          width: '100% !important',
                          maxWidth: '400px !important',
                        },
                      }}
                    >
                      <Slider.Track>
                        <Slider.Range />
                      </Slider.Track>
                      <Slider.Thumb />
                    </Slider.Root>
                    <div
                      css={{
                        marginTop: theme.spacing.xs,
                        fontSize: theme.typography.fontSizeSm,
                        color: theme.colors.textSecondary,
                        textAlign: 'left',
                      }}
                    >
                      {field.value || 0}%
                    </div>
                  </div>
                )}
              />
            </div>

            {/* Filter String Input */}
            <div css={sectionStyles}>
              <FormUI.Label htmlFor="mlflow-experiment-scorers-filter-string">
                <FormattedMessage
                  defaultMessage="Filter string (optional)"
                  description="Section header for filter string"
                />
              </FormUI.Label>
              <FormUI.Hint>
                <FormattedMessage
                  defaultMessage="Only run on traces matching this filter; leave blank to run on all. Uses MLflow {link}."
                  description="Hint text for filter string input"
                  values={{
                    link: (
                      <Typography.Link
                        componentId={`${COMPONENT_ID_PREFIX}.search-traces-syntax-link`}
                        href="https://mlflow.org/docs/latest/genai/tracing/search-traces/"
                        openInNewTab
                      >
                        {intl.formatMessage({
                          defaultMessage: 'search_traces syntax',
                          description: 'Link text for search traces documentation',
                        })}
                      </Typography.Link>
                    ),
                  }}
                />
              </FormUI.Hint>
              <Controller
                name="filterString"
                control={control}
                render={({ field }) => (
                  <Input
                    {...field}
                    componentId={`${COMPONENT_ID_PREFIX}.filter-string-input`}
                    id="mlflow-experiment-scorers-filter-string"
                    readOnly={mode === SCORER_FORM_MODE.DISPLAY}
                    placeholder={
                      mode === SCORER_FORM_MODE.EDIT || mode === SCORER_FORM_MODE.CREATE
                        ? intl.formatMessage({
                            defaultMessage: "trace.status = 'OK'",
                            description: 'Placeholder example for filter string input',
                          })
                        : undefined
                    }
                    css={{
                      cursor: mode === SCORER_FORM_MODE.EDIT || mode === SCORER_FORM_MODE.CREATE ? 'text' : 'auto',
                      width: '100%',
                      maxWidth: '400px',
                    }}
                    onClick={stopPropagationClick}
                  />
                )}
              />
            </div>
          </Accordion.Panel>
        </Accordion>
      )}
    </>
  );
};

export default EvaluateTracesSectionRenderer;
