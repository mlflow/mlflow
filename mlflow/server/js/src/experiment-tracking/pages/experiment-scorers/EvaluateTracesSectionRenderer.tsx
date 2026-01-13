import React from 'react';
import { useDesignSystemTheme, Typography, FormUI, Slider, Input, Checkbox } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { Controller, type Control, useWatch } from 'react-hook-form';
import { COMPONENT_ID_PREFIX, type ScorerFormMode, SCORER_FORM_MODE, ScorerEvaluationScope } from './constants';

interface EvaluateTracesSectionRendererProps {
  control: Control<any>;
  mode: ScorerFormMode;
}

const EvaluateTracesSectionRenderer: React.FC<EvaluateTracesSectionRendererProps> = ({ control, mode }) => {
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
  const evaluationScope = useWatch({
    control,
    name: 'evaluationScope',
  });

  const isAutomaticEvaluationEnabled = sampleRate > 0;
  const isSessionLevelScorer = evaluationScope === ScorerEvaluationScope.SESSIONS;

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
          render={({ field }) => (
            <Checkbox
              componentId="codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_evaluatetracessectionrenderer_61"
              isChecked={isAutomaticEvaluationEnabled}
              onChange={(checked) => {
                // If unchecked, set sample rate to 0; if checked and currently 0, set to 100
                field.onChange(checked ? 100 : 0);
              }}
              disabled={mode === SCORER_FORM_MODE.DISPLAY}
              onClick={stopPropagationClick}
            >
              <FormattedMessage
                defaultMessage="Automatically evaluate future traces using this judge"
                description="Checkbox label for enabling automatic evaluation"
              />
            </Checkbox>
          )}
        />
      </div>
      {/* Sample Rate and Filter String - stacked vertically (hidden when automatic evaluation is disabled) */}
      {isAutomaticEvaluationEnabled && (
        <>
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
              {isSessionLevelScorer ? (
                <FormattedMessage
                  defaultMessage="Filter applies to the first trace in each session. Only run on sessions where the first trace matches this filter; leave blank to run on all. Uses MLflow {link}."
                  description="Hint text for filter string input for session-level scorers"
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
              ) : (
                <FormattedMessage
                  defaultMessage="Only run on traces matching this filter; leave blank to run on all. Uses MLflow {link}."
                  description="Hint text for filter string input"
                  values={{
                    link: (
                      <Typography.Link
                        componentId="codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_evaluatetracessectionrenderer_159"
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
              )}
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
        </>
      )}
    </>
  );
};

export default EvaluateTracesSectionRenderer;
