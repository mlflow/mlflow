import React, { useEffect, useState } from 'react';
import {
  useDesignSystemTheme,
  Typography,
  FormUI,
  Slider,
  Input,
  Switch,
  ChevronDownIcon,
  ChevronUpIcon,
  Button,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { Controller, type Control, type UseFormSetValue, useWatch } from 'react-hook-form';
import { type ScorerFormMode, SCORER_FORM_MODE, ScorerEvaluationScope } from './constants';
import { ModelProvider } from '../../../gateway/utils/gatewayUtils';
import { hasTemplateVariable } from './utils/templateUtils';
import { templateRequiresExpectations } from './types';

interface EvaluateTracesSectionRendererProps {
  control: Control<any>;
  mode: ScorerFormMode;
  setValue?: UseFormSetValue<any>;
}

const EvaluateTracesSectionRenderer: React.FC<EvaluateTracesSectionRendererProps> = ({ control, mode, setValue }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);

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
  const instructions = useWatch({
    control,
    name: 'instructions',
  });
  const modelInputMode = useWatch({
    control,
    name: 'modelInputMode',
  });
  const llmTemplate = useWatch({
    control,
    name: 'llmTemplate',
  });

  // Check if scorer requires expectations - either via built-in template or custom instructions containing {{ expectations }}
  const hasExpectations =
    templateRequiresExpectations(llmTemplate) || hasTemplateVariable(instructions, 'expectations');

  // Check if using a non-gateway model - automatic evaluation only works with gateway models
  const isNonGatewayModel = modelInputMode === ModelProvider.OTHER;

  // Set sampleRate based on whether automatic evaluation is allowed
  useEffect(() => {
    if (!setValue) return;
    if (hasExpectations || isNonGatewayModel) {
      setValue('sampleRate', 0);
    } else {
      setValue('sampleRate', 100);
    }
  }, [hasExpectations, isNonGatewayModel, setValue]);

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
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div
        css={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
        }}
      >
        <div>
          <FormUI.Label>
            <FormattedMessage
              defaultMessage="Run on all future traces"
              description="Label for toggle to enable automatic evaluation"
            />
          </FormUI.Label>
          <FormUI.Hint>
            <FormattedMessage
              defaultMessage="Automatically evaluate new traces using this scorer"
              description="Hint text for automatic evaluation toggle"
            />
          </FormUI.Hint>
        </div>
        <Controller
          name="sampleRate"
          control={control}
          render={({ field }) => (
            <Switch
              componentId="mlflow.experiment_page.scorers.auto_evaluate_toggle"
              checked={isAutomaticEvaluationEnabled}
              onChange={(checked) => {
                // If unchecked, set sample rate to 0; if checked and currently 0, set to 100
                field.onChange(checked ? 100 : 0);
              }}
              disabled={mode === SCORER_FORM_MODE.DISPLAY || hasExpectations || isNonGatewayModel}
            />
          )}
        />
      </div>
      {hasExpectations && (
        <FormUI.Hint css={{ marginTop: theme.spacing.xs }}>
          <FormattedMessage
            defaultMessage="Automatic evaluation is not available for judges that use expectations."
            description="Hint text explaining why automatic evaluation is disabled for judges with expectations"
          />
        </FormUI.Hint>
      )}
      {isNonGatewayModel && !hasExpectations && (
        <FormUI.Hint css={{ marginTop: theme.spacing.xs }}>
          <FormattedMessage
            defaultMessage="Automatic evaluation is only available for judges that use gateway endpoints."
            description="Hint text explaining why automatic evaluation is disabled for non-gateway models"
          />
        </FormUI.Hint>
      )}

      {isAutomaticEvaluationEnabled && (
        <div>
          <Button
            componentId="mlflow.experiment_page.scorers.advanced_settings_toggle"
            type="link"
            size="small"
            onClick={() => setIsAdvancedOpen(!isAdvancedOpen)}
            icon={isAdvancedOpen ? <ChevronUpIcon /> : <ChevronDownIcon />}
          >
            <FormattedMessage
              defaultMessage="Advanced settings"
              description="Collapsible header for advanced scoring job settings"
            />
          </Button>

          {isAdvancedOpen && (
            <div
              css={{
                display: 'flex',
                flexDirection: 'column',
                gap: theme.spacing.md,
                paddingTop: theme.spacing.md,
                paddingLeft: theme.spacing.lg,
              }}
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
                  {isSessionLevelScorer ? (
                    <FormattedMessage
                      defaultMessage="Filter applies to the first trace in each session. Only run on sessions where the first trace matches this filter; leave blank to run on all. Uses MLflow {link}."
                      description="Hint text for filter string input for session-level scorers"
                      values={{
                        link: (
                          <Typography.Link
                            componentId="mlflow.experiment_page.scorers.search_traces_syntax_link"
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
                            componentId="mlflow.experiment_page.scorers.filter_string_syntax_link"
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
                      componentId="mlflow.experiment_page.scorers.filter_string_input"
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
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default EvaluateTracesSectionRenderer;
