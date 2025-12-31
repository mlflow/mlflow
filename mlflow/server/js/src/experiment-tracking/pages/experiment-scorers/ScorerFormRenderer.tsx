import React, { useState, useRef, useCallback } from 'react';
import {
  type UseFormHandleSubmit,
  type Control,
  type UseFormSetValue,
  type UseFormGetValues,
  Controller,
  useWatch,
} from 'react-hook-form';
import { useDesignSystemTheme, Button, FormUI, Alert, Radio, Tag } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { isEvaluatingSessionsInScorersEnabled, isRunningScorersEnabled } from '../../../common/utils/FeatureUtils';
import {
  ModelTraceExplorerResizablePane,
  type ModelTraceExplorerResizablePaneRef,
} from '@databricks/web-shared/model-trace-explorer';
import LLMScorerFormRenderer, { type LLMScorerFormData } from './LLMScorerFormRenderer';
import CustomCodeScorerFormRenderer, { type CustomCodeScorerFormData } from './CustomCodeScorerFormRenderer';
import SampleScorerOutputPanelContainer from './SampleScorerOutputPanelContainer';
import type { ScheduledScorer } from './types';
import { getTypeDisplayName, getTypeIcon } from './scorerCardUtils';
import type { ScorerFormData } from './utils/scorerTransformUtils';
import { COMPONENT_ID_PREFIX, SCORER_FORM_MODE, type ScorerFormMode } from './constants';
import { ScorerFormEvaluationScopeSelect } from './ScorerFormEvaluationScopeSelect';

interface ScorerFormRendererProps {
  mode: ScorerFormMode;
  handleSubmit: UseFormHandleSubmit<ScorerFormData>;
  onFormSubmit: (data: ScorerFormData) => void;
  control: Control<ScorerFormData>;
  setValue: UseFormSetValue<ScorerFormData>;
  getValues: UseFormGetValues<ScorerFormData>;
  scorerType: ScorerFormData['scorerType'];
  mutation: {
    isLoading: boolean;
    error: any;
  };
  componentError: string | null;
  handleCancel: () => void;
  isSubmitDisabled: boolean;
  experimentId: string;
}

// Extracted form content component
interface ScorerFormContentProps {
  mode: ScorerFormMode;
  control: Control<ScorerFormData>;
  setValue: UseFormSetValue<ScorerFormData>;
  getValues: UseFormGetValues<ScorerFormData>;
  scorerType: ScorerFormData['scorerType'];
  mutation: {
    isLoading: boolean;
    error: any;
  };
  componentError: string | null;
}

const ScorerFormContent: React.FC<ScorerFormContentProps> = ({
  mode,
  control,
  setValue,
  getValues,
  scorerType,
  mutation,
  componentError,
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  return (
    <>
      {isEvaluatingSessionsInScorersEnabled() && (
        <div>
          <ScorerFormEvaluationScopeSelect />
        </div>
      )}
      {/* Scorer Type Selection - only show in create mode */}
      {mode === SCORER_FORM_MODE.CREATE && (
        <div>
          <FormUI.Label>
            <FormattedMessage defaultMessage="Judge type" description="Label for judge type selection" />
          </FormUI.Label>
          <Controller
            name="scorerType"
            control={control}
            rules={{ required: true }}
            render={({ field }) => (
              <Radio.Group
                {...field}
                componentId={`${COMPONENT_ID_PREFIX}.new-scorer-type.radio-group`}
                name="scorer-type"
              >
                <Radio value="llm">
                  <Tag
                    componentId={`${COMPONENT_ID_PREFIX}.llm-type-tag`}
                    color="purple"
                    icon={getTypeIcon({ type: 'llm' } as ScheduledScorer)}
                  >
                    {getTypeDisplayName({ type: 'llm' } as ScheduledScorer, intl)}
                  </Tag>
                </Radio>
                <FormUI.Hint>
                  <FormattedMessage
                    defaultMessage="Use a large language model to automatically evaluate traces."
                    description="Hint text for LLM scorer type option"
                  />
                </FormUI.Hint>

                <Radio value="custom-code">
                  <Tag
                    componentId={`${COMPONENT_ID_PREFIX}.custom-code-type-tag`}
                    color="pink"
                    icon={getTypeIcon({ type: 'custom-code' } as ScheduledScorer)}
                  >
                    {getTypeDisplayName({ type: 'custom-code' } as ScheduledScorer, intl)}
                  </Tag>
                </Radio>
                <FormUI.Hint>
                  <FormattedMessage
                    defaultMessage="Create your own judge using a Python function. Useful if your requirements are not met by LLM-as-a-judge judges."
                    description="Hint text for custom code judge type option"
                  />
                </FormUI.Hint>
              </Radio.Group>
            )}
          />
        </div>
      )}
      {/* Conditional Form Content */}
      {scorerType === 'llm' ? (
        <LLMScorerFormRenderer
          mode={mode}
          control={control as Control<LLMScorerFormData>}
          setValue={setValue as UseFormSetValue<LLMScorerFormData>}
          getValues={getValues as UseFormGetValues<LLMScorerFormData>}
        />
      ) : (
        <CustomCodeScorerFormRenderer mode={mode} control={control as Control<CustomCodeScorerFormData>} />
      )}
      {/* Error message - display with priority: local error first, then mutation error */}
      {(mutation.error || componentError) && (
        <Alert
          componentId={`${COMPONENT_ID_PREFIX}.scorer-form-error`}
          type="error"
          message={componentError || mutation.error?.message || mutation.error?.displayMessage}
          closable={false}
          css={{ marginTop: theme.spacing.md }}
        />
      )}
    </>
  );
};

const ScorerFormRenderer: React.FC<ScorerFormRendererProps> = ({
  mode,
  handleSubmit,
  onFormSubmit,
  control,
  setValue,
  getValues,
  scorerType,
  mutation,
  componentError,
  handleCancel,
  isSubmitDisabled,
  experimentId,
}) => {
  const { theme } = useDesignSystemTheme();
  const [leftPaneWidth, setLeftPaneWidth] = useState(800);
  const resizablePaneRef = useRef<ModelTraceExplorerResizablePaneRef>(null);
  const isRunningScorersFeatureEnabled = isRunningScorersEnabled();

  // Callback to adjust panel ratio after scorer runs
  const handleScorerFinished = useCallback(() => {
    // Update to 34% left / 66% right split
    const containerWidth = resizablePaneRef.current?.getContainerWidth();
    if (containerWidth) {
      const newLeftWidth = containerWidth * 0.34;
      setLeftPaneWidth(newLeftWidth);
      resizablePaneRef.current?.updateRatio(newLeftWidth);
    }
  }, []);

  return (
    <form
      onSubmit={handleSubmit(onFormSubmit)}
      css={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        overflow: 'hidden',
      }}
    >
      {isRunningScorersFeatureEnabled && scorerType === 'llm' ? (
        // Two-column resizable layout with sample scorer output panel (only for LLM scorers)
        <div css={{ flex: 1, display: 'flex', minHeight: 0, overflow: 'hidden' }}>
          <ModelTraceExplorerResizablePane
            ref={resizablePaneRef}
            initialRatio={0.55}
            paneWidth={leftPaneWidth}
            setPaneWidth={setLeftPaneWidth}
            leftMinWidth={200}
            rightMinWidth={200}
            leftChild={
              <div
                css={{
                  display: 'flex',
                  flexDirection: 'column',
                  overflow: 'hidden',
                  minHeight: 0,
                  height: '100%',
                  width: '100%',
                }}
              >
                {/* Scrollable content area */}
                <div
                  css={{
                    flex: 1,
                    overflowY: 'auto',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: theme.spacing.sm,
                    paddingBottom: theme.spacing.md,
                    paddingRight: theme.spacing.md,
                  }}
                >
                  <ScorerFormContent
                    mode={mode}
                    control={control}
                    setValue={setValue}
                    getValues={getValues}
                    scorerType={scorerType}
                    mutation={mutation}
                    componentError={componentError}
                  />
                </div>
              </div>
            }
            rightChild={
              <div
                css={{
                  width: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  overflow: 'hidden',
                  minHeight: 0,
                  height: '100%',
                  paddingLeft: theme.spacing.md,
                  borderLeft: `1px solid ${theme.colors.border}`,
                }}
              >
                <SampleScorerOutputPanelContainer
                  control={control}
                  experimentId={experimentId}
                  onScorerFinished={handleScorerFinished}
                />
              </div>
            }
          />
        </div>
      ) : (
        // Single-column layout (no sample scorer output panel)
        <div
          css={{
            flex: 1,
            overflowY: 'auto',
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.sm,
            paddingBottom: theme.spacing.md,
          }}
        >
          <ScorerFormContent
            mode={mode}
            control={control}
            setValue={setValue}
            getValues={getValues}
            scorerType={scorerType}
            mutation={mutation}
            componentError={componentError}
          />
        </div>
      )}
      {/* Sticky footer with buttons */}
      <div
        css={{
          display: 'flex',
          justifyContent: 'flex-end',
          gap: theme.spacing.sm,
          paddingTop: theme.spacing.md,
          position: 'sticky',
          bottom: 0,
        }}
      >
        <Button componentId={`${COMPONENT_ID_PREFIX}.new-scorer-form.cancel-button`} onClick={handleCancel}>
          <FormattedMessage defaultMessage="Cancel" description="Cancel button text" />
        </Button>
        <Button
          componentId={`${COMPONENT_ID_PREFIX}.scorer-form.submit-button`}
          type="primary"
          htmlType="submit"
          loading={mutation.isLoading}
          disabled={isSubmitDisabled}
        >
          {mode === SCORER_FORM_MODE.EDIT ? (
            <FormattedMessage defaultMessage="Save" description="Save judge button text" />
          ) : (
            <FormattedMessage defaultMessage="Create judge" description="Create judge button text" />
          )}
        </Button>
      </div>
    </form>
  );
};

export default ScorerFormRenderer;
