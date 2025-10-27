import { FormUI, InfoSmallIcon, Input, LegacyTooltip, useDesignSystemTheme, Tag } from '@databricks/design-system';
import { usePromptEvaluationParameters } from './hooks/usePromptEvaluationParameters';
import { FormattedMessage } from 'react-intl';
import { LineSmoothSlider } from '../LineSmoothSlider';
import { isArray, uniq } from 'lodash';
import { useState } from 'react';

const EvaluationCreateParameterListControl = ({
  parameterValue,
  updateParameter,
  disabled,
}: {
  parameterValue: number | string[] | undefined;
  updateParameter: (value: number | string[]) => void;
  disabled?: boolean;
}) => {
  const [draftValue, setDraftValue] = useState<string>('');
  const { theme } = useDesignSystemTheme();

  if (!isArray(parameterValue)) {
    return null;
  }

  return (
    <>
      <div css={{ marginTop: theme.spacing.xs, marginBottom: theme.spacing.sm }}>
        {parameterValue.map((stop, index) => (
          <Tag
            componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptparameters.tsx_28"
            key={index}
            closable
            onClose={() => {
              updateParameter(parameterValue.filter((s) => s !== stop));
            }}
          >
            {stop}
          </Tag>
        ))}
      </div>
      <Input
        componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptparameters.tsx_39"
        allowClear
        css={{ width: '100%' }}
        disabled={disabled}
        onChange={(e) => setDraftValue(e.target.value)}
        value={draftValue}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && draftValue.trim()) {
            updateParameter(uniq([...parameterValue, draftValue]));
            setDraftValue('');
          }
        }}
      />
    </>
  );
};

export const EvaluationCreatePromptParameters = ({
  disabled = false,
  parameters,
  updateParameter,
}: {
  disabled?: boolean;
  parameters: {
    temperature: number;
    max_tokens: number;
    stop?: string[] | undefined;
  };
  updateParameter: (name: string, value: number | string[]) => void;
}) => {
  const { parameterDefinitions } = usePromptEvaluationParameters();
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ marginBottom: theme.spacing.lg }}>
      <FormUI.Label css={{ marginBottom: theme.spacing.md }}>
        <FormattedMessage
          defaultMessage="Model parameters"
          description="Experiment page > new run modal > served LLM model parameters label"
        />
      </FormUI.Label>
      {parameterDefinitions.map((parameterDef) => (
        <div key={parameterDef.name} css={{ marginBottom: theme.spacing.md }}>
          <>
            <FormUI.Label htmlFor={parameterDef.name} css={{ span: { fontWeight: 'normal' } }}>
              <FormattedMessage {...parameterDef.string} />
              <LegacyTooltip title={<FormattedMessage {...parameterDef.helpString} />} placement="right">
                <InfoSmallIcon
                  css={{
                    marginLeft: theme.spacing.sm,
                    verticalAlign: 'text-top',
                    color: theme.colors.textSecondary,
                  }}
                />
              </LegacyTooltip>
            </FormUI.Label>
            <FormUI.Hint />
            {parameterDef.name === 'temperature' && (
              <LineSmoothSlider
                data-testid={parameterDef.name}
                disabled={disabled}
                max={parameterDef.max}
                min={parameterDef.min}
                step={parameterDef.step}
                value={parameters[parameterDef.name] || 0}
                onChange={(value) => updateParameter(parameterDef.name, value)}
              />
            )}
            {parameterDef.type === 'input' && (
              <Input
                componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptparameters.tsx_107"
                data-testid={parameterDef.name}
                type="number"
                disabled={disabled}
                max={parameterDef.max}
                min={parameterDef.min}
                step={parameterDef.step}
                value={parameters[parameterDef.name] || 0}
                onChange={(e) => updateParameter(parameterDef.name, parseInt(e.target.value, 10))}
              />
            )}
            {parameterDef.type === 'list' && (
              <EvaluationCreateParameterListControl
                parameterValue={parameters[parameterDef.name] ?? []}
                disabled={disabled}
                updateParameter={(value) => updateParameter(parameterDef.name, value)}
              />
            )}
          </>
        </div>
      ))}
    </div>
  );
};
