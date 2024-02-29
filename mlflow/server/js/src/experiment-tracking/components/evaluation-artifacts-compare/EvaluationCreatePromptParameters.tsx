import { FormUI, InfoIcon, Input, Tooltip, LegacySelect, useDesignSystemTheme } from '@databricks/design-system';
import { usePromptEvaluationParameters } from './hooks/usePromptEvaluationParameters';
import { FormattedMessage } from 'react-intl';
import { LineSmoothSlider } from '../LineSmoothSlider';

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
              <Tooltip title={<FormattedMessage {...parameterDef.helpString} />} placement="right">
                <InfoIcon
                  css={{
                    marginLeft: theme.spacing.sm,
                    verticalAlign: 'text-top',
                    color: theme.colors.textSecondary,
                  }}
                />
              </Tooltip>
            </FormUI.Label>
            <FormUI.Hint></FormUI.Hint>
            {parameterDef.name === 'temperature' && (
              <LineSmoothSlider
                data-testid={parameterDef.name}
                disabled={disabled}
                max={parameterDef.max}
                min={parameterDef.min}
                step={parameterDef.step}
                defaultValue={parameters[parameterDef.name] || 0}
                handleLineSmoothChange={(value) => updateParameter(parameterDef.name, value)}
              />
            )}
            {parameterDef.type === 'input' && (
              <Input
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
              <LegacySelect
                allowClear
                mode="tags"
                style={{ width: '100%' }}
                open={false}
                disabled={disabled}
                onChange={(e) => updateParameter(parameterDef.name, e)}
                value={parameters[parameterDef.name] || []}
                dangerouslySetAntdProps={{ suffixIcon: null }}
              />
            )}
          </>
        </div>
      ))}
    </div>
  );
};
