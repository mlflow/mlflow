import { SimpleSelect, SimpleSelectOption, FormUI, Spinner, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useExperimentsForSelect } from '../../hooks/useExperimentsForSelect';

interface ExperimentSelectProps {
  value: string;
  onChange: (experimentId: string) => void;
  disabled?: boolean;
  error?: string;
  componentIdPrefix?: string;
}

export const ExperimentSelect = ({
  value,
  onChange,
  disabled,
  error,
  componentIdPrefix = 'mlflow.gateway.experiment-select',
}: ExperimentSelectProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { experiments, isLoading, error: queryError } = useExperimentsForSelect();

  if (queryError) {
    return (
      <div>
        <FormUI.Message type="error" message={queryError.message || 'Failed to load experiments'} />
      </div>
    );
  }

  if (isLoading) {
    return (
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Spinner size="small" />
        <FormattedMessage defaultMessage="Loading experiments..." description="Loading message for experiments" />
      </div>
    );
  }

  return (
    <SimpleSelect
      id={componentIdPrefix}
      componentId={componentIdPrefix}
      value={value}
      onChange={({ target }) => onChange(target.value)}
      disabled={disabled}
      placeholder={intl.formatMessage({
        defaultMessage: 'Auto-create experiment',
        description: 'Placeholder for experiment selector when no experiment is selected',
      })}
      validationState={error ? 'error' : undefined}
      contentProps={{
        matchTriggerWidth: true,
        maxHeight: 300,
      }}
      css={{ width: '100%' }}
      allowClear
    >
      {experiments.map((experiment) => (
        <SimpleSelectOption key={experiment.experimentId} value={experiment.experimentId}>
          {experiment.name}
        </SimpleSelectOption>
      ))}
    </SimpleSelect>
  );
};
