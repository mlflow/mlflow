import { Button, SaveIcon, Tooltip } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

interface Props {
  onOpen: () => void;
  disabled?: boolean;
  /** Shows only the icon, with the label as a tooltip, when the top bar is narrow. */
  compact?: boolean;
}

export const SaveToRegistryButton = ({ onOpen, disabled, compact = false }: Props) => {
  const intl = useIntl();
  const label = intl.formatMessage({
    defaultMessage: 'Save prompt',
    description: 'Aria label for the playground top-bar button that opens the save-prompt-version modal',
  });
  const button = (
    <Button
      componentId="mlflow.playground.save_to_registry"
      icon={<SaveIcon />}
      onClick={onOpen}
      disabled={disabled}
      aria-label={label}
    >
      {compact ? null : (
        <FormattedMessage
          defaultMessage="Save prompt"
          description="Label for the playground top-bar button that opens the save-prompt-version modal"
        />
      )}
    </Button>
  );
  return compact ? (
    <Tooltip componentId="mlflow.playground.save_to_registry.tooltip" content={label}>
      {button}
    </Tooltip>
  ) : (
    button
  );
};
