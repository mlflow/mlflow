import { Button, SaveIcon } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

interface Props {
  onOpen: () => void;
  disabled?: boolean;
}

export const SaveToRegistryButton = ({ onOpen, disabled }: Props) => {
  const intl = useIntl();
  return (
    <Button
      componentId="mlflow.playground.save_to_registry"
      icon={<SaveIcon />}
      onClick={onOpen}
      disabled={disabled}
      aria-label={intl.formatMessage({
        defaultMessage: 'Save prompt to registry',
        description: 'Aria label for the playground top-bar button that opens the save-prompt-version drawer',
      })}
    >
      <FormattedMessage
        defaultMessage="Save prompt to registry"
        description="Label for the playground top-bar button that opens the save-prompt-version drawer"
      />
    </Button>
  );
};
