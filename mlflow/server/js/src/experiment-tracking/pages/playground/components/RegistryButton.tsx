import { BookIcon, Button } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

interface Props {
  onOpen: () => void;
}

export const RegistryButton = ({ onOpen }: Props) => {
  const intl = useIntl();
  return (
    <Button
      componentId="mlflow.playground.load_from_registry"
      icon={<BookIcon />}
      onClick={onOpen}
      aria-label={intl.formatMessage({
        defaultMessage: 'Load prompt',
        description: 'Aria label for the playground top-bar button that opens the registered prompt picker',
      })}
    >
      <FormattedMessage
        defaultMessage="Load prompt"
        description="Label for the playground top-bar button that opens the registered prompt picker"
      />
    </Button>
  );
};
