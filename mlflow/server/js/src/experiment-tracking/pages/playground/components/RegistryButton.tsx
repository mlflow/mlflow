import { BookIcon, Button, Tooltip } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

interface Props {
  onOpen: () => void;
  /** Shows only the icon, with the label as a tooltip, when the top bar is narrow. */
  compact?: boolean;
}

export const RegistryButton = ({ onOpen, compact = false }: Props) => {
  const intl = useIntl();
  const label = intl.formatMessage({
    defaultMessage: 'Load prompt',
    description: 'Aria label for the playground top-bar button that opens the registered prompt picker',
  });
  const button = (
    <Button componentId="mlflow.playground.load_from_registry" icon={<BookIcon />} onClick={onOpen} aria-label={label}>
      {compact ? null : (
        <FormattedMessage
          defaultMessage="Load prompt"
          description="Label for the playground top-bar button that opens the registered prompt picker"
        />
      )}
    </Button>
  );
  return compact ? (
    <Tooltip componentId="mlflow.playground.load_from_registry.tooltip" content={label}>
      {button}
    </Tooltip>
  ) : (
    button
  );
};
