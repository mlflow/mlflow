import { Button, DatabaseIcon, Tooltip } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

interface Props {
  onOpen: () => void;
  disabled?: boolean;
  /** Shows only the icon, with the label as a tooltip, when the top bar is narrow. */
  compact?: boolean;
}

export const AddToDatasetButton = ({ onOpen, disabled, compact = false }: Props) => {
  const intl = useIntl();
  const label = intl.formatMessage({
    defaultMessage: 'Add to datasets',
    description: 'Aria label for the playground top-bar button that opens the add-to-evaluation-datasets modal',
  });
  const button = (
    <Button
      componentId="mlflow.playground.add_to_dataset"
      icon={<DatabaseIcon />}
      onClick={onOpen}
      disabled={disabled}
      aria-label={label}
    >
      {compact ? null : (
        <FormattedMessage
          defaultMessage="Add to datasets"
          description="Label for the playground top-bar button that opens the add-to-evaluation-datasets modal"
        />
      )}
    </Button>
  );
  return compact ? (
    <Tooltip componentId="mlflow.playground.add_to_dataset.tooltip" content={label}>
      {button}
    </Tooltip>
  ) : (
    button
  );
};
