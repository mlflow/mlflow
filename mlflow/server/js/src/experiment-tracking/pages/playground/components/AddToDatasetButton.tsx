import { Button, DatabaseIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

interface Props {
  onOpen: () => void;
  disabled?: boolean;
}

export const AddToDatasetButton = ({ onOpen, disabled }: Props) => {
  return (
    <Button componentId="mlflow.playground.add_to_dataset" icon={<DatabaseIcon />} onClick={onOpen} disabled={disabled}>
      <FormattedMessage
        defaultMessage="Add to datasets"
        description="Label for the playground top-bar button that opens the add-to-evaluation-datasets modal"
      />
    </Button>
  );
};
