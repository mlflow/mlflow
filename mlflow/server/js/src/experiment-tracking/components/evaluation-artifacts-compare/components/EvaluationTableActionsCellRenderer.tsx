import { Button, PlusIcon, Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

interface EvaluationTableActionsCellRendererProps {
  onAddNewInputs: () => void;
  displayAddNewInputsButton: boolean;
}

export const EvaluationTableActionsCellRenderer = ({
  onAddNewInputs,
  displayAddNewInputsButton,
}: EvaluationTableActionsCellRendererProps) => {
  const { theme } = useDesignSystemTheme();
  if (!displayAddNewInputsButton) {
    return null;
  }

  return (
    <div
      css={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        padding: theme.spacing.xs,
      }}
    >
      <Tooltip
        placement='right'
        title={
          <FormattedMessage
            defaultMessage='Add row'
            description='Experiment page > artifact compare view > add new row button'
          />
        }
      >
        <Button icon={<PlusIcon />} onClick={onAddNewInputs} />
      </Tooltip>
    </div>
  );
};
