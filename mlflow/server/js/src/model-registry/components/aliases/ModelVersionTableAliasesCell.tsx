import { Button, PencilIcon } from '@databricks/design-system';
import { ModelVersionAliasTag } from './ModelVersionAliasTag';
import { FormattedMessage } from 'react-intl';

interface ModelVersionTableAliasesCellProps {
  aliases?: string[];
  modelName: string;
  version: string;
  onAddEdit: () => void;
}

export const ModelVersionTableAliasesCell = ({
  aliases = [],
  onAddEdit,
}: ModelVersionTableAliasesCellProps) => {
  return (
    <div css={{ maxWidth: 300 }}>
      {aliases.length < 1 ? (
        <Button size='small' type='link' onClick={onAddEdit}>
          <FormattedMessage
            defaultMessage='Add'
            description="Model registry > model version table > aliases column > 'add' button label"
          />
        </Button>
      ) : (
        <>
          {aliases.map((alias) => (
            <ModelVersionAliasTag value={alias} key={alias} />
          ))}
          <Button size='small' icon={<PencilIcon />} onClick={onAddEdit} />
        </>
      )}
    </div>
  );
};
