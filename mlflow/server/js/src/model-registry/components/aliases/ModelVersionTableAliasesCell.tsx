import { Button, PencilIcon, useDesignSystemTheme } from '@databricks/design-system';
import { AliasTag } from '../../../common/components/AliasTag';
import { FormattedMessage } from 'react-intl';

interface ModelVersionTableAliasesCellProps {
  aliases?: string[];
  modelName: string;
  version: string;
  onAddEdit: () => void;
  className?: string;
}

export const ModelVersionTableAliasesCell = ({
  aliases = [],
  onAddEdit,
  className,
}: ModelVersionTableAliasesCellProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        maxWidth: 300,
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'flex-start',
        '> *': {
          marginRight: '0 !important',
        },
        rowGap: theme.spacing.xs / 2,
        columnGap: theme.spacing.xs,
      }}
      className={className}
    >
      {aliases.length < 1 ? (
        <Button
          componentId="codegen_mlflow_app_src_model-registry_components_aliases_modelversiontablealiasescell.tsx_30"
          size="small"
          type="link"
          onClick={onAddEdit}
        >
          <FormattedMessage
            defaultMessage="Add"
            description="Model registry > model version table > aliases column > 'add' button label"
          />
        </Button>
      ) : (
        <>
          {aliases.map((alias) => (
            <AliasTag value={alias} key={alias} css={{ marginTop: theme.spacing.xs / 2 }} />
          ))}
          <Button
            componentId="codegen_mlflow_app_src_model-registry_components_aliases_modelversiontablealiasescell.tsx_41"
            size="small"
            icon={<PencilIcon />}
            onClick={onAddEdit}
          />
        </>
      )}
    </div>
  );
};
