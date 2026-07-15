import { Button, PencilIcon, Tag, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import { AliasTag } from '../../common/components/AliasTag';
import { tagListStyles } from '../styles';
import { LATEST_ALIAS } from '../utils';

interface MCPServerAliasesCellProps {
  aliases: string[];
  onEdit?: () => void;
  className?: string;
}

export const MCPServerAliasesCell = ({ aliases, onEdit, className }: MCPServerAliasesCellProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  return (
    <div css={tagListStyles(theme)} className={className}>
      {aliases.map((alias) =>
        alias === LATEST_ALIAS ? (
          <Tag key={alias} componentId="mlflow.mcp_registry.latest_alias" color="brown">
            @ {alias}
          </Tag>
        ) : (
          <AliasTag value={alias} key={alias} />
        ),
      )}
      {onEdit && (
        <Button
          componentId="mlflow.mcp_registry.edit_aliases"
          size="small"
          icon={<PencilIcon />}
          onClick={onEdit}
          aria-label={intl.formatMessage({
            defaultMessage: 'Edit aliases',
            description: 'Aria label for edit aliases button',
          })}
        />
      )}
    </div>
  );
};
