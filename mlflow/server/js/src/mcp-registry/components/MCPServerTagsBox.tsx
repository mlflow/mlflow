import { Button, PencilIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { KeyValueTag } from '../../common/components/KeyValueTag';
import type { MCPServer } from '../types';
import { tagsRecordToArray } from '../utils';
import { useUpdateMCPServerTags } from '../hooks/useUpdateMCPServerTags';

export const MCPServerTagsBox = ({ server }: { server?: MCPServer }) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const { EditTagsModal, showEditServerTagsModal } = useUpdateMCPServerTags();

  const visibleTags = server ? tagsRecordToArray(server.tags) : [];
  const containsTags = visibleTags.length > 0;

  return (
    <div
      css={{
        paddingTop: theme.spacing.xs,
        paddingBottom: theme.spacing.xs,
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        '> *': { marginRight: '0 !important' },
        gap: theme.spacing.xs,
      }}
    >
      {visibleTags.map((tag) => (
        <KeyValueTag key={tag.key} tag={tag} />
      ))}
      <Button
        componentId="mlflow.mcp_registry.detail.tags.edit"
        size="small"
        icon={!containsTags ? undefined : <PencilIcon />}
        disabled={!server}
        onClick={() => server && showEditServerTagsModal(server)}
        aria-label={intl.formatMessage({
          defaultMessage: 'Edit tags',
          description: 'Label for the edit tags button on the MCP server detail page',
        })}
        children={
          !containsTags ? (
            <FormattedMessage
              defaultMessage="Add tags"
              description="Label for the add tags button on the MCP server detail page"
            />
          ) : undefined
        }
        type="tertiary"
      />
      {EditTagsModal}
    </div>
  );
};
