import type { ReactElement, ReactNode } from 'react';
import { Button, Empty, NoIcon, PlusIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { emptyCenterStyles } from '../styles';

export const MCPRegistryEmptyState = ({
  title,
  description,
  button,
  image,
}: {
  title: ReactNode;
  description?: ReactNode;
  button?: ReactElement;
  image?: ReactElement;
}) => {
  return (
    <div css={emptyCenterStyles}>
      <Empty title={title} description={description ?? null} button={button} image={image} />
    </div>
  );
};

export const MCPServersEmptyState = ({
  isFiltered,
  componentId,
  onCreateServer,
}: {
  isFiltered?: boolean;
  componentId: string;
  onCreateServer?: () => void;
}) => {
  if (isFiltered) {
    return (
      <MCPRegistryEmptyState
        image={<NoIcon />}
        title={
          <FormattedMessage
            defaultMessage="No servers found"
            description="Empty state when MCP server search returns no results"
          />
        }
      />
    );
  }
  return (
    <MCPRegistryEmptyState
      title={<FormattedMessage defaultMessage="Create MCP server" description="Empty state title for MCP servers" />}
      description={
        <FormattedMessage
          defaultMessage="Create and manage MCP servers using MLflow."
          description="Empty state description for MCP servers"
        />
      }
      button={
        <Button
          componentId={componentId}
          type="primary"
          icon={<PlusIcon />}
          disabled={!onCreateServer}
          onClick={onCreateServer}
        >
          <FormattedMessage defaultMessage="Create MCP server" description="MCP servers empty state CTA button" />
        </Button>
      }
    />
  );
};
