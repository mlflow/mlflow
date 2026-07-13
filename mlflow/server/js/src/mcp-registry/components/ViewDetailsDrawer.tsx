import type { TagProps } from '@databricks/design-system';
import {
  Button,
  Drawer,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import type { ServerJSONArgument } from '../types';

export const ViewDetailsDrawer = ({ title, children }: { title: string; children: React.ReactNode }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <Drawer.Root>
      <Drawer.Trigger>
        <Button componentId="mlflow.mcp_registry.detail.view_details_drawer">
          <FormattedMessage defaultMessage="View details" description="Button to open details drawer" />
        </Button>
      </Drawer.Trigger>
      <Drawer.Content
        componentId="mlflow.mcp_registry.detail.details_drawer"
        width={480}
        title={title}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {children}
        </div>
      </Drawer.Content>
    </Drawer.Root>
  );
};

export const DetailField = ({ label, value, mono, tagColor, componentId, link }: { label: string; value: string; mono?: boolean; tagColor?: TagProps['color']; componentId?: string; link?: boolean }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
      <Typography.Text bold size="sm">{label}:</Typography.Text>
      {tagColor ? (
        <Tag componentId={componentId ?? 'mlflow.mcp_registry.detail.field_tag'} color={tagColor}>{value}</Tag>
      ) : link ? (
        <Typography.Link componentId={componentId ?? 'mlflow.mcp_registry.detail.field_link'} href={value} target="_blank" rel="noopener noreferrer" css={{ fontSize: theme.typography.fontSizeSm, ...(mono ? { fontFamily: 'monospace' } : {}) }}>{value}</Typography.Link>
      ) : (
        <Typography.Text size="sm" css={mono ? { fontFamily: 'monospace' } : undefined}>{value}</Typography.Text>
      )}
    </div>
  );
};

export const ArgumentList = ({ label, args }: { label: string; args: ServerJSONArgument[] }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div>
      <Typography.Text bold size="sm" css={{ display: 'block', marginBottom: theme.spacing.xs }}>
        {label} ({args.length})
      </Typography.Text>
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          border: `1px solid ${theme.colors.border}`,
          borderRadius: theme.borders.borderRadiusSm,
          overflow: 'hidden',
        }}
      >
        {args.map((arg, i) => {
          const name = 'name' in arg ? (arg as { name: string }).name : arg.valueHint ?? `arg ${i + 1}`;
          return (
            <div
              key={name}
              css={{
                padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                borderTop: i > 0 ? `1px solid ${theme.colors.border}` : 'none',
              }}
            >
              <Typography.Text bold size="sm" css={{ fontFamily: 'monospace' }}>{name}</Typography.Text>
              {arg.description && (
                <Typography.Text color="secondary" size="sm" css={{ display: 'block' }}>
                  {arg.description}
                </Typography.Text>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
