import type { TagProps } from '@databricks/design-system';
import { Button, Drawer, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import type { ServerJSONArgument } from '../types';
import { sanitizeHref } from '../utils';
import {
  flexColumnGapStyles,
  flexRowStyles,
  borderedListContainerStyles,
  borderedListItemStyles,
  blockLabelStyles,
  monoFontStyles,
} from '../styles';

export const ViewDetailsDrawer = ({ title, children }: { title: string; children: React.ReactNode }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <Drawer.Root>
      <Drawer.Trigger>
        <Button componentId="mlflow.mcp_registry.detail.view_details_drawer">
          <FormattedMessage defaultMessage="View details" description="Button to open details drawer" />
        </Button>
      </Drawer.Trigger>
      <Drawer.Content componentId="mlflow.mcp_registry.detail.details_drawer" width={480} title={title}>
        <div css={flexColumnGapStyles(theme, theme.spacing.md)}>{children}</div>
      </Drawer.Content>
    </Drawer.Root>
  );
};

export const DetailField = ({
  label,
  value,
  mono,
  tagColor,
  componentId,
  link,
}: {
  label: string;
  value: string;
  mono?: boolean;
  tagColor?: TagProps['color'];
  componentId?: string;
  link?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();

  let content: React.ReactNode;
  if (tagColor) {
    content = (
      <Tag componentId={componentId ?? 'mlflow.mcp_registry.detail.field_tag'} color={tagColor}>
        {value}
      </Tag>
    );
  } else if (link) {
    content = (
      <Typography.Link
        componentId={componentId ?? 'mlflow.mcp_registry.detail.field_link'}
        href={sanitizeHref(value)}
        target="_blank"
        rel="noopener noreferrer"
        css={{ fontSize: theme.typography.fontSizeSm, ...(mono ? monoFontStyles : {}) }}
      >
        {value}
      </Typography.Link>
    );
  } else {
    content = (
      <Typography.Text size="sm" css={mono ? monoFontStyles : undefined}>
        {value}
      </Typography.Text>
    );
  }

  return (
    <div css={flexRowStyles(theme)}>
      <Typography.Text bold size="sm">
        {label}:
      </Typography.Text>
      {content}
    </div>
  );
};

export const ArgumentList = ({ label, args }: { label: string; args: ServerJSONArgument[] }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div>
      <Typography.Text bold size="sm" css={blockLabelStyles(theme)}>
        {label} ({args.length})
      </Typography.Text>
      <div css={borderedListContainerStyles(theme)}>
        {args.map((arg, i) => {
          const name = 'name' in arg ? (arg as { name: string }).name : (arg.valueHint ?? `arg ${i + 1}`);
          return (
            <div key={i} css={borderedListItemStyles(theme, i > 0)}>
              <Typography.Text bold size="sm" css={monoFontStyles}>
                {name}
              </Typography.Text>
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
