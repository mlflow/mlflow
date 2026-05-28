import {
  Button,
  Card,
  DropdownMenu,
  OverflowIcon,
  PencilIcon,
  Tag,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import type { LabelSchema } from '../../components/label-schemas/types';

const describeInput = (schema: LabelSchema): string => {
  if (schema.input.pass_fail) {
    return `Pass/Fail (${schema.input.pass_fail.positive_label} / ${schema.input.pass_fail.negative_label})`;
  }
  if (schema.input.categorical) {
    const count = schema.input.categorical.options.length;
    const mode = schema.input.categorical.multi_select ? 'multi-select' : 'single-select';
    return `Categorical, ${mode}, ${count} option${count === 1 ? '' : 's'}`;
  }
  if (schema.input.numeric) {
    const { min_value, max_value } = schema.input.numeric;
    const minStr = min_value == null ? '−∞' : String(min_value);
    const maxStr = max_value == null ? '+∞' : String(max_value);
    return `Numeric (${minStr} … ${maxStr})`;
  }
  return 'Unknown input';
};

export interface LabelSchemaCardProps {
  schema: LabelSchema;
  onEdit: (schema: LabelSchema) => void;
  onDelete: (schema: LabelSchema) => void;
}

export const LabelSchemaCard = ({ schema, onEdit, onDelete }: LabelSchemaCardProps) => {
  const { theme } = useDesignSystemTheme();
  return (
    <Card
      componentId="mlflow.experiment-label-schemas.card"
      data-testid={`label-schema-card-${schema.schema_id}`}
      css={{
        padding: theme.spacing.md,
        position: 'relative',
        width: '100%',
        boxSizing: 'border-box',
      }}
    >
      <div
        css={{
          display: 'grid',
          gridTemplateColumns: '1fr auto',
          gap: theme.spacing.xs,
          alignItems: 'flex-start',
        }}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <Typography.Title level={4} css={{ margin: 0, marginBottom: '0 !important' }}>
              {schema.title}
            </Typography.Title>
            <Tag componentId="mlflow.experiment-label-schemas.card.type-tag">{schema.type}</Tag>
          </div>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <Tag componentId="mlflow.experiment-label-schemas.card.name-tag">{schema.name}</Tag>
            <Typography.Hint>{describeInput(schema)}</Typography.Hint>
          </div>
          {schema.instruction && <Typography.Hint css={{ fontStyle: 'italic' }}>{schema.instruction}</Typography.Hint>}
        </div>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
          <Button
            componentId="mlflow.experiment-label-schemas.card.edit-button"
            size="small"
            icon={<PencilIcon />}
            onClick={() => onEdit(schema)}
            aria-label="Edit label schema"
          >
            <FormattedMessage defaultMessage="Edit" description="Edit label schema button" />
          </Button>
          <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
              <Button
                componentId="mlflow.experiment-label-schemas.card.overflow-button"
                size="small"
                icon={<OverflowIcon />}
                aria-label="More actions"
              />
            </DropdownMenu.Trigger>
            <DropdownMenu.Content align="end">
              <DropdownMenu.Item
                componentId="mlflow.experiment-label-schemas.card.delete-menu-item"
                onClick={() => onDelete(schema)}
              >
                <DropdownMenu.IconWrapper>
                  <TrashIcon />
                </DropdownMenu.IconWrapper>
                <FormattedMessage defaultMessage="Delete" description="Delete label schema menu item" />
              </DropdownMenu.Item>
            </DropdownMenu.Content>
          </DropdownMenu.Root>
        </div>
      </div>
    </Card>
  );
};
