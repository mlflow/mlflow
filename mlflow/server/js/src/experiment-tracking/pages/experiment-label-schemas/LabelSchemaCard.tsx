import { Button, Card, PencilIcon, Tag, TrashIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
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
  /**
   * When true, the card renders with a highlighted border indicating it
   * is the schema currently driving the preview pane.
   */
  selected?: boolean;
  /**
   * Fires when the card body (anywhere outside the Edit / Delete
   * buttons) is clicked. Drives preview-pane selection.
   */
  onSelect?: (schema: LabelSchema) => void;
  onEdit: (schema: LabelSchema) => void;
  onDelete: (schema: LabelSchema) => void;
}

export const LabelSchemaCard = ({ schema, selected, onSelect, onEdit, onDelete }: LabelSchemaCardProps) => {
  const { theme } = useDesignSystemTheme();
  return (
    <Card
      componentId="mlflow.experiment-label-schemas.card"
      data-testid={`label-schema-card-${schema.schema_id}`}
      onClick={onSelect ? () => onSelect(schema) : undefined}
      aria-pressed={onSelect ? !!selected : undefined}
      css={{
        padding: theme.spacing.md,
        marginBottom: theme.spacing.md,
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        cursor: onSelect ? 'pointer' : 'default',
        // Use an inset box-shadow for the selection highlight rather
        // than toggling `borderWidth` (which shifts adjacent cards by
        // 1px on selection-change). The shadow doesn't affect layout.
        boxShadow: selected ? `inset 0 0 0 2px ${theme.colors.actionPrimaryBackgroundDefault}` : undefined,
      }}
    >
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography.Title level={4} withoutMargins>
          {schema.title}
        </Typography.Title>
        <div css={{ display: 'flex', gap: theme.spacing.sm }}>
          <Button
            componentId="mlflow.experiment-label-schemas.card.edit-button"
            icon={<PencilIcon />}
            onClick={(event) => {
              // Avoid triggering the card's onSelect when the user
              // clicks the Edit affordance.
              event.stopPropagation();
              onEdit(schema);
            }}
            aria-label="Edit label schema"
          >
            <FormattedMessage defaultMessage="Edit" description="Edit label schema button" />
          </Button>
          <Button
            componentId="mlflow.experiment-label-schemas.card.delete-button"
            icon={<TrashIcon />}
            danger
            onClick={(event) => {
              event.stopPropagation();
              onDelete(schema);
            }}
            aria-label="Delete label schema"
          >
            <FormattedMessage defaultMessage="Delete" description="Delete label schema button" />
          </Button>
        </div>
      </div>
      <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
        <Tag componentId="mlflow.experiment-label-schemas.card.name-tag">{schema.name}</Tag>
        <Tag componentId="mlflow.experiment-label-schemas.card.type-tag">{schema.type}</Tag>
        <Typography.Text color="secondary">{describeInput(schema)}</Typography.Text>
      </div>
      {schema.instruction && (
        <Typography.Text color="secondary" css={{ fontStyle: 'italic' }}>
          {schema.instruction}
        </Typography.Text>
      )}
    </Card>
  );
};
