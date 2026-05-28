import { Empty, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import { LabelSchemaInputRenderer } from '../../components/label-schemas/widgets/LabelSchemaInputRenderer';
import {
  buildLabelSchemaInputFromForm,
  validateLabelSchemaForm,
  type LabelSchemaFormData,
} from './labelSchemaFormUtils';

export interface LabelSchemaPreviewProps {
  /**
   * Form data driving the preview. `null` renders an empty state, e.g.
   * when there are no schemas in the experiment and the user hasn't
   * opened the create modal yet.
   */
  formData: LabelSchemaFormData | null;
}

/**
 * Non-interactive renderer of how an SME will see the schema in the
 * review UI. Single data source: takes a `LabelSchemaFormData` (either
 * derived from a saved schema via `getFormValuesFromSchema` or piped
 * live from the create / edit modal) and renders the title, instruction,
 * and the input widget. The widget surface is rendered behind a
 * `pointer-events: none` overlay so the preview never accepts input.
 */
export const LabelSchemaPreview = ({ formData }: LabelSchemaPreviewProps) => {
  const { theme } = useDesignSystemTheme();

  if (!formData) {
    return (
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          minHeight: 400,
          width: '100%',
          padding: theme.spacing.lg,
          // Override the Design System's internal wrapper styles to
          // properly center the Empty content (per mlflow/server/js
          // CLAUDE.md empty-states guidance).
          '& > div': {
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
          },
        }}
      >
        <Empty
          title={
            <FormattedMessage
              defaultMessage="No schema selected"
              description="Label schema preview empty state title"
            />
          }
          description={
            <FormattedMessage
              defaultMessage="Select a schema from the list or create a new one to preview what an SME will see when annotating a trace."
              description="Label schema preview empty state description"
            />
          }
        />
      </div>
    );
  }

  // Build the input variant from the live form. If the form is too
  // incomplete to build a variant (e.g. user just opened the create
  // modal and hasn't picked an inputKind that has all required fields),
  // surface a soft empty state rather than a thrown error.
  let input;
  try {
    input = buildLabelSchemaInputFromForm(formData);
  } catch {
    return (
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          minHeight: 400,
          width: '100%',
          padding: theme.spacing.lg,
          // Override the Design System's internal wrapper styles to
          // properly center the Empty content (per mlflow/server/js
          // CLAUDE.md empty-states guidance).
          '& > div': {
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
          },
        }}
      >
        <Empty
          title={
            <FormattedMessage
              defaultMessage="Preview unavailable"
              description="Label schema preview render-error empty title"
            />
          }
          description={
            <FormattedMessage
              defaultMessage="Continue editing the schema to see a preview."
              description="Label schema preview render-error empty description"
            />
          }
        />
      </div>
    );
  }

  const validationErrors = validateLabelSchemaForm(formData);
  const hasErrors = Object.keys(validationErrors).length > 0;

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
        padding: theme.spacing.lg,
        height: '100%',
        overflowY: 'auto',
      }}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
        <Typography.Title level={4} color="secondary" withoutMargins>
          <FormattedMessage defaultMessage="Preview" description="Label schema preview pane heading" />
        </Typography.Title>
        <Typography.Text size="sm" color="secondary">
          <FormattedMessage
            defaultMessage="This is how an SME will see the schema when annotating a trace."
            description="Label schema preview pane helper text"
          />
        </Typography.Text>
      </div>
      <div
        css={{
          border: `1px dashed ${theme.colors.border}`,
          borderRadius: theme.borders.borderRadiusMd,
          padding: theme.spacing.lg,
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.sm,
          opacity: hasErrors ? 0.6 : 1,
          pointerEvents: 'none',
          backgroundColor: theme.colors.backgroundSecondary,
        }}
      >
        {formData.title ? (
          <Typography.Title level={4} withoutMargins>
            {formData.title}
          </Typography.Title>
        ) : (
          <Typography.Text color="secondary" css={{ fontStyle: 'italic' }}>
            <FormattedMessage
              defaultMessage="(no title yet)"
              description="Label schema preview placeholder for blank title"
            />
          </Typography.Text>
        )}
        {formData.instruction && <Typography.Text color="secondary">{formData.instruction}</Typography.Text>}
        <div css={{ marginTop: theme.spacing.sm }}>
          <LabelSchemaInputRenderer
            input={input}
            value={null}
            onChange={() => {
              // The preview is non-interactive; the wrapping
              // `pointer-events: none` should already prevent the user
              // from triggering onChange, but the widget contract
              // requires a handler so we accept and discard.
            }}
            disabled
            componentId="mlflow.experiment-label-schemas.preview"
          />
        </div>
      </div>
    </div>
  );
};
