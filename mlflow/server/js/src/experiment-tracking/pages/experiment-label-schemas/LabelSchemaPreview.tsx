import { Empty, FormUI, Input, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import { LabelSchemaInputRenderer } from '../../components/label-schemas/widgets/LabelSchemaInputRenderer';
import {
  buildLabelSchemaInputFromForm,
  validateLabelSchemaForm,
  type LabelSchemaFormData,
} from './labelSchemaFormUtils';

export interface LabelSchemaPreviewProps {
  /** Live form data driving the preview, watched from the create/edit modal. */
  formData: LabelSchemaFormData;
}

/**
 * Non-interactive renderer of how an SME will see the schema in the
 * review UI. Rendered inside the create / edit modal driven by the
 * form's live `useWatch` values; renders a panel-style header bar +
 * scrollable body so the layout matches the judges flow's
 * `SampleScorerOutputPanelRenderer`.
 */
export const LabelSchemaPreview = ({ formData }: LabelSchemaPreviewProps) => {
  const { theme } = useDesignSystemTheme();

  const header = (
    <div
      css={{
        padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
        backgroundColor: theme.colors.backgroundSecondary,
        borderBottom: `1px solid ${theme.colors.border}`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}
    >
      <Typography.Text bold>
        <FormattedMessage defaultMessage="Preview" description="Label schema preview panel header" />
      </Typography.Text>
      <Typography.Hint>
        <FormattedMessage
          defaultMessage="How an SME sees the schema"
          description="Label schema preview panel header subtitle"
        />
      </Typography.Hint>
    </div>
  );

  // Build the input variant from the live form. If the form is too
  // incomplete to build a variant (e.g. user just opened the create
  // modal and hasn't picked an inputKind that has all required fields),
  // surface a soft empty state rather than a thrown error.
  let input;
  try {
    input = buildLabelSchemaInputFromForm(formData);
  } catch {
    return (
      <div css={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        {header}
        <div
          css={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
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
      </div>
    );
  }

  const validationErrors = validateLabelSchemaForm(formData);
  const hasErrors = Object.keys(validationErrors).length > 0;

  return (
    <div css={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {header}
      <div
        css={{
          flex: 1,
          overflowY: 'auto',
          padding: theme.spacing.md,
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.md,
        }}
      >
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
          {input.categorical && input.categorical.options.length > 0 && (
            <div
              css={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: theme.spacing.xs,
                marginTop: theme.spacing.xs,
              }}
            >
              {input.categorical.options.map((option) => (
                <Tag key={option} componentId="mlflow.experiment-label-schemas.preview.categorical-option-tag">
                  {option}
                </Tag>
              ))}
            </div>
          )}
          {formData.enable_comment && (
            <div css={{ display: 'flex', flexDirection: 'column', marginTop: theme.spacing.sm }}>
              <FormUI.Label htmlFor="mlflow.experiment-label-schemas.preview.comment">
                <FormattedMessage
                  defaultMessage="Comment (optional)"
                  description="Label schema preview free-form comment label"
                />
              </FormUI.Label>
              <Input.TextArea
                componentId="mlflow.experiment-label-schemas.preview.comment"
                id="mlflow.experiment-label-schemas.preview.comment"
                disabled
                rows={2}
                placeholder=""
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
