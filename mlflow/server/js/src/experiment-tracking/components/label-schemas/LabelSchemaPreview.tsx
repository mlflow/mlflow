import { useEffect, useState } from 'react';
import { Empty, Input, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { LabelSchemaInputRenderer } from './widgets/LabelSchemaInputRenderer';
import type { LabelSchemaValue } from './widgets/LabelSchemaInputRenderer';
import {
  PASS_FAIL_NEGATIVE_DEFAULT,
  PASS_FAIL_POSITIVE_DEFAULT,
  buildLabelSchemaInputFromForm,
  validateLabelSchemaForm,
  type LabelSchemaFormData,
} from './labelSchemaFormUtils';

export interface LabelSchemaPreviewProps {
  /** Live form data driving the preview, watched from the create/edit modal. */
  formData: LabelSchemaFormData;
}

/**
 * Interactive sandbox renderer of how a reviewer will see the schema in the
 * review UI. Rendered inside the create / edit modal driven by the
 * form's live `useWatch` values; renders a panel-style header bar +
 * scrollable body matching the judges flow's
 * `SampleScorerOutputPanelRenderer`. The preview widget keeps a local
 * value (and a local comment) so the author can click the dropdown,
 * type in the comment, etc., without leaking state into the form being
 * authored.
 */
export const LabelSchemaPreview = ({ formData }: LabelSchemaPreviewProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  // Sandbox state — never read by the form, never submitted. Reset when
  // the input variant changes so a value from a different shape (e.g.,
  // an array from multi-select categorical) doesn't leak into the
  // pass/fail or numeric widget on the next render.
  const [previewValue, setPreviewValue] = useState<LabelSchemaValue>(null);
  useEffect(() => {
    setPreviewValue(null);
  }, [formData.inputKind]);

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
          defaultMessage="How a reviewer sees the schema"
          description="Label schema preview panel header subtitle"
        />
      </Typography.Hint>
    </div>
  );

  // Substitute placeholder defaults for blank pass/fail labels so the
  // preview button row never shows two empty pills when the author
  // hasn't typed anything yet. Categorical and numeric fall back to
  // their own empty-state surfaces (chips list / "No minimum"
  // placeholder), so only pass/fail needs this fill.
  const previewFormData: LabelSchemaFormData = {
    ...formData,
    passFailPositiveLabel: formData.passFailPositiveLabel || PASS_FAIL_POSITIVE_DEFAULT,
    passFailNegativeLabel: formData.passFailNegativeLabel || PASS_FAIL_NEGATIVE_DEFAULT,
  };

  // Build the input variant from the live form. If the form is too
  // incomplete to build a variant (e.g. user just opened the create
  // modal and hasn't picked an inputKind that has all required fields),
  // surface a soft empty state rather than a thrown error.
  let input;
  try {
    input = buildLabelSchemaInputFromForm(previewFormData);
  } catch {
    return (
      <div css={{ display: 'flex', flexDirection: 'column', height: '100%', width: '100%' }}>
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
    <div css={{ display: 'flex', flexDirection: 'column', height: '100%', width: '100%' }}>
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
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borders.borderRadiusMd,
            padding: theme.spacing.lg,
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.sm,
            opacity: hasErrors ? 0.6 : 1,
            backgroundColor: theme.colors.backgroundPrimary,
          }}
        >
          {formData.name ? (
            <Typography.Title level={4} withoutMargins>
              {formData.name}
            </Typography.Title>
          ) : (
            <Typography.Text color="secondary" css={{ fontStyle: 'italic' }}>
              <FormattedMessage
                defaultMessage="(no name yet)"
                description="Label schema preview placeholder for blank name"
              />
            </Typography.Text>
          )}
          {/* The text widget renders the instruction inside its box as a
              placeholder, so showing it again as a line above would be
              redundant; other variants have no text box, so keep the line. */}
          {formData.instruction && formData.inputKind !== 'text' && (
            <Typography.Text color="secondary">{formData.instruction}</Typography.Text>
          )}
          <div css={{ marginTop: theme.spacing.sm }}>
            <LabelSchemaInputRenderer
              input={input}
              value={previewValue}
              onChange={setPreviewValue}
              componentId="mlflow.experiment-label-schemas.preview"
              instruction={formData.instruction}
            />
          </div>
          {formData.enable_comment && (
            // Rationale is secondary to the structured value; surface its
            // label as the box's own placeholder rather than a separate
            // line above it, mirroring how the text widget renders the
            // instruction inside its box.
            <Input.TextArea
              componentId="mlflow.experiment-label-schemas.preview.comment"
              id="mlflow.experiment-label-schemas.preview.comment"
              rows={2}
              css={{ marginTop: theme.spacing.sm }}
              placeholder={intl.formatMessage({
                defaultMessage: 'Rationale (optional)',
                description: 'Label schema preview free-form rationale placeholder',
              })}
            />
          )}
        </div>
      </div>
    </div>
  );
};
