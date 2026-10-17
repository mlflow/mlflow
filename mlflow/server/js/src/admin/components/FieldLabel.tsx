import type { ReactNode } from 'react';
import { InfoTooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';

/**
 * ``hint`` is deliberately all-or-nothing: an ``InfoTooltip`` needs both a
 * ``componentId`` (for the interaction registry) and an ``iconTitle`` (the icon's
 * accessible name, since the icon itself carries no text), so the union makes it
 * impossible to add hover help without them.
 */
export type FieldLabelProps = { children: ReactNode } & (
  | { hint?: undefined; componentId?: never; hintIconTitle?: never }
  | { hint: ReactNode; componentId: string; hintIconTitle: string }
);

/**
 * Bold label for a form field. Renders as a block with a small bottom
 * margin so the label sits cleanly above its input — works for both
 * block inputs (``Input``, ``SimpleSelect``, ``Radio.Group``,
 * ``DialogCombobox``) and inline-text content (e.g. the read-only
 * "Workspace: <name>" line in the role-permission form).
 *
 * With ``hint``, an info icon follows the label and reveals the text on hover or
 * keyboard focus. Use it for the explanation a first-time admin needs and a
 * returning one does not; anything that must be read before the field is filled in
 * belongs under the input, where it cannot be missed.
 */
export const FieldLabel = ({ componentId, children, hint, hintIconTitle }: FieldLabelProps) => {
  const { theme } = useDesignSystemTheme();
  return (
    <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
      {children}
      {hint ? (
        <span css={{ marginLeft: theme.spacing.xs, verticalAlign: 'middle' }}>
          <InfoTooltip componentId={componentId} iconTitle={hintIconTitle} content={hint} />
        </span>
      ) : null}
    </Typography.Text>
  );
};
