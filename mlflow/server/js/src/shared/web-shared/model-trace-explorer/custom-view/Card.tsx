import { z } from 'zod';
import { createComponentImplementation } from '@a2ui/react/v0_9';
import { type ComponentApi, ComponentIdSchema } from '@a2ui/web_core/v0_9';
import { useDesignSystemTheme } from '@databricks/design-system';

/**
 * Schema (API) for our custom Card. Functionally identical to the basic
 * catalog's Card — a bordered container around a SINGLE child — but themed with
 * Databricks Design System tokens instead of the basic catalog's `--a2ui-*` CSS
 * variables, so it sits natively in the MLflow UI.
 */
export const CardApi = {
  name: 'Card',
  schema: z
    .object({
      child: ComponentIdSchema.describe(
        'The id of the single child component rendered inside the card. To show multiple elements, wrap them in a Row/Column and pass that container id.',
      ),
      weight: z.number().describe('Relative flex weight when placed directly inside a Row/Column.').optional(),
    })
    .strict(),
} satisfies ComponentApi;

export const Card = createComponentImplementation(CardApi, ({ props, buildChild }) => {
  const { theme } = useDesignSystemTheme();
  const weight = typeof props.weight === 'number' ? props.weight : undefined;

  // A very light tint: mostly white (primary) with a hint of the secondary
  // surface mixed in, so the card reads as a card without a heavy grey fill.
  return (
    <div
      css={{
        boxSizing: 'border-box',
        backgroundColor: `color-mix(in srgb, ${theme.colors.backgroundSecondary} 30%, ${theme.colors.backgroundPrimary})`,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        padding: theme.spacing.md,
        ...(weight !== undefined ? { flex: `${weight}`, minWidth: 0, minHeight: 0 } : {}),
      }}
    >
      {props.child ? buildChild(props.child as string) : null}
    </div>
  );
});
