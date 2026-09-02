import { z } from 'zod';
import { createComponentImplementation, type ReactComponentImplementation } from '@a2ui/react/v0_9';
import { type ComponentApi, ComponentIdSchema } from '@a2ui/web_core/v0_9';
import { useDesignSystemTheme } from '@databricks/design-system';

/**
 * Schema (API) for our custom Card.
 */
const CardApi = {
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

export const Card: ReactComponentImplementation = createComponentImplementation(CardApi, ({ props, buildChild }) => {
  const { theme } = useDesignSystemTheme();
  const weight = typeof props.weight === 'number' ? props.weight : undefined;

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
      {typeof props.child === 'string' ? buildChild(props.child) : null}
    </div>
  );
});
