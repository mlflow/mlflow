import { createComponentImplementation } from '@a2ui/react/v0_9';
import { TextApi } from '@a2ui/web_core/v0_9/basic_catalog';
import { Typography } from '@databricks/design-system';

/**
 * Custom Text primitive that overrides the basic catalog's Text.
 */
const asString = (value: unknown): string => (typeof value === 'string' ? value : String(value ?? ''));

const HEADING_STYLE: Record<string, { fontSize: number; lineHeight: string; fontWeight: number }> = {
  h1: { fontSize: 32, lineHeight: '40px', fontWeight: 700 },
  h2: { fontSize: 26, lineHeight: '34px', fontWeight: 700 },
  h3: { fontSize: 22, lineHeight: '30px', fontWeight: 600 },
  h4: { fontSize: 19, lineHeight: '26px', fontWeight: 600 },
  h5: { fontSize: 16, lineHeight: '22px', fontWeight: 600 },
};

export const Text = createComponentImplementation(TextApi, ({ props }) => {
  const text = asString(props.text);
  const variant = typeof props.variant === 'string' ? props.variant : 'body';
  const weight = typeof props.weight === 'number' ? props.weight : undefined;
  const flexStyle = weight !== undefined ? { flex: `${weight}`, minWidth: 0 } : undefined;

  const headingStyle = HEADING_STYLE[variant];
  if (headingStyle) {
    // h1–h3 map to a Title element for semantics; h4/h5 use the explicit size
    // on a lower-emphasis Title level so the override drives the visual size.
    const level = variant === 'h1' ? 1 : variant === 'h2' ? 2 : 3;
    return (
      <Typography.Title level={level} withoutMargins css={{ ...flexStyle, ...headingStyle }}>
        {text}
      </Typography.Title>
    );
  }

  if (variant === 'caption') {
    return (
      <Typography.Text size="sm" color="secondary" css={flexStyle}>
        {text}
      </Typography.Text>
    );
  }

  return <Typography.Text css={flexStyle}>{text}</Typography.Text>;
});
