import { createComponentImplementation, type ReactComponentImplementation } from '@a2ui/react/v0_9';
import { TextApi } from '@a2ui/web_core/v0_9/basic_catalog';
import { Typography } from '@databricks/design-system';

import { asString } from '../catalogPrimitiveUtils';

/**
 * Custom Text primitive that overrides the basic catalog's Text.
 */
const HEADING_STYLE = {
  h1: { fontSize: 32, lineHeight: '40px', fontWeight: 700 },
  h2: { fontSize: 26, lineHeight: '34px', fontWeight: 700 },
  h3: { fontSize: 22, lineHeight: '30px', fontWeight: 600 },
  h4: { fontSize: 19, lineHeight: '26px', fontWeight: 600 },
  h5: { fontSize: 16, lineHeight: '22px', fontWeight: 600 },
} satisfies Record<string, { fontSize: number; lineHeight: string; fontWeight: number }>;

const isHeadingVariant = (value: string): value is keyof typeof HEADING_STYLE => Object.hasOwn(HEADING_STYLE, value);

const HEADING_ELEMENT_LEVEL: Record<keyof typeof HEADING_STYLE, 1 | 2 | 3 | 4 | 5> = {
  h1: 1,
  h2: 2,
  h3: 3,
  h4: 4,
  h5: 5,
};

export const Text: ReactComponentImplementation = createComponentImplementation(TextApi, ({ props }) => {
  const text = asString(props.text);
  const variant = typeof props.variant === 'string' ? props.variant : 'body';
  const weight = typeof props.weight === 'number' ? props.weight : undefined;
  const flexStyle = weight !== undefined ? { flex: `${weight}`, minWidth: 0 } : undefined;

  if (isHeadingVariant(variant)) {
    // `level` only picks the Title's visual emphasis, and the explicit HEADING_STYLE size
    // overrides it anyway, so h4/h5 can share level 3. `elementLevel` renders the heading
    // element for the requested variant so the document outline stays accurate.
    const level = variant === 'h1' ? 1 : variant === 'h2' ? 2 : 3;
    return (
      <Typography.Title
        level={level}
        elementLevel={HEADING_ELEMENT_LEVEL[variant]}
        withoutMargins
        css={{ ...flexStyle, ...HEADING_STYLE[variant] }}
      >
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
