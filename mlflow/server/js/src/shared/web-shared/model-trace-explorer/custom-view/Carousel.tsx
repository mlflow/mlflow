import { useMemo, useState } from 'react';

import { z } from 'zod';
import { createComponentImplementation } from '@a2ui/react/v0_9';
import { type ComponentApi, ChildListSchema, DynamicStringSchema } from '@a2ui/web_core/v0_9';
import {
  Button,
  ChevronLeftIcon,
  ChevronRightIcon,
  Empty,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';

/**
 * Schema (API) for the generic Carousel component. It renders one child at a
 * time with previous/next controls and a position indicator. Domain-agnostic:
 * callers supply a list of child component ids (e.g. one FeedbackForm per
 * scoped span) and the carousel steps through them.
 */
export const CarouselApi = {
  name: 'Carousel',
  schema: z
    .object({
      children: ChildListSchema.describe('The child component ids to step through, in order.'),
      emptyMessage: DynamicStringSchema.describe('Text shown when there are no children.').optional(),
    })
    .strict(),
} satisfies ComponentApi;

const asString = (value: unknown): string => (typeof value === 'string' ? value : String(value ?? ''));

export const Carousel = createComponentImplementation(CarouselApi, ({ props, buildChild }) => {
  const { theme } = useDesignSystemTheme();

  // `children` resolves to a static array of child component ids.
  const childIds = useMemo(() => (Array.isArray(props.children) ? (props.children as string[]) : []), [props.children]);
  const emptyMessage = props.emptyMessage ? asString(props.emptyMessage) : 'Nothing to review.';

  const [index, setIndex] = useState(0);
  const count = childIds.length;
  const safeIndex = count > 0 ? Math.min(index, count - 1) : 0;

  if (count === 0) {
    return <Empty description={emptyMessage} />;
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: theme.spacing.sm }}>
        <Button
          componentId="shared.model-trace-explorer.custom-view.carousel.prev"
          size="small"
          icon={<ChevronLeftIcon />}
          aria-label="Previous"
          disabled={safeIndex === 0}
          onClick={() => setIndex((prev) => Math.max(prev - 1, 0))}
        />
        <Typography.Text color="secondary" size="sm">
          {safeIndex + 1} of {count}
        </Typography.Text>
        <Button
          componentId="shared.model-trace-explorer.custom-view.carousel.next"
          size="small"
          icon={<ChevronRightIcon />}
          aria-label="Next"
          disabled={safeIndex >= count - 1}
          onClick={() => setIndex((prev) => Math.min(prev + 1, count - 1))}
        />
      </div>
      <div>{buildChild(childIds[safeIndex])}</div>
    </div>
  );
});
