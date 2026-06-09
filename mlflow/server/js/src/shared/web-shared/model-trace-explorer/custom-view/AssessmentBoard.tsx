import type { ComponentType } from 'react';
import { useMemo } from 'react';

import { z } from 'zod';
import { createComponentImplementation } from '@a2ui/react/v0_9';
import { type ComponentApi, ChildListSchema, DynamicStringSchema } from '@a2ui/web_core/v0_9';
import {
  ChecklistIcon,
  CheckCircleIcon,
  ListBorderIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';

const ICON_NAMES = ['checklist', 'list', 'checkCircle'] as const;
type IconName = (typeof ICON_NAMES)[number];

const ICON_BY_NAME: Record<IconName, ComponentType> = {
  checklist: ChecklistIcon,
  list: ListBorderIcon,
  checkCircle: CheckCircleIcon,
};

/**
 * Schema (API) for the AssessmentBoard container. It renders an optional titled
 * header followed by a wrapping row of child component ids — typically one
 * AssessmentCard per assessment. Because children are just ids, more
 * assessments can be added by appending more AssessmentCards; each one flows
 * into the row and wraps automatically.
 */
export const AssessmentBoardApi = {
  name: 'AssessmentBoard',
  schema: z
    .object({
      title: DynamicStringSchema.describe('Optional heading shown above the boxes.').optional(),
      icon: z.enum(ICON_NAMES).describe('Optional icon shown next to the title.').optional(),
      children: ChildListSchema.describe('The card component ids to lay out in a wrapping row.'),
      emptyMessage: DynamicStringSchema.describe('Text shown when there are no cards.').optional(),
    })
    .strict(),
} satisfies ComponentApi;

const asString = (value: unknown): string => (typeof value === 'string' ? value : String(value ?? ''));

export const AssessmentBoard = createComponentImplementation(AssessmentBoardApi, ({ props, buildChild }) => {
  const { theme } = useDesignSystemTheme();

  const childIds = useMemo(() => (Array.isArray(props.children) ? (props.children as string[]) : []), [props.children]);
  const title = props.title ? asString(props.title) : undefined;
  const emptyMessage = props.emptyMessage ? asString(props.emptyMessage) : 'No assessments to display.';
  const IconComponent = props.icon ? ICON_BY_NAME[props.icon as IconName] : ChecklistIcon;

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      {title && (
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <span css={{ display: 'flex', color: theme.colors.textSecondary }}>
            <IconComponent />
          </span>
          <Typography.Text bold size="lg">
            {title}
          </Typography.Text>
        </div>
      )}

      {childIds.length === 0 ? (
        <Typography.Text color="secondary">{emptyMessage}</Typography.Text>
      ) : (
        <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.md, alignItems: 'stretch' }}>
          {childIds.map((id) => (
            <div key={id} css={{ flex: '1 1 260px', minWidth: 260, maxWidth: 420, display: 'flex' }}>
              {buildChild(id)}
            </div>
          ))}
        </div>
      )}
    </div>
  );
});
