import type { ComponentType } from 'react';

import { z } from 'zod';
import { createComponentImplementation } from '@a2ui/react/v0_9';
import { type ComponentApi, DynamicStringSchema } from '@a2ui/web_core/v0_9';
import {
  ChecklistIcon,
  ClockIcon,
  Empty,
  HashIcon,
  ListBorderIcon,
  Typography,
  useDesignSystemTheme,
  WrenchIcon,
} from '@databricks/design-system';

import { ModelTraceExplorerCodeSnippet } from '../ModelTraceExplorerCodeSnippet';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';

const ICON_NAMES = ['list', 'wrench', 'clock', 'hash', 'checklist'] as const;
type IconName = (typeof ICON_NAMES)[number];

const ICON_BY_NAME: Record<IconName, ComponentType> = {
  list: ListBorderIcon,
  wrench: WrenchIcon,
  clock: ClockIcon,
  hash: HashIcon,
  checklist: ChecklistIcon,
};

/**
 * Schema (API) for the generic ContentViewer component. It renders a single
 * titled, optionally-collapsible content block whose fields are shown with the
 * same code snippet renderer as the trace explorer's "Details & Timeline" view
 * (per-field JSON / Text / Markdown / Table toggle).
 *
 * Intentionally domain-agnostic and scoped to ONE content payload: callers
 * render it once for inputs and once for outputs (or attributes, etc.), rather
 * than baking input+output into a single component. Each `fields[].value` should
 * be a JSON-encoded string (objects are rendered as JSON; plain strings render
 * as text/markdown), mirroring how the Details view lists key/value entries.
 */
export const ContentViewerApi = {
  name: 'ContentViewer',
  schema: z
    .object({
      title: DynamicStringSchema.describe('Heading for the block, e.g. "Inputs" or "Outputs".').optional(),
      icon: z.enum(ICON_NAMES).describe('Optional icon shown next to the title.').optional(),
      fields: z
        .array(
          z.object({
            label: DynamicStringSchema.describe('The field key shown above its value.').optional(),
            value: DynamicStringSchema.describe('JSON-encoded field value (object -> JSON, string -> text).'),
          }),
        )
        .describe('The key/value entries to display.')
        .optional(),
      collapsible: z.boolean().describe('Whether the block is wrapped in a collapsible section.').optional(),
      emptyMessage: DynamicStringSchema.describe('Text shown when there are no fields.').optional(),
    })
    .strict(),
} satisfies ComponentApi;

const asString = (value: unknown): string => (typeof value === 'string' ? value : String(value ?? ''));

// ModelTraceExplorerCodeSnippet does `JSON.parse(data)`, so `data` must be a
// valid JSON string. Values are expected to already be JSON-encoded by the
// host; if a raw (non-JSON) string slips through, encode it so rendering is safe.
const toJsonString = (value: unknown): string => {
  const str = asString(value);
  try {
    JSON.parse(str);
    return str;
  } catch {
    return JSON.stringify(str);
  }
};

// Plain (non-A2UI) renderer for a single content payload. Shared by the A2UI
// `ContentViewer` component and by any host React component that wants the same
// "Details & Timeline"-style content block (e.g. the FeedbackForm card).
export const ContentBlock = ({
  title,
  icon,
  fields,
  collapsible = true,
  emptyMessage = 'No content to display.',
}: {
  title?: string;
  icon?: IconName;
  fields: { label?: unknown; value?: unknown }[];
  collapsible?: boolean;
  emptyMessage?: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const IconComponent = icon ? ICON_BY_NAME[icon] : undefined;

  const body =
    fields.length === 0 ? (
      <Empty description={emptyMessage} />
    ) : (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        {fields.map((field, index) => (
          <ModelTraceExplorerCodeSnippet
            key={field?.label ? asString(field.label) : index}
            title={field?.label ? asString(field.label) : ''}
            data={toJsonString(field?.value)}
          />
        ))}
      </div>
    );

  const titleNode = (
    <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}>
      {IconComponent && <IconComponent />}
      {title}
    </span>
  );

  if (!collapsible) {
    return (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        {title && (
          <Typography.Title withoutMargins level={4}>
            {titleNode}
          </Typography.Title>
        )}
        {body}
      </div>
    );
  }

  return (
    <ModelTraceExplorerCollapsibleSection withBorder sectionKey={title ?? 'content'} title={titleNode}>
      {body}
    </ModelTraceExplorerCollapsibleSection>
  );
};

export const ContentViewer = createComponentImplementation(ContentViewerApi, ({ props }) => {
  const fields = Array.isArray(props.fields) ? props.fields : [];
  const title = props.title ? asString(props.title) : undefined;
  const collapsible = props.collapsible !== false;
  const emptyMessage = props.emptyMessage ? asString(props.emptyMessage) : 'No content to display.';

  return (
    <ContentBlock
      title={title}
      icon={props.icon as IconName | undefined}
      fields={fields}
      collapsible={collapsible}
      emptyMessage={emptyMessage}
    />
  );
});
