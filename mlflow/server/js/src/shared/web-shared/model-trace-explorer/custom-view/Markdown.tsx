import type { ReactNode } from 'react';
import { useMemo } from 'react';

import { z } from 'zod';
import { createComponentImplementation } from '@a2ui/react/v0_9';
import { type ComponentApi, DynamicStringSchema } from '@a2ui/web_core/v0_9';
import { Typography, useDesignSystemTheme } from '@databricks/design-system';

import { GenAIMarkdownRenderer } from '../../genai-markdown-renderer/GenAIMarkdownRenderer';
import type { GenAIMarkdownRendererProps } from '../../genai-markdown-renderer/GenAIMarkdownRenderer';
import { useTreeSelection } from './TreeSelectionContext';

// Markdown link scheme that selects the node representing a span instead of
// navigating: `[label](#span:<spanId>)`. Only active when rendered inside a
// TreeView (otherwise it falls back to a normal link).
const SPAN_DEEPLINK_PREFIX = '#span:';

/**
 * Schema (API) for the Markdown component: renders a markdown `text` block
 * (optionally under a `title`). Links of the form `[text](#span:<spanId>)`
 * select the TreeView node representing that span instead of navigating, so a
 * span summary can deeplink into the spans it references.
 */
export const MarkdownApi = {
  name: 'Markdown',
  schema: z
    .object({
      text: DynamicStringSchema.describe(
        'The markdown body. Supports span deeplinks of the form [text](#span:<spanId>) when rendered inside a TreeView.',
      ),
      title: DynamicStringSchema.describe('Optional heading shown above the markdown.').optional(),
    })
    .strict(),
} satisfies ComponentApi;

const asString = (value: unknown): string => (typeof value === 'string' ? value : String(value ?? ''));

export const Markdown = createComponentImplementation(MarkdownApi, ({ props }) => {
  const { theme } = useDesignSystemTheme();
  const selection = useTreeSelection();

  const text = props.text ? asString(props.text) : '';
  const title = props.title ? asString(props.title) : undefined;

  // Anchor override: intercept span deeplinks (when inside a TreeView); render
  // normal links as usual.
  const markdownComponents = useMemo<GenAIMarkdownRendererProps['components']>(
    () => ({
      a: ({ href, children }: { href?: string; children?: ReactNode }) => {
        if (selection.enabled && href && href.startsWith(SPAN_DEEPLINK_PREFIX)) {
          const targetSpanId = href.slice(SPAN_DEEPLINK_PREFIX.length);
          return (
            <Typography.Link
              componentId="shared.model-trace-explorer.custom-view.markdown.span-deeplink"
              href={href}
              onClick={(event) => {
                event.preventDefault();
                selection.selectSpan(targetSpanId);
              }}
            >
              {children}
            </Typography.Link>
          );
        }
        return (
          <Typography.Link
            componentId="shared.model-trace-explorer.custom-view.markdown.link"
            href={href}
            openInNewTab
          >
            {children}
          </Typography.Link>
        );
      },
    }),
    [selection],
  );

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
      {title && (
        <Typography.Text bold size="lg">
          {title}
        </Typography.Text>
      )}
      <GenAIMarkdownRenderer components={markdownComponents} compact>
        {text}
      </GenAIMarkdownRenderer>
    </div>
  );
});
