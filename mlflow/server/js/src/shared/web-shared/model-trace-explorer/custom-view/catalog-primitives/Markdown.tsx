import type { ReactNode } from 'react';
import { useMemo } from 'react';

import { z } from 'zod';
import { createComponentImplementation, type ReactComponentImplementation } from '@a2ui/react/v0_9';
import { type ComponentApi, DynamicStringSchema } from '@a2ui/web_core/v0_9';
import { Typography, useDesignSystemTheme } from '@databricks/design-system';

import { asString } from '../catalogPrimitiveUtils';

import { GenAIMarkdownRenderer } from '../../../genai-markdown-renderer/GenAIMarkdownRenderer';
import type { GenAIMarkdownRendererProps } from '../../../genai-markdown-renderer/GenAIMarkdownRenderer';

/**
 * Schema (API) for the Markdown component
 */
const MarkdownApi = {
  name: 'Markdown',
  schema: z
    .object({
      text: DynamicStringSchema.describe('The markdown body.'),
      title: DynamicStringSchema.describe('Optional heading shown above the markdown.').optional(),
    })
    .strict(),
} satisfies ComponentApi;

export const Markdown: ReactComponentImplementation = createComponentImplementation(MarkdownApi, ({ props }) => {
  const { theme } = useDesignSystemTheme();

  const text = props.text ? asString(props.text) : '';
  const title = props.title ? asString(props.title) : undefined;

  // Render markdown links as normal links that open in a new tab.
  const markdownComponents = useMemo<GenAIMarkdownRendererProps['components']>(
    () => ({
      a: ({ href, children }: { href?: string; children?: ReactNode }) => (
        <Typography.Link componentId="shared.model-trace-explorer.custom-view.markdown.link" href={href} openInNewTab>
          {children}
        </Typography.Link>
      ),
    }),
    [],
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
