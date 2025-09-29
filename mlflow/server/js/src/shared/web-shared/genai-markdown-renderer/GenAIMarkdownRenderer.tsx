import React, { type ComponentType, useMemo } from 'react';
import type { Components, Options, UrlTransform } from 'react-markdown-10';
import ReactMarkdown, { defaultUrlTransform } from 'react-markdown-10';
import remarkGfm from 'remark-gfm-4';

import { TableCell, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { CodeSnippetLanguage } from '@databricks/web-shared/snippet';
import { CodeSnippet, SnippetCopyAction } from '@databricks/web-shared/snippet';
import { TableRenderer, VirtualizedTableCell, VirtualizedTableRow } from './TableRenderer';
import type { ReactMarkdownComponent, ReactMarkdownComponents, ReactMarkdownProps } from './types';

/**
 * NOTE: react-markdown sanitizes urls by default, including `data:` urls, with the `urlTransform` prop, documented here: https://github.com/remarkjs/react-markdown?tab=readme-ov-file#defaulturltransformurl
 * It uses `micromark-util-sanitize-uri` package under the hood to escape urls and prevent injection: https://github.com/micromark/micromark/tree/main/packages/micromark-util-sanitize-uri#readme
 * We can allow jpeg and png data urls, and use the default transformer for everything else.
 */
const urlTransform: UrlTransform = (value) => {
  if (value.startsWith('data:image/png') || value.startsWith('data:image/jpeg')) {
    return value;
  }
  return defaultUrlTransform(value);
};

export const GenAIMarkdownRenderer = (props: { children: string; components?: ExtendedComponents }) => {
  const components: Components = useMemo(
    () => getMarkdownComponents({ extensions: props.components }),
    [props.components],
  );
  return (
    <ReactMarkdown components={components} remarkPlugins={RemarkPlugins} urlTransform={urlTransform}>
      {props.children}
    </ReactMarkdown>
  );
};

const CodeMarkdownComponent = ({
  codeBlock,
  codeInline,
  node,
  ...codeProps
}: Required<ExtededCodeRenderers> & ReactMarkdownProps<'code'>) => {
  const language = React.useMemo(() => {
    const match = /language-(\w+)/.exec(codeProps.className ?? '');
    return match && match[1] ? match[1] : undefined;
  }, [codeProps.className]);

  if (node?.position?.start.line === node?.position?.end.line) {
    return React.createElement(codeInline, codeProps);
  }

  return React.createElement(codeBlock, { ...codeProps, language });
};

const InlineCode = ({ children }: ReactMarkdownProps<'code'>) => <Typography.Text code>{children}</Typography.Text>;

/**
 * Since this component is quite expensive to render we memoize it so if multiple
 * code blocks are being rendered, we only update the code blocks with changing props
 */
const CodeBlock = React.memo(({ children, language }: ReactMarkdownProps<'code'> & { language?: string }) => {
  const { theme } = useDesignSystemTheme();
  const code = String(children).replace(/\n$/, '');
  return (
    <div css={{ position: 'relative' }}>
      <CodeSnippet
        actions={<SnippetCopyAction componentId="genai.util.markdown-copy-code-block" copyText={code} />}
        theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
        children={code}
        language={language && isCodeSnippetLanguage(language) ? language : 'text'}
        style={{
          padding: '8px 0',
          borderRadius: 8,
          width: '100%',
          boxSizing: 'border-box',
          // Setting a reasonable max height to avoid long code blocks taking up the entire screen.
          // Component handles scrolling inside it gracefully
          maxHeight: 640,
          // Using column-reverse flex layout so the scroll position will stick to the bottom
          // as new content is streamed in.
          display: 'flex',
          flexDirection: 'column-reverse',
        }}
        showLineNumbers
      />
    </div>
  );
});

const RemarkPlugins: Options['remarkPlugins'] = [remarkGfm];

// react-markdown handles both inline and block code rendering in the same component
// however, we want to render them differently so we need to split them into two components.
// This also allows callees to override the default renderers separately
type ExtededCodeRenderers = {
  codeInline?: ReactMarkdownComponent<'code'>;
  codeBlock?: ComponentType<React.PropsWithChildren<Omit<ReactMarkdownProps<'code'>, 'ref'> & { language?: string }>>;
};

type ExtendedComponents = Omit<ReactMarkdownComponents, 'code'> & ExtededCodeRenderers;

export const getMarkdownComponents = (props: { extensions?: ExtendedComponents }) =>
  ({
    a: ({ href, children }) => (
      <Typography.Link
        componentId="codegen_webapp_js_genai_util_markdown.tsx_71"
        href={href}
        // If the link is to the footnote (starts with #user-content-fn), set id so footnote can link back to it
        id={
          href?.startsWith('#user-content-fn-') ? href.replace('#user-content-fn-', 'user-content-fnref-') : undefined
        }
        disabled={href?.startsWith('.')}
        // If the link is to the footnote, add brackets around the children to make it appear as a footnote reference
        children={href?.startsWith('#user-content-fn-') ? <>[{children}]</> : children}
        // If the link is an id link, don't open in new tab
        openInNewTab={!(href && href.startsWith('#'))}
      />
    ),
    code: (codeProps) => (
      <CodeMarkdownComponent
        {...codeProps}
        codeBlock={props.extensions?.codeBlock ?? CodeBlock} // Optionally override the default code block renderer
        codeInline={props.extensions?.codeInline ?? InlineCode} // Optionally override the default inline code renderer
      />
    ),
    p: ({ children }) => <Typography.Paragraph children={children} />,
    h1: ({ children }) => <Typography.Title level={1} children={children} />,
    h2: ({ children }) => <Typography.Title level={2} children={children} />,
    h3: ({ children }) => <Typography.Title level={3} children={children} />,
    h4: ({ children }) => <Typography.Title level={4} children={children} />,
    h5: ({ children }) => <Typography.Title level={5} children={children} />,
    table: ({ children, node }) => <TableRenderer children={children} node={node} />,
    tr: ({ children, node }) => <VirtualizedTableRow children={children} node={node} />,
    th: ({ children, node }) => <VirtualizedTableCell children={children} node={node} />,
    // Without the multiline prop, the table cell will add ellipsis to the text effictively hiding the content
    // for long text. This is not the desired behavior for markdown tables.
    td: ({ children }) => <TableCell children={children} multiline />,
    // Design system's table does not use thead and tbody elements
    thead: ({ children }) => <>{children}</>,
    tbody: ({ children }) => <>{children}</>,
    img: ({ src, alt }) => <img src={src} alt={alt} css={{ maxWidth: '100%' }} />,
  } satisfies ReactMarkdownComponents);

const isCodeSnippetLanguage = (languageString: string): languageString is CodeSnippetLanguage => {
  // Casting the string to string literal so we can exhaust the union
  const typeCast = languageString as CodeSnippetLanguage;
  switch (typeCast) {
    case 'go':
    case 'java':
    case 'javascript':
    case 'json':
    case 'python':
    case 'sql':
    case 'text':
    case 'yaml':
      return true;
    default:
      return false;
  }
};
