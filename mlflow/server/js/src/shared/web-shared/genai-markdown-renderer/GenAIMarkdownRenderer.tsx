import type { CSSObject } from '@emotion/react';
import React, { type ComponentType, useMemo } from 'react';
import type { Components, Options } from 'react-markdown-10';
import ReactMarkdown, { defaultUrlTransform } from 'react-markdown-10';
import remarkGfm from 'remark-gfm-4';

import { TableCell, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { CodeSnippetLanguage } from '../snippet/CodeSnippet';
import { CodeSnippet } from '../snippet/CodeSnippet';
import { SnippetCopyAction } from '../snippet/actions/SnippetCopyAction';

import { TableRenderer, VirtualizedTableCell, VirtualizedTableRow } from './TableRenderer';
import type { ReactMarkdownComponent, ReactMarkdownComponents, ReactMarkdownProps } from './types';

interface OverrideStyles {
  paragraph?: CSSObject;
  heading?: CSSObject;
  list?: CSSObject;
}

const useCompactMarkdownStyles = (): OverrideStyles => {
  const { theme } = useDesignSystemTheme();
  return useMemo(
    () => ({
      paragraph: { marginBottom: theme.spacing.xs, marginTop: 0 },
      heading: { marginBottom: theme.spacing.xs, marginTop: 0 },
      list: { marginBottom: theme.spacing.xs, marginTop: 0, paddingLeft: theme.spacing.lg },
    }),
    [theme.spacing.xs, theme.spacing.lg],
  );
};

export interface GenAIMarkdownRendererProps extends Pick<Options, 'urlTransform'> {
  children: string;
  components?: ExtendedComponents;
  /** When true, renders markdown with reduced margins/spacing for compact UI contexts */
  compact?: boolean;
}

export const GenAIMarkdownRenderer = (props: GenAIMarkdownRendererProps) => {
  const compactStyles = useCompactMarkdownStyles();
  const overrideStyles = props.compact ? compactStyles : undefined;

  const components: Components = useMemo(
    () => getMarkdownComponents({ extensions: props.components, overrideStyles }),
    [props.components, overrideStyles],
  );
  return (
    <ReactMarkdown
      components={components}
      remarkPlugins={RemarkPlugins}
      urlTransform={props.urlTransform ?? urlTransform}
    >
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

const InlineCode = ({ children }: ReactMarkdownProps<'code'>) => (
  <Typography.Text code css={{ whiteSpace: 'pre-wrap' }}>
    {children}
  </Typography.Text>
);

/**
 * Since this component is quite expensive to render we memoize it so if multiple
 * code blocks are being rendered, we only update the code blocks with changing props
 */
// eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
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

/**
 * Custom URL transform that extends the default to also allow data: and blob: URLs.
 * The default urlTransform in react-markdown-10 only allows http(s), mailto, irc(s), and xmpp protocols,
 * which causes data URLs (e.g., base64 images) to be stripped.
 */
const urlTransform: Options['urlTransform'] = (url: string) => {
  // Allow data: URLs (e.g., base64 encoded images)
  if (url.startsWith('data:')) {
    return url;
  }
  // Allow blob: URLs (locally generated)
  if (url.startsWith('blob:')) {
    return url;
  }
  // Fall back to default transform for other URLs
  return defaultUrlTransform(url);
};

const isRelativeUrl = (url: string | undefined): boolean => {
  if (!url) return false;
  // Check if URL has a protocol (absolute URL)
  return !/^[a-z][a-z0-9+.-]*:/i.test(url);
};

// react-markdown handles both inline and block code rendering in the same component
// however, we want to render them differently so we need to split them into two components.
// This also allows callees to override the default renderers separately
type ExtededCodeRenderers = {
  codeInline?: ReactMarkdownComponent<'code'>;
  codeBlock?: ComponentType<React.PropsWithChildren<Omit<ReactMarkdownProps<'code'>, 'ref'> & { language?: string }>>;
};

type ExtendedComponents = Omit<ReactMarkdownComponents, 'code'> & ExtededCodeRenderers;

export const getMarkdownComponents = (props: { extensions?: ExtendedComponents; overrideStyles?: OverrideStyles }) =>
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
    p: ({ children }) => <Typography.Paragraph css={props.overrideStyles?.paragraph}>{children}</Typography.Paragraph>,
    h1: ({ children }) => (
      <Typography.Title level={1} css={props.overrideStyles?.heading}>
        {children}
      </Typography.Title>
    ),
    h2: ({ children }) => (
      <Typography.Title level={2} css={props.overrideStyles?.heading}>
        {children}
      </Typography.Title>
    ),
    h3: ({ children }) => (
      <Typography.Title level={3} css={props.overrideStyles?.heading}>
        {children}
      </Typography.Title>
    ),
    h4: ({ children }) => (
      <Typography.Title level={4} css={props.overrideStyles?.heading}>
        {children}
      </Typography.Title>
    ),
    h5: ({ children }) => (
      <Typography.Title level={5} css={props.overrideStyles?.heading}>
        {children}
      </Typography.Title>
    ),
    ul: ({ children }) => <ul css={props.overrideStyles?.list}>{children}</ul>,
    ol: ({ children }) => <ol css={props.overrideStyles?.list}>{children}</ol>,
    table: ({ children, node }) => <TableRenderer children={children} node={node} />,
    tr: ({ children, node }) => <VirtualizedTableRow children={children} node={node} />,
    th: ({ children, node }) => <VirtualizedTableCell children={children} node={node} />,
    // Without the multiline prop, the table cell will add ellipsis to the text effictively hiding the content
    // for long text. This is not the desired behavior for markdown tables.
    td: ({ children }) => <TableCell children={children} multiline />,
    // Design system's table does not use thead and tbody elements
    thead: ({ children }) => <>{children}</>,
    tbody: ({ children }) => <>{children}</>,
    img: ({ src, alt }) =>
      isRelativeUrl(src) ? <span>{`[${alt}](${src})`}</span> : <img src={src} alt={alt} css={{ maxWidth: '100%' }} />,
    ...props.extensions,
  }) satisfies ReactMarkdownComponents;

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
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      const exhaust: never = typeCast;
      return false;
  }
};
