import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter';
import go from 'react-syntax-highlighter/dist/cjs/languages/prism/go';
import java from 'react-syntax-highlighter/dist/cjs/languages/prism/java';
import javascript from 'react-syntax-highlighter/dist/cjs/languages/prism/javascript';
import json from 'react-syntax-highlighter/dist/cjs/languages/prism/json';
import python from 'react-syntax-highlighter/dist/cjs/languages/prism/python';
import yaml from 'react-syntax-highlighter/dist/cjs/languages/prism/yaml';
import sql from 'react-syntax-highlighter/dist/cjs/languages/prism/sql';

SyntaxHighlighter.registerLanguage('sql', sql);
SyntaxHighlighter.registerLanguage('java', java);
SyntaxHighlighter.registerLanguage('python', python);
SyntaxHighlighter.registerLanguage('go', go);
SyntaxHighlighter.registerLanguage('javascript', javascript);
SyntaxHighlighter.registerLanguage('yaml', yaml);
SyntaxHighlighter.registerLanguage('json', json);

import duotoneDarkStyle from './theme/databricks-duotone-dark';
import lightStyle from './theme/databricks-light';
import type { CSSProperties, ReactNode } from 'react';
import { pick } from 'lodash';
export type CodeSnippetTheme = 'duotoneDark' | 'light';
export const buttonBackgroundColorDark = 'rgba(140, 203, 255, 0)';
export const buttonColorDark = 'rgba(255, 255, 255, 0.84)';
export const buttonHoverColorDark = '#8ccbffcc';
export const buttonHoverBackgroundColorDark = 'rgba(140, 203, 255, 0.08)';
export const duboisAlertBackgroundColor = '#fff0f0';
export const snippetPadding = '24px';
const themesStyles: Record<CodeSnippetTheme, any> = {
  light: lightStyle,
  duotoneDark: duotoneDarkStyle,
};

export type CodeSnippetLanguage = 'sql' | 'java' | 'python' | 'javascript' | 'go' | 'yaml' | 'text' | 'json';

export interface CodeSnippetProps {
  /**
   * The code string
   */
  children: string;
  /**
   * The actions that are displayed on the right top corner of the component
   *  see `./actions` for built-in actions
   */
  actions?: NonNullable<ReactNode> | NonNullable<ReactNode>[];
  /**
   * The theme, default theme is `light`
   */
  theme?: CodeSnippetTheme;
  /**
   * Language of the code (`children`)
   */
  language: CodeSnippetLanguage;
  /**
   * Custom styles (passed to the internal `<pre>`)
   */
  style?: CSSProperties;
  /**
   * Whether to show line numbers on the left or not
   */
  showLineNumbers?: boolean;
  /**
   * Custom styles for line numbers
   */
  lineNumberStyle?: CSSProperties;
  /**
   * Boolean to specify whether to style the <code> block with white-space: pre-wrap or white-space: pre
   */
  wrapLongLines?: boolean;
  /**
   * Boolean that determines whether or not each line of code should be wrapped in a parent element
   */
  wrapLines?: boolean;
  /**
   * Props to pass to the line elements
   */
  lineProps?: React.HTMLProps<HTMLElement> | undefined;
  /**
   * Custom tag to use for the `<pre>` element
   */
  PreTag?: keyof JSX.IntrinsicElements | React.ComponentType<React.PropsWithChildren<any>> | undefined;
}

/**
 * `CodeSnippet` is used for highlighting code, use this instead of
 */
export function CodeSnippet({
  theme = 'light',
  language,
  actions,
  style,
  children,
  showLineNumbers,
  lineNumberStyle,
  wrapLongLines,
  wrapLines,
  PreTag,
}: CodeSnippetProps) {
  const customStyle = {
    border: 'none',
    borderRadius: 0,
    margin: 0,
    padding: snippetPadding,
    ...style,
  };
  return (
    <SyntaxHighlighter
      showLineNumbers={showLineNumbers}
      lineNumberStyle={lineNumberStyle}
      language={language}
      style={themesStyles[theme]}
      customStyle={customStyle}
      codeTagProps={{
        style: pick(style, 'backgroundColor'),
      }}
      wrapLongLines={wrapLongLines}
      wrapLines={wrapLines}
      PreTag={PreTag}
    >
      {children}
    </SyntaxHighlighter>
  );
}

export * from './actions/SnippetCopyAction';
