import { isNil, isString } from 'lodash';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { Button, ChevronDownIcon, ChevronUpIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { SearchMatch } from './ModelTrace.types';
import { CodeSnippetRenderMode } from './ModelTrace.types';
import { CollapsibleJsonViewer, getJsonColors } from './CollapsibleJsonViewer';
import { ModelTraceExplorerHighlightedCodeSnippet } from '../ModelTraceExplorerHighlightedCodeSnippet';
import { formatJsonStringAsYaml } from './ModelTraceExplorerYaml.utils';
import { GenAIMarkdownRenderer } from '../../genai-markdown-renderer/GenAIMarkdownRenderer';
import { CodeSnippet } from '../../snippet/CodeSnippet';
import { SnippetCopyAction } from '../../snippet/actions/SnippetCopyAction';

const MAX_LINES_FOR_PREVIEW = 10;
// the `isContentLong` check does not work for
// markdown rendering, since the content is wrapped
const MAX_CHARS_FOR_PREVIEW = 300;
const YAML_MAX_LINES_FOR_PREVIEW = 100;
const YAML_MAX_CHARS_FOR_PREVIEW = 12000;
const JSON_PREVIEW_HEIGHT = 110;

export function ModelTraceExplorerCodeSnippetBody({
  data,
  searchFilter = '',
  activeMatch = null,
  containsActiveMatch = false,
  renderMode = CodeSnippetRenderMode.JSON,
  initialExpanded = false,
}: {
  data: string;
  searchFilter?: string;
  activeMatch?: SearchMatch | null;
  containsActiveMatch?: boolean;
  renderMode?: CodeSnippetRenderMode;
  initialExpanded?: boolean;
}): JSX.Element {
  const { theme } = useDesignSystemTheme();
  const isYamlRenderMode = renderMode === CodeSnippetRenderMode.YAML;
  const yamlData = useMemo(() => (isYamlRenderMode ? formatJsonStringAsYaml(data) : data), [data, isYamlRenderMode]);
  const searchableData = isYamlRenderMode ? yamlData : data;
  const containsMatches =
    Boolean(searchFilter) && !isNil(activeMatch) && searchableData.toLowerCase().includes(searchFilter);
  const [isContentLong, setIsContentLong] = useState(false);
  const [expanded, setExpanded] = useState(initialExpanded || containsMatches);
  const snippetRef = useRef<HTMLPreElement>(null);
  const jsonViewerRef = useRef<HTMLDivElement>(null);
  // if the data is rendered in text / markdown mode, then
  // we need to parse it so that the newlines are unescaped
  const dataToTruncate: string = useMemo(() => {
    if (renderMode === 'json') {
      return data;
    }

    if (isYamlRenderMode) {
      return yamlData;
    }

    const parsedData = JSON.parse(data);
    if (isString(parsedData)) {
      return parsedData;
    }

    return data;
  }, [data, renderMode, yamlData, isYamlRenderMode]);

  const maxLinesForPreview = isYamlRenderMode ? YAML_MAX_LINES_FOR_PREVIEW : MAX_LINES_FOR_PREVIEW;
  const maxCharsForPreview = isYamlRenderMode ? YAML_MAX_CHARS_FOR_PREVIEW : MAX_CHARS_FOR_PREVIEW;
  const yamlColors = useMemo(
    () => getJsonColors(theme.isDarkMode, theme.colors.textPrimary, theme.colors.textSecondary),
    [theme.isDarkMode, theme.colors.textPrimary, theme.colors.textSecondary],
  );

  const expandable =
    isContentLong ||
    dataToTruncate.split('\n').length > maxLinesForPreview ||
    dataToTruncate.length > maxCharsForPreview;

  // Truncate after first 3 lines if not expanded
  const displayedData = useMemo(() => {
    if (expandable && !expanded) {
      const split = dataToTruncate.split('\n').slice(0, maxLinesForPreview).join('\n');
      return split.length > maxCharsForPreview ? split.slice(0, maxCharsForPreview) : split;
    }

    return dataToTruncate;
  }, [dataToTruncate, expandable, expanded, maxLinesForPreview, maxCharsForPreview]);

  useEffect(() => {
    if (renderMode === CodeSnippetRenderMode.JSON && jsonViewerRef.current) {
      setIsContentLong(jsonViewerRef.current.scrollHeight > JSON_PREVIEW_HEIGHT);
      return;
    }

    if (snippetRef.current) {
      setIsContentLong(snippetRef.current.scrollWidth > snippetRef.current.clientWidth);
    }
  }, [renderMode, data]);

  // add a ref to the <pre> component within <CodeSnippet>.
  // we use the ref to check whether the <pre>'s content is overflowing
  const PreWithRef = useCallback((preProps: any) => <pre {...preProps} ref={snippetRef} />, []);

  if (containsMatches) {
    return (
      // if the snippet contains matches, render the search-highlighted version
      <ModelTraceExplorerHighlightedCodeSnippet
        data={searchableData}
        searchFilter={searchFilter}
        activeMatch={activeMatch}
        containsActiveMatch={!activeMatch.isKeyMatch && containsActiveMatch}
      />
    );
  }

  if (renderMode === 'json') {
    return (
      <div css={{ position: 'relative' }}>
        <SnippetCopyAction
          key="copy-snippet"
          componentId="shared.model-trace-explorer.copy-snippet"
          copyText={data}
          size="small"
          css={{ position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs, zIndex: 1 }}
        />
        <div
          ref={jsonViewerRef}
          css={{
            maxHeight: expandable && !expanded ? JSON_PREVIEW_HEIGHT : 'none',
            overflow: expandable && !expanded ? 'hidden' : 'visible',
            position: 'relative',
          }}
        >
          <CollapsibleJsonViewer data={data} initialExpanded />
          {expandable && !expanded && (
            <div
              css={{
                position: 'absolute',
                bottom: 0,
                left: 0,
                right: 0,
                height: '40px',
                pointerEvents: 'none',
              }}
            />
          )}
        </div>
        {expandable && (
          <div
            css={{
              backgroundColor: theme.colors.backgroundSecondary,
              borderTop: `1px solid ${theme.colors.border}`,
            }}
          >
            <Button
              css={{ width: '100%', padding: 0 }}
              componentId={
                expanded
                  ? 'shared.model-trace-explorer.snippet-see-less'
                  : 'shared.model-trace-explorer.snippet-see-more'
              }
              icon={expanded ? <ChevronUpIcon /> : <ChevronDownIcon />}
              type="tertiary"
              onClick={() => setExpanded(!expanded)}
            >
              {expanded ? (
                <FormattedMessage
                  defaultMessage="See less"
                  description="Model trace explorer > selected span > code snippet > see less button"
                />
              ) : (
                <FormattedMessage
                  defaultMessage="See more"
                  description="Model trace explorer > selected span > code snippet > see more button"
                />
              )}
            </Button>
          </div>
        )}
      </div>
    );
  }

  return (
    <div
      css={{
        position: 'relative',
        ...(isYamlRenderMode
          ? {
              'code, pre': {
                color: theme.colors.textPrimary,
              },
              '.token.atrule, .token.attr-name, .token.key, .token.property, .token.tag': {
                color: yamlColors.key,
              },
              '.token.boolean': {
                color: yamlColors.boolean,
              },
              '.token.number': {
                color: yamlColors.number,
              },
              '.token.null': {
                color: yamlColors.null,
              },
              '.token.punctuation': {
                color: yamlColors.punctuation,
              },
              '.token.scalar, .token.string': {
                color: yamlColors.string,
              },
            }
          : {}),
      }}
    >
      {renderMode === 'markdown' ? (
        <div
          css={{
            padding: theme.spacing.md,
            backgroundColor: theme.colors.backgroundSecondary,
            marginBottom: -theme.spacing.md,
          }}
        >
          <GenAIMarkdownRenderer>{displayedData}</GenAIMarkdownRenderer>
        </div>
      ) : (
        <>
          <SnippetCopyAction
            key="copy-snippet"
            componentId="shared.model-trace-explorer.copy-snippet"
            copyText={isYamlRenderMode ? dataToTruncate : data}
            size="small"
            css={{
              position: 'absolute',
              top: theme.spacing.xs,
              right: theme.spacing.xs,
              zIndex: 1,
            }}
          />
          <CodeSnippet
            PreTag={PreWithRef}
            showLineNumbers={!isYamlRenderMode}
            language={renderMode}
            useInlineStyles={!isYamlRenderMode}
            lineProps={{ style: { wordBreak: 'break-word', whiteSpace: 'pre-wrap' } }}
            wrapLines={expanded}
            theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
            style={{
              backgroundColor: theme.colors.backgroundSecondary,
              padding: theme.spacing.sm,
              paddingBottom: expandable ? 0 : theme.spacing.sm,
              paddingRight: theme.spacing.md * 2,
              overflow: expanded ? 'auto' : 'hidden',
              textOverflow: 'ellipsis',
              fontSize: theme.typography.fontSizeSm,
              lineHeight: theme.typography.lineHeightBase,
            }}
          >
            {displayedData}
          </CodeSnippet>
        </>
      )}
      {expandable && (
        <div css={{ backgroundColor: theme.colors.backgroundSecondary }}>
          <Button
            css={{ width: '100%', padding: 0 }}
            componentId={
              expanded ? 'shared.model-trace-explorer.snippet-see-less' : 'shared.model-trace-explorer.snippet-see-more'
            }
            icon={expanded ? <ChevronUpIcon /> : <ChevronDownIcon />}
            type="tertiary"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? (
              <FormattedMessage
                defaultMessage="See less"
                description="Model trace explorer > selected span > code snippet > see less button"
              />
            ) : (
              <FormattedMessage
                defaultMessage="See more"
                description="Model trace explorer > selected span > code snippet > see more button"
              />
            )}
          </Button>
        </div>
      )}
    </div>
  );
}
