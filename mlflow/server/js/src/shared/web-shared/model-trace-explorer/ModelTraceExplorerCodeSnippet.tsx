import { isString } from 'lodash';
import { useEffect, useMemo, useState } from 'react';

import { ChevronDownIcon, DropdownMenu, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';

import type { SearchMatch } from './ModelTrace.types';
import { CodeSnippetRenderMode } from './ModelTrace.types';
import { ModelTraceExplorerCodeSnippetBody } from './ModelTraceExplorerCodeSnippetBody';
import { ModelTraceExplorerHighlightedSnippetTitle } from './ModelTraceExplorerHighlightedSnippetTitle';

// return the initial render mode if specified, otherwise
// default to markdown for string data and json for non-string data
function getInitialRenderMode(dataIsString: boolean, initialRenderMode?: CodeSnippetRenderMode) {
  if (initialRenderMode) {
    return initialRenderMode;
  }

  if (dataIsString) {
    return CodeSnippetRenderMode.MARKDOWN;
  }

  return CodeSnippetRenderMode.JSON;
}

function getRenderModeDisplayText(renderMode: CodeSnippetRenderMode) {
  switch (renderMode) {
    case CodeSnippetRenderMode.JSON:
      return 'JSON';
    case CodeSnippetRenderMode.TEXT:
      return 'Text';
    case CodeSnippetRenderMode.MARKDOWN:
      return 'Markdown';
    case CodeSnippetRenderMode.PYTHON:
      return 'Python';
  }
}

export function ModelTraceExplorerCodeSnippet({
  title,
  tokens,
  data,
  searchFilter = '',
  activeMatch = null,
  containsActiveMatch = false,
  initialRenderMode,
}: {
  title: string;
  tokens?: number;
  data: string;
  searchFilter?: string;
  // the current active search match
  activeMatch?: SearchMatch | null;
  // whether the snippet being rendered contains the
  // current active match (either in the key or value)
  containsActiveMatch?: boolean;
  initialRenderMode?: CodeSnippetRenderMode;
}) {
  const parsedData = useMemo(() => JSON.parse(data), [data]);
  const dataIsString = isString(parsedData);
  const { theme } = useDesignSystemTheme();
  // string data can be rendered in multiple formats
  const [renderMode, setRenderMode] = useState<CodeSnippetRenderMode>(
    getInitialRenderMode(dataIsString, initialRenderMode),
  );
  const isTitleMatch = containsActiveMatch && (activeMatch?.isKeyMatch ?? false);
  const shouldShowRenderModeDropdown = dataIsString && !searchFilter;

  // we need to reset the render mode when the data changes
  useEffect(() => {
    setRenderMode(getInitialRenderMode(dataIsString, initialRenderMode));
  }, [dataIsString, initialRenderMode]);

  return (
    <div
      css={{
        position: 'relative',
      }}
    >
      <div
        css={{
          borderRadius: theme.borders.borderRadiusSm,
          border: `1px solid ${theme.colors.border}`,
          overflow: 'hidden',
        }}
      >
        {(title || shouldShowRenderModeDropdown) && (
          <div
            css={{
              display: 'flex',
              flexDirection: 'row',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: theme.spacing.sm,
            }}
          >
            {/* TODO: support other types of formatting, e.g. markdown */}
            <Typography.Title
              css={{
                maxWidth: '100%',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
              level={4}
              color="secondary"
              withoutMargins
            >
              <ModelTraceExplorerHighlightedSnippetTitle
                title={title}
                searchFilter={searchFilter}
                isActiveMatch={isTitleMatch}
              />
            </Typography.Title>
            <div css={{ display: 'flex', flexDirection: 'row', alignItems: 'center' }}>
              {shouldShowRenderModeDropdown && (
                <DropdownMenu.Root>
                  <DropdownMenu.Trigger asChild>
                    <Tag
                      componentId="shared.model-trace-explorer.snippet-render-mode-tag"
                      css={{
                        height: 'min-content',
                        margin: 0,
                      }}
                    >
                      {/* for some reason `cursor: pointer` doesn't work if you set it on the Tag css */}
                      <div css={{ paddingLeft: theme.spacing.xs, marginRight: theme.spacing.xs, cursor: 'pointer' }}>
                        <Typography.Text size="sm" color="secondary">
                          {getRenderModeDisplayText(renderMode)}
                        </Typography.Text>
                        <ChevronDownIcon />
                      </div>
                    </Tag>
                  </DropdownMenu.Trigger>
                  <DropdownMenu.Content>
                    <DropdownMenu.RadioGroup
                      componentId="shared.model-trace-explorer.snippet-render-mode-radio"
                      value={renderMode}
                      onValueChange={(value) => setRenderMode(value as CodeSnippetRenderMode)}
                    >
                      {Object.values(CodeSnippetRenderMode).map((mode) => {
                        if (mode === CodeSnippetRenderMode.PYTHON) {
                          return null;
                        }
                        return (
                          <DropdownMenu.RadioItem key={mode} value={mode}>
                            <DropdownMenu.ItemIndicator />
                            {getRenderModeDisplayText(mode)}
                          </DropdownMenu.RadioItem>
                        );
                      })}
                    </DropdownMenu.RadioGroup>
                    <DropdownMenu.Arrow />
                  </DropdownMenu.Content>
                </DropdownMenu.Root>
              )}
            </div>
          </div>
        )}
        <ModelTraceExplorerCodeSnippetBody
          data={data}
          searchFilter={searchFilter}
          activeMatch={activeMatch}
          containsActiveMatch={containsActiveMatch}
          renderMode={renderMode}
        />
      </div>
    </div>
  );
}
