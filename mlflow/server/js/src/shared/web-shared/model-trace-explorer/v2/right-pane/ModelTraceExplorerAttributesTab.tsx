import { isNil, keys } from 'lodash';
import { useState } from 'react';

import { Button, ChevronDownIcon, DropdownMenu, Empty, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { CodeSnippetRenderMode, type ModelTraceSpanNode, type SearchMatch } from '../ModelTrace.types';
import { ModelTraceExplorerCodeSnippet } from '../ModelTraceExplorerCodeSnippet';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';
import { useModelTraceExplorerPreferences } from '../ModelTraceExplorerPreferencesContext';
import { ModelTraceExplorerFieldRenderer } from '../field-renderers/ModelTraceExplorerFieldRenderer';

type AttributesRenderMode = 'pretty' | 'json' | 'yaml';

export function ModelTraceExplorerAttributesTab({
  activeSpan,
  searchFilter,
  activeMatch,
}: {
  activeSpan: ModelTraceSpanNode;
  searchFilter: string;
  activeMatch: SearchMatch | null;
}): React.ReactElement | null {
  const { theme } = useDesignSystemTheme();
  const { renderMode } = useModelTraceExplorerPreferences();
  const [attributesRenderMode, setAttributesRenderMode] = useState<AttributesRenderMode>(
    renderMode === 'json' || renderMode === 'yaml' ? renderMode : 'pretty',
  );
  const { attributes } = activeSpan;
  const containsAttributes = keys(attributes).length > 0;
  const isActiveMatchSpan = !isNil(activeMatch) && activeMatch.span.key === activeSpan.key;
  if (!containsAttributes || isNil(attributes)) {
    return (
      <div css={{ marginTop: theme.spacing.lg }}>
        <Empty
          description={
            <FormattedMessage
              defaultMessage="No attributes found"
              description="Empty state for the attributes tab in the model trace explorer. Attributes are properties of a span that the user defines."
            />
          }
        />
      </div>
    );
  }

  const effectiveRenderMode = isActiveMatchSpan ? 'pretty' : attributesRenderMode;

  const renderModeDropdown = (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <Button
          componentId="shared.model-trace-explorer.attributes.render-mode"
          type="tertiary"
          size="small"
          endIcon={<ChevronDownIcon />}
          css={{
            color: `${theme.colors.textSecondary} !important`,
            fontSize: theme.typography.fontSizeSm,
            paddingInline: theme.spacing.xs,
            '& > span, svg': {
              color: `${theme.colors.textSecondary} !important`,
            },
          }}
        >
          {attributesRenderMode === 'pretty' ? (
            <FormattedMessage
              defaultMessage="Pretty"
              description="Label for the pretty render mode in the model trace explorer attributes tab"
            />
          ) : attributesRenderMode === 'json' ? (
            <FormattedMessage
              defaultMessage="JSON"
              description="Label for the JSON render mode in the model trace explorer attributes tab"
            />
          ) : (
            <FormattedMessage
              defaultMessage="YAML"
              description="Label for the YAML render mode in the model trace explorer attributes tab"
            />
          )}
        </Button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Content align="end">
        <DropdownMenu.RadioGroup
          componentId="shared.model-trace-explorer.attributes.render-mode-radio"
          value={attributesRenderMode}
          onValueChange={(value) => {
            if (value === 'pretty' || value === 'json' || value === 'yaml') {
              setAttributesRenderMode(value);
            }
          }}
        >
          <DropdownMenu.RadioItem value="pretty">
            <DropdownMenu.ItemIndicator />
            <FormattedMessage
              defaultMessage="Pretty"
              description="Label for the pretty render mode dropdown item in the model trace explorer attributes tab"
            />
          </DropdownMenu.RadioItem>
          <DropdownMenu.RadioItem value="json">
            <DropdownMenu.ItemIndicator />
            <FormattedMessage
              defaultMessage="JSON"
              description="Label for the JSON render mode dropdown item in the model trace explorer attributes tab"
            />
          </DropdownMenu.RadioItem>
          <DropdownMenu.RadioItem value="yaml">
            <DropdownMenu.ItemIndicator />
            <FormattedMessage
              defaultMessage="YAML"
              description="Label for the YAML render mode dropdown item in the model trace explorer attributes tab"
            />
          </DropdownMenu.RadioItem>
        </DropdownMenu.RadioGroup>
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        paddingLeft: theme.spacing.md + theme.spacing.xs,
        paddingRight: theme.spacing.md + theme.spacing.xs,
        paddingTop: theme.spacing.sm,
      }}
    >
      <ModelTraceExplorerCollapsibleSection
        withBorder
        sectionKey="attributes"
        headerPadding={`${theme.spacing.xs}px 0`}
        contentPadding={`${theme.spacing.xs}px 0`}
        title={
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, width: '100%' }}>
            <FormattedMessage defaultMessage="Attributes" description="Title for the span attributes section" />
            {renderModeDropdown}
          </div>
        }
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          {effectiveRenderMode === 'pretty' ? (
            Object.entries(attributes).map(([key, value]) => (
              <ModelTraceExplorerFieldRenderer
                key={key}
                title={key}
                data={JSON.stringify(value, null, 2)}
                renderMode="default"
                searchFilter={searchFilter}
                activeMatch={activeMatch}
                containsActiveMatch={
                  isActiveMatchSpan && activeMatch.section === 'attributes' && activeMatch.key === key
                }
              />
            ))
          ) : (
            <ModelTraceExplorerCodeSnippet
              title=""
              data={JSON.stringify(attributes, null, 2)}
              searchFilter={searchFilter}
              activeMatch={activeMatch}
              containsActiveMatch={isActiveMatchSpan && activeMatch.section === 'attributes'}
              initialRenderMode={
                attributesRenderMode === 'yaml' ? CodeSnippetRenderMode.YAML : CodeSnippetRenderMode.JSON
              }
              hideRenderModeDropdown
            />
          )}
        </div>
      </ModelTraceExplorerCollapsibleSection>
    </div>
  );
}
