import { isNil, keys } from 'lodash';
import { useState } from 'react';

import {
  Button,
  ChevronDownIcon,
  DropdownMenu,
  Empty,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { CodeSnippetRenderMode, type ModelTraceSpanNode, type SearchMatch } from '../ModelTrace.types';
import { ModelTraceExplorerCodeSnippet } from '../ModelTraceExplorerCodeSnippet';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';
import { useModelTraceExplorerPreferences } from '../ModelTraceExplorerPreferencesContext';
import {
  areModelTraceComparisonValuesEqual,
  ModelTraceExplorerFieldRenderer,
} from '../field-renderers/ModelTraceExplorerFieldRenderer';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';

type AttributesRenderMode = 'pretty' | 'json' | 'yaml';

export const normalizeSpanAttributes = (attributes: ModelTraceSpanNode['attributes']): Record<string, unknown> => {
  if (!attributes) {
    return {};
  }
  if (!Array.isArray(attributes)) {
    return attributes;
  }
  return Object.fromEntries(
    attributes.map(({ key, value }) => [
      key,
      value.string_value ?? value.int_value ?? value.bool_value ?? value.double_value ?? value,
    ]),
  );
};

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
  const { selectedComparisonNode, selectedNodeComparisonRole: comparisonRole } = useModelTraceExplorerViewState();
  const [attributesRenderMode, setAttributesRenderMode] = useState<AttributesRenderMode>(
    renderMode === 'json' || renderMode === 'yaml' ? renderMode : 'pretty',
  );
  // A comparison flattens array-form attributes so both sides are keyed by attribute name.
  const attributes = (comparisonRole ? normalizeSpanAttributes(activeSpan.attributes) : activeSpan.attributes) as
    | Record<string, unknown>
    | undefined;
  const comparisonAttributes = normalizeSpanAttributes(selectedComparisonNode?.attributes);
  const containsAttributes = keys(attributes).length > 0 || keys(comparisonAttributes).length > 0;
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

  const effectiveRenderMode = comparisonRole || isActiveMatchSpan ? 'pretty' : attributesRenderMode;
  const baselineAttributeKeys = comparisonRole === 'candidate' ? keys(comparisonAttributes) : keys(attributes);
  const candidateAttributeKeys = comparisonRole === 'candidate' ? keys(attributes) : keys(comparisonAttributes);
  const attributeKeys = comparisonRole
    ? [...baselineAttributeKeys, ...candidateAttributeKeys.filter((key) => !baselineAttributeKeys.includes(key))]
    : keys(attributes);
  const sortedAttributeKeys = comparisonRole
    ? [...attributeKeys].sort(
        (firstKey, secondKey) =>
          Number(
            areModelTraceComparisonValuesEqual(
              JSON.stringify(attributes[firstKey], null, 2),
              JSON.stringify(comparisonAttributes[firstKey], null, 2),
            ),
          ) -
          Number(
            areModelTraceComparisonValuesEqual(
              JSON.stringify(attributes[secondKey], null, 2),
              JSON.stringify(comparisonAttributes[secondKey], null, 2),
            ),
          ),
      )
    : attributeKeys;
  const attributesHaveDifferences = comparisonRole
    ? sortedAttributeKeys.some(
        (key) =>
          !areModelTraceComparisonValuesEqual(
            JSON.stringify(attributes[key], null, 2),
            JSON.stringify(comparisonAttributes[key], null, 2),
          ),
      )
    : false;
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
        key={comparisonRole ? `${activeSpan.key}-attributes` : undefined}
        withBorder
        initialExpanded={!comparisonRole || attributesHaveDifferences}
        sectionKey="attributes"
        headerPadding={`${theme.spacing.xs}px 0`}
        contentPadding={`${theme.spacing.xs}px 0`}
        title={
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, width: '100%' }}>
            <FormattedMessage defaultMessage="Attributes" description="Title for the span attributes section" />
            {!comparisonRole && renderModeDropdown}
            {comparisonRole && !attributesHaveDifferences && (
              <Typography.Text data-tools-expand-hint color="secondary" size="sm" css={{ marginLeft: 'auto' }}>
                <FormattedMessage
                  defaultMessage="Click to Expand"
                  description="Hint next to an unchanged comparison section chevron that the section can be expanded"
                />
              </Typography.Text>
            )}
          </div>
        }
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          {effectiveRenderMode === 'pretty' ? (
            sortedAttributeKeys.map((key) => (
              <ModelTraceExplorerFieldRenderer
                key={key}
                title={key}
                data={JSON.stringify(attributes[key], null, 2)}
                comparisonData={JSON.stringify(comparisonAttributes[key], null, 2)}
                comparisonRole={comparisonRole}
                comparisonEnabled={Boolean(comparisonRole)}
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
