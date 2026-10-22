import type { Interpolation, Theme } from '@emotion/react';
import { isNil } from 'lodash';
import React, { useCallback, useMemo, useState } from 'react';

import { Empty, Tabs, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import { ModelTraceExplorerAttributesTab, normalizeSpanAttributes } from './ModelTraceExplorerAttributesTab';
import { ModelTraceExplorerContentTab } from './ModelTraceExplorerContentTab';
import { getSpanComparisonFields } from './ModelTraceExplorerDefaultSpanView';
import { ModelTraceExplorerEventsTab } from './ModelTraceExplorerEventsTab';
import { ModelTraceExplorerLinksTab } from './ModelTraceExplorerLinksTab';
import { ModelTraceExplorerRightPaneHeader } from './ModelTraceExplorerRightPaneHeader';
import { SimplifiedAssessmentView } from './SimplifiedAssessmentView';
import type { ModelTrace, ModelTraceExplorerTab, ModelTraceSpanNode, SearchMatch } from '../ModelTrace.types';
import { getDefaultActiveTab, getSpanExceptionCount, getTraceLevelAssessments } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerBadge } from '../../ModelTraceExplorerBadge';
import ModelTraceExplorerResizablePane from '../ModelTraceExplorerResizablePane';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { AssessmentsPane } from '../assessments-pane/AssessmentsPane';
import { ASSESSMENT_PANE_MIN_WIDTH } from '../../assessments-pane/AssessmentsPane.utils';
import {
  getModelTraceComparisonLineCounts,
  type ModelTraceComparisonLineCounts,
} from '../field-renderers/ModelTraceExplorerFieldRenderer';

export const CONTENT_PANE_MIN_WIDTH = 250;
export const RIGHT_PANE_MIN_WIDTH: number = CONTENT_PANE_MIN_WIDTH + ASSESSMENT_PANE_MIN_WIDTH;

const stringifyComparisonValue = (value: unknown): string | undefined =>
  value === undefined ? undefined : JSON.stringify(value, null, 2);

const sumLineCounts = (counts: ModelTraceComparisonLineCounts[]): ModelTraceComparisonLineCounts =>
  counts.reduce<ModelTraceComparisonLineCounts>(
    (total, count) => ({ added: total.added + count.added, removed: total.removed + count.removed }),
    { added: 0, removed: 0 },
  );

// Counted field by field, the way the tabs show the diff, so a field missing on one side counts only its own lines
// (not the braces of the object around it).
const getFieldLineCounts = (
  baselineFields: { key: string; value: string }[],
  candidateFields: { key: string; value: string }[],
): ModelTraceComparisonLineCounts => {
  const baselineValues = new Map(baselineFields.map(({ key, value }) => [key, value]));
  const candidateValues = new Map(candidateFields.map(({ key, value }) => [key, value]));
  const keys = new Set([...baselineValues.keys(), ...candidateValues.keys()]);
  return sumLineCounts(
    [...keys].map((key) => getModelTraceComparisonLineCounts(baselineValues.get(key), candidateValues.get(key))),
  );
};

const getAttributeFields = (attributes: ModelTraceSpanNode['attributes']) =>
  Object.entries(normalizeSpanAttributes(attributes)).map(([key, value]) => ({
    key,
    value: JSON.stringify(value, null, 2),
  }));

const getComparisonLineCounts = (baseline: unknown, candidate: unknown): ModelTraceComparisonLineCounts =>
  getModelTraceComparisonLineCounts(stringifyComparisonValue(baseline), stringifyComparisonValue(candidate));

function ModelTraceExplorerRightPaneTabsImpl({
  activeSpan,
  modelTraceInfo,
  searchFilter,
  activeMatch,
  activeTab,
  setActiveTab,
  onSelectSpan,
}: {
  activeSpan: ModelTraceSpanNode | undefined;
  modelTraceInfo?: ModelTrace['info'];
  searchFilter: string;
  activeMatch: SearchMatch | null;
  activeTab: ModelTraceExplorerTab;
  setActiveTab: (tab: ModelTraceExplorerTab) => void;
  onSelectSpan?: (spanId: string) => void;
}): JSX.Element {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const {
    assessmentsPaneExpanded,
    assessmentsPaneEnabled,
    updatePaneSizeRatios,
    getPaneSizeRatios,
    readOnly: displayReadOnlyAssessments,
    isTraceInitialLoading,
    subscribeToHighlightEvent,
    nodeMap,
    selectedComparisonNode,
    selectedNodeComparisonRole,
  } = useModelTraceExplorerViewState();
  const [paneWidth, setPaneWidth] = useState(500);
  const rightPaneHorizontalPadding = theme.spacing.md + theme.spacing.xs;
  const contentStyle: Interpolation<Theme> = {
    flex: 1,
    minHeight: 0,
    marginTop: -theme.spacing.md,
    overflowY: 'auto',
  };

  const displayedAssessments = useMemo(
    () => getTraceLevelAssessments(activeSpan?.assessments),
    [activeSpan?.assessments],
  );

  const onSizeRatioChange = useCallback(
    (ratio: number) => {
      updatePaneSizeRatios({ detailsSidebar: ratio });
    },
    [updatePaneSizeRatios],
  );

  if (isNil(activeSpan)) {
    return <Empty description="Please select a span to view more information" />;
  }

  const exceptionCount = getSpanExceptionCount(activeSpan);
  const hasException = exceptionCount > 0;
  const hasComparisonEvents = (selectedComparisonNode?.events?.length ?? 0) > 0;
  const hasComparisonException = selectedComparisonNode ? getSpanExceptionCount(selectedComparisonNode) > 0 : false;
  const hasEvents =
    (activeSpan.events?.length ?? 0) > 0 || hasException || hasComparisonEvents || hasComparisonException;
  const linkCount = activeSpan.links?.length ?? 0;
  const hasLinks = linkCount > 0;
  const hasComparisonContent = selectedComparisonNode
    ? getDefaultActiveTab({ ...selectedComparisonNode, events: [] }) === 'content'
    : false;
  const hasContent = getDefaultActiveTab({ ...activeSpan, events: [] }) === 'content' || hasComparisonContent;
  const baselineSpan = selectedNodeComparisonRole === 'candidate' ? selectedComparisonNode : activeSpan;
  const candidateSpan = selectedNodeComparisonRole === 'candidate' ? activeSpan : selectedComparisonNode;
  const baselineFields = selectedNodeComparisonRole ? getSpanComparisonFields(baselineSpan) : undefined;
  const candidateFields = selectedNodeComparisonRole ? getSpanComparisonFields(candidateSpan) : undefined;
  const contentLineCounts =
    baselineFields && candidateFields
      ? sumLineCounts([
          getFieldLineCounts(baselineFields.inputs, candidateFields.inputs),
          getFieldLineCounts(baselineFields.outputs, candidateFields.outputs),
        ])
      : undefined;
  const attributeLineCounts = selectedNodeComparisonRole
    ? getFieldLineCounts(getAttributeFields(baselineSpan?.attributes), getAttributeFields(candidateSpan?.attributes))
    : undefined;
  const eventLineCounts = selectedNodeComparisonRole
    ? getComparisonLineCounts(baselineSpan?.events, candidateSpan?.events)
    : undefined;
  const renderLineCounts = (counts: ModelTraceComparisonLineCounts | undefined) => {
    if (!counts || (counts.added === 0 && counts.removed === 0)) {
      return null;
    }
    return (
      <span
        aria-label={intl.formatMessage(
          {
            defaultMessage: '{removed} removed lines and {added} added lines',
            description: 'Accessible label for line difference counts on a trace comparison tab',
          },
          { removed: counts.removed, added: counts.added },
        )}
        css={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: theme.spacing.xs,
          marginLeft: theme.spacing.xs,
          fontSize: theme.typography.fontSizeSm - 1,
          fontWeight: theme.typography.typographyRegularFontWeight,
          fontVariantNumeric: 'tabular-nums',
        }}
      >
        {counts.removed > 0 && <span css={{ color: theme.colors.textValidationDanger }}>−{counts.removed}</span>}
        {counts.added > 0 && <span css={{ color: theme.colors.textValidationSuccess }}>+{counts.added}</span>}
      </span>
    );
  };

  const tabContent = (
    <Tabs.Root
      componentId="shared.model-trace-explorer.right-pane-tabs"
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        borderLeft: `1px solid ${theme.colors.border}`,
        minWidth: 200,
        minHeight: 0,
        position: 'relative',
      }}
      value={activeTab}
      onValueChange={(tab: string) => setActiveTab(tab as ModelTraceExplorerTab)}
    >
      <ModelTraceExplorerRightPaneHeader
        activeSpan={activeSpan}
        modelTraceInfo={modelTraceInfo}
        showAssessmentsToggle={!displayReadOnlyAssessments && !assessmentsPaneExpanded}
      />
      <Tabs.List
        tabListCss={{ borderBottom: `0.5px solid ${theme.colors.border}` }}
        css={{
          padding: 0,
          paddingLeft: rightPaneHorizontalPadding,
          paddingRight: rightPaneHorizontalPadding,
          boxSizing: 'border-box',
          width: '100%',
        }}
        dangerouslyAppendEmotionCSS={{
          '&>div': { flex: 1 },
        }}
      >
        {hasContent && (
          <Tabs.Trigger value="content">
            <FormattedMessage
              defaultMessage="Inputs / Outputs"
              description="Label for the inputs and outputs tab of the model trace explorer."
            />
            {renderLineCounts(contentLineCounts)}
          </Tabs.Trigger>
        )}
        <Tabs.Trigger value="attributes">
          {selectedNodeComparisonRole ? (
            <>
              <FormattedMessage defaultMessage="Attributes" description="Label for the span attributes tab." />
              {renderLineCounts(attributeLineCounts)}
            </>
          ) : (
            'Attributes'
          )}
        </Tabs.Trigger>
        {hasEvents && (
          <Tabs.Trigger value="events">
            {selectedNodeComparisonRole ? (
              <>
                <FormattedMessage defaultMessage="Events" description="Label for the span events tab." />
                {renderLineCounts(eventLineCounts)}
                {hasException && <ModelTraceExplorerBadge count={exceptionCount} />}
              </>
            ) : (
              <>Events {hasException && <ModelTraceExplorerBadge count={exceptionCount} />}</>
            )}
          </Tabs.Trigger>
        )}
        {hasLinks && (
          <Tabs.Trigger value="links">
            Links <ModelTraceExplorerBadge count={linkCount} variant="neutral" />
          </Tabs.Trigger>
        )}
        {displayReadOnlyAssessments && (
          <Tabs.Trigger value="assessments">
            <FormattedMessage
              defaultMessage="Assessments"
              description="Label for the read-only assessments tab of the model trace explorer."
            />
          </Tabs.Trigger>
        )}
      </Tabs.List>
      <Tabs.Content css={contentStyle} value="content">
        <ModelTraceExplorerContentTab activeSpan={activeSpan} searchFilter={searchFilter} activeMatch={activeMatch} />
      </Tabs.Content>
      <Tabs.Content css={contentStyle} value="attributes">
        <ModelTraceExplorerAttributesTab
          activeSpan={activeSpan}
          searchFilter={searchFilter}
          activeMatch={activeMatch}
        />
      </Tabs.Content>
      {hasEvents && (
        <Tabs.Content css={contentStyle} value="events">
          <ModelTraceExplorerEventsTab activeSpan={activeSpan} searchFilter={searchFilter} activeMatch={activeMatch} />
        </Tabs.Content>
      )}
      {hasLinks && (
        <Tabs.Content css={contentStyle} value="links">
          <ModelTraceExplorerLinksTab
            activeSpan={activeSpan}
            searchFilter={searchFilter}
            activeMatch={activeMatch}
            onSelectSpan={onSelectSpan}
            spanNodes={nodeMap}
          />
        </Tabs.Content>
      )}
      {displayReadOnlyAssessments && (
        <Tabs.Content css={contentStyle} value="assessments">
          <SimplifiedAssessmentView
            assessments={getTraceLevelAssessments(activeSpan.assessments)}
            css={{ height: 'auto', borderLeft: 0 }}
          />
        </Tabs.Content>
      )}
    </Tabs.Root>
  );

  const AssessmentsPaneComponent = (
    <AssessmentsPane
      key={activeSpan.key}
      assessments={displayedAssessments}
      traceId={activeSpan.traceId}
      activeSpanId={activeSpan.parentId ? String(activeSpan.key) : undefined}
    />
  );

  return assessmentsPaneEnabled && assessmentsPaneExpanded ? (
    <ModelTraceExplorerResizablePane
      initialRatio={getPaneSizeRatios().detailsSidebar}
      paneWidth={paneWidth}
      setPaneWidth={setPaneWidth}
      leftChild={tabContent}
      leftMinWidth={CONTENT_PANE_MIN_WIDTH}
      rightChild={AssessmentsPaneComponent}
      rightMinWidth={ASSESSMENT_PANE_MIN_WIDTH + 2 * theme.spacing.sm}
      onRatioChange={onSizeRatioChange}
    />
  ) : (
    tabContent
  );
}

export const ModelTraceExplorerRightPaneTabs: React.NamedExoticComponent<
  React.ComponentProps<typeof ModelTraceExplorerRightPaneTabsImpl>
> = React.memo(ModelTraceExplorerRightPaneTabsImpl);
