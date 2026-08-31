import type { Interpolation, Theme } from '@emotion/react';
import { isNil } from 'lodash';
import React, { useCallback, useMemo, useState } from 'react';

import { Empty, Tabs, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { ModelTraceExplorerAttributesTab } from './ModelTraceExplorerAttributesTab';
import { ModelTraceExplorerContentTab } from './ModelTraceExplorerContentTab';
import { ModelTraceExplorerEventsTab } from './ModelTraceExplorerEventsTab';
import { ModelTraceExplorerRightPaneHeader } from './ModelTraceExplorerRightPaneHeader';
import { SimplifiedAssessmentView } from './SimplifiedAssessmentView';
import type { ModelTrace, ModelTraceExplorerTab, ModelTraceSpanNode, SearchMatch } from '../ModelTrace.types';
import { getDefaultActiveTab, getSpanExceptionCount, getTraceLevelAssessments } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerBadge } from '../../ModelTraceExplorerBadge';
import ModelTraceExplorerResizablePane from '../ModelTraceExplorerResizablePane';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { AssessmentsPane } from '../assessments-pane/AssessmentsPane';
import { ASSESSMENT_PANE_MIN_WIDTH } from '../../assessments-pane/AssessmentsPane.utils';

export const CONTENT_PANE_MIN_WIDTH = 250;
export const RIGHT_PANE_MIN_WIDTH: number = CONTENT_PANE_MIN_WIDTH + ASSESSMENT_PANE_MIN_WIDTH;

function ModelTraceExplorerRightPaneTabsImpl({
  activeSpan,
  modelTraceInfo,
  searchFilter,
  activeMatch,
  activeTab,
  setActiveTab,
}: {
  activeSpan: ModelTraceSpanNode | undefined;
  modelTraceInfo?: ModelTrace['info'];
  searchFilter: string;
  activeMatch: SearchMatch | null;
  activeTab: ModelTraceExplorerTab;
  setActiveTab: (tab: ModelTraceExplorerTab) => void;
}): JSX.Element {
  const { theme } = useDesignSystemTheme();
  const {
    assessmentsPaneExpanded,
    assessmentsPaneEnabled,
    updatePaneSizeRatios,
    getPaneSizeRatios,
    readOnly: displayReadOnlyAssessments,
    isTraceInitialLoading,
    subscribeToHighlightEvent,
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
  const hasEvents = (activeSpan.events?.length ?? 0) > 0 || hasException;
  const hasContent = getDefaultActiveTab({ ...activeSpan, events: [] }) === 'content';

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
          </Tabs.Trigger>
        )}
        <Tabs.Trigger value="attributes">Attributes</Tabs.Trigger>
        {hasEvents && (
          <Tabs.Trigger value="events">
            Events {hasException && <ModelTraceExplorerBadge count={exceptionCount} />}
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
