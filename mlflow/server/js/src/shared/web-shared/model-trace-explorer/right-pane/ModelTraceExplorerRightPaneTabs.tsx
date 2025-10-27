import type { Interpolation, Theme } from '@emotion/react';
import { isNil } from 'lodash';
import React, { useState } from 'react';

import { Empty, Tabs, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { ModelTraceExplorerAttributesTab } from './ModelTraceExplorerAttributesTab';
import { ModelTraceExplorerChatTab } from './ModelTraceExplorerChatTab';
import { ModelTraceExplorerContentTab } from './ModelTraceExplorerContentTab';
import { ModelTraceExplorerEventsTab } from './ModelTraceExplorerEventsTab';
import type { ModelTraceExplorerTab, ModelTraceSpanNode, SearchMatch } from '../ModelTrace.types';
import { getSpanExceptionCount } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerBadge } from '../ModelTraceExplorerBadge';
import ModelTraceExplorerResizablePane from '../ModelTraceExplorerResizablePane';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { AssessmentPaneToggle } from '../assessments-pane/AssessmentPaneToggle';
import { AssessmentsPane } from '../assessments-pane/AssessmentsPane';
import { ASSESSMENT_PANE_MIN_WIDTH } from '../assessments-pane/AssessmentsPane.utils';

export const CONTENT_PANE_MIN_WIDTH = 250;
// used by the parent component to set min-width on the resizable box
export const RIGHT_PANE_MIN_WIDTH = CONTENT_PANE_MIN_WIDTH + ASSESSMENT_PANE_MIN_WIDTH;
const DEFAULT_SPLIT_RATIO = 0.7;

function ModelTraceExplorerRightPaneTabsImpl({
  activeSpan,
  searchFilter,
  activeMatch,
  activeTab,
  setActiveTab,
}: {
  activeSpan: ModelTraceSpanNode | undefined;
  searchFilter: string;
  activeMatch: SearchMatch | null;
  activeTab: ModelTraceExplorerTab;
  setActiveTab: (tab: ModelTraceExplorerTab) => void;
}) {
  const { theme } = useDesignSystemTheme();
  const [paneWidth, setPaneWidth] = useState(500);
  const contentStyle: Interpolation<Theme> = { flex: 1, marginTop: -theme.spacing.md, overflowY: 'auto' };
  const { assessmentsPaneExpanded, assessmentsPaneEnabled } = useModelTraceExplorerViewState();

  if (isNil(activeSpan)) {
    return <Empty description="Please select a span to view more information" />;
  }

  const exceptionCount = getSpanExceptionCount(activeSpan);
  const hasException = exceptionCount > 0;
  const hasInputsOrOutputs = !isNil(activeSpan?.inputs) || !isNil(activeSpan?.outputs);

  const tabContent = (
    <Tabs.Root
      componentId="shared.model-trace-explorer.right-pane-tabs"
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        borderLeft: `1px solid ${theme.colors.border}`,
        minWidth: 200,
        position: 'relative',
      }}
      value={activeTab}
      onValueChange={(tab: string) => setActiveTab(tab as ModelTraceExplorerTab)}
    >
      <div
        css={{
          position: 'absolute',
          right: assessmentsPaneExpanded ? theme.spacing.xs : theme.spacing.md,
          top: theme.spacing.xs,
        }}
      >
        <AssessmentPaneToggle />
      </div>
      <Tabs.List
        css={{
          padding: 0,
          paddingLeft: theme.spacing.md,
          paddingRight: theme.spacing.sm,
          boxSizing: 'border-box',
          width: '100%',
        }}
      >
        {activeSpan.chatMessages && (
          <Tabs.Trigger value="chat">
            <FormattedMessage defaultMessage="Chat" description="Label for the chat tab of the model trace explorer." />
          </Tabs.Trigger>
        )}
        {hasInputsOrOutputs && (
          <Tabs.Trigger value="content">
            <FormattedMessage
              defaultMessage="Inputs / Outputs"
              description="Label for the inputs and outputs tab of the model trace explorer."
            />
          </Tabs.Trigger>
        )}
        {/* no i18n for attributes and events as these are properties specified by code,
            and it might be confusing for users to have different labels here */}
        <Tabs.Trigger value="attributes">Attributes</Tabs.Trigger>
        <Tabs.Trigger value="events">
          Events {hasException && <ModelTraceExplorerBadge count={exceptionCount} />}
        </Tabs.Trigger>
      </Tabs.List>
      {activeSpan.chatMessages && (
        <Tabs.Content css={contentStyle} value="chat">
          <ModelTraceExplorerChatTab chatMessages={activeSpan.chatMessages} chatTools={activeSpan.chatTools} />
        </Tabs.Content>
      )}
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
      <Tabs.Content css={contentStyle} value="events">
        <ModelTraceExplorerEventsTab activeSpan={activeSpan} searchFilter={searchFilter} activeMatch={activeMatch} />
      </Tabs.Content>
    </Tabs.Root>
  );

  return assessmentsPaneEnabled && assessmentsPaneExpanded ? (
    <ModelTraceExplorerResizablePane
      initialRatio={DEFAULT_SPLIT_RATIO}
      paneWidth={paneWidth}
      setPaneWidth={setPaneWidth}
      leftChild={tabContent}
      leftMinWidth={CONTENT_PANE_MIN_WIDTH}
      rightChild={
        <AssessmentsPane
          assessments={activeSpan.assessments}
          traceId={activeSpan.traceId}
          activeSpanId={activeSpan.parentId ? String(activeSpan.key) : undefined}
        />
      }
      rightMinWidth={ASSESSMENT_PANE_MIN_WIDTH}
    />
  ) : (
    tabContent
  );
}

export const ModelTraceExplorerRightPaneTabs = React.memo(ModelTraceExplorerRightPaneTabsImpl);
