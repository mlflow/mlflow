import { useMemo, useState } from 'react';

import { Button, ChevronRightIcon, ChevronDownIcon, useDesignSystemTheme, Typography } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { ModelTraceExplorerSummaryViewExceptionsSection } from './ModelTraceExplorerSummaryViewExceptionsSection';
import { type ModelTraceSpanNode } from '../ModelTrace.types';
import { CodeSnippetRenderMode } from '../ModelTrace.types';
import {
  createListFromObject,
  buildAggregatedJsonFromKeyValueList,
  getSpanExceptionEvents,
} from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { SpanNameDetailViewLink } from '../assessments-pane/SpanNameDetailViewLink';
import { ModelTraceExplorerFieldRenderer } from '../field-renderers/ModelTraceExplorerFieldRenderer';
import { ModelTraceExplorerCodeSnippet } from '../ModelTraceExplorerCodeSnippet';
import { spanTimeFormatter } from '../timeline-tree/TimelineTree.utils';

export const ModelTraceExplorerSummaryIntermediateNode = ({
  node,
  renderMode,
  className,
}: {
  node: ModelTraceSpanNode;
  renderMode: 'default' | 'json' | 'table';
  className?: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const [expanded, setExpanded] = useState(false);
  const inputList = useMemo(() => createListFromObject(node.inputs), [node]);
  const outputList = useMemo(() => createListFromObject(node.outputs), [node]);
  const aggregatedInputJson = useMemo(
    () => (inputList.length > 0 ? buildAggregatedJsonFromKeyValueList(inputList) : null),
    [inputList],
  );
  const aggregatedOutputJson = useMemo(
    () => (outputList.length > 0 ? buildAggregatedJsonFromKeyValueList(outputList) : null),
    [outputList],
  );
  const exceptionEvents = getSpanExceptionEvents(node);
  const chatMessageFormat = node.chatMessageFormat;

  const hasException = exceptionEvents.length > 0;
  const containsInputs = inputList.length > 0;
  const containsOutputs = outputList.length > 0;

  const { setSelectedNode, setActiveView, setShowTimelineTreeGantt } = useModelTraceExplorerViewState();

  return (
    <div
      className={className}
      css={{
        display: 'flex',
        flexDirection: 'row',
        flexShrink: 0,
        padding: theme.spacing.sm,
        paddingRight: theme.spacing.md,
      }}
    >
      <div css={{ display: 'flex', alignItems: 'flex-start' }}>
        <Button
          size="small"
          data-testid={`toggle-span-expanded-${node.key}`}
          css={{ flexShrink: 0, marginRight: theme.spacing.xs }}
          icon={expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
          onClick={() => setExpanded(!expanded)}
          componentId="shared.model-trace-explorer.toggle-span"
        />
      </div>
      <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0 }}>
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: expanded ? theme.spacing.sm : 0,
          }}
        >
          <Typography.Text color="secondary" css={{ display: 'inline-flex', alignItems: 'center' }}>
            <FormattedMessage
              defaultMessage="{spanName} was called"
              description="Label for an intermediate node in the trace explorer summary view, indicating that a span/function was called in the course of execution."
              values={{
                spanName: <SpanNameDetailViewLink node={node} />,
              }}
            />
          </Typography.Text>
          <span
            onClick={() => {
              setSelectedNode(node);
              setActiveView('detail');
              setShowTimelineTreeGantt(true);
            }}
          >
            <Typography.Text
              css={{
                '&:hover': {
                  textDecoration: 'underline',
                  cursor: 'pointer',
                },
              }}
              color="secondary"
            >
              {spanTimeFormatter(node.end - node.start)}
            </Typography.Text>
          </span>
        </div>
        {expanded && (
          <div>
            {hasException && <ModelTraceExplorerSummaryViewExceptionsSection node={node} />}
            {containsInputs && (
              <ModelTraceExplorerCollapsibleSection
                sectionKey="input"
                title={
                  <FormattedMessage
                    defaultMessage="Inputs"
                    description="Model trace explorer > selected span > inputs header"
                  />
                }
              >
                <div
                  css={{
                    marginBottom: theme.spacing.sm,
                  }}
                >
                  {renderMode === 'table' && aggregatedInputJson ? (
                    <ModelTraceExplorerCodeSnippet
                      title=""
                      data={aggregatedInputJson}
                      initialRenderMode={CodeSnippetRenderMode.TABLE}
                      hideRenderModeDropdown
                    />
                  ) : (
                    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                      {inputList.map(({ key, value }, index) => (
                        <ModelTraceExplorerFieldRenderer
                          key={key || index}
                          title={key}
                          data={value}
                          renderMode={renderMode}
                          chatMessageFormat={chatMessageFormat}
                        />
                      ))}
                    </div>
                  )}
                </div>
              </ModelTraceExplorerCollapsibleSection>
            )}
            {containsOutputs && (
              <ModelTraceExplorerCollapsibleSection
                sectionKey="output"
                title={
                  <FormattedMessage
                    defaultMessage="Outputs"
                    description="Model trace explorer > selected span > outputs header"
                  />
                }
              >
                <div
                  css={{
                    marginBottom: theme.spacing.sm,
                  }}
                >
                  {renderMode === 'table' && aggregatedOutputJson ? (
                    <ModelTraceExplorerCodeSnippet
                      title=""
                      data={aggregatedOutputJson}
                      initialRenderMode={CodeSnippetRenderMode.TABLE}
                      hideRenderModeDropdown
                    />
                  ) : (
                    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                      {outputList.map(({ key, value }) => (
                        <ModelTraceExplorerFieldRenderer
                          key={key}
                          title={key}
                          data={value}
                          renderMode={renderMode}
                          chatMessageFormat={chatMessageFormat}
                          assessments={node.assessments}
                        />
                      ))}
                    </div>
                  )}
                </div>
              </ModelTraceExplorerCollapsibleSection>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
