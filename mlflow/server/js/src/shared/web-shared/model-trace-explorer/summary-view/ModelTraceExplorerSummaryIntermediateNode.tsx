import { useMemo, useState } from 'react';

import { Button, ChevronRightIcon, ChevronDownIcon, useDesignSystemTheme, Typography } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { ModelTraceExplorerSummaryViewExceptionsSection } from './ModelTraceExplorerSummaryViewExceptionsSection';
import { type ModelTraceSpanNode } from '../ModelTrace.types';
import { createListFromObject, getSpanExceptionEvents } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { SpanNameDetailViewLink } from '../assessments-pane/SpanNameDetailViewLink';
import { ModelTraceExplorerFieldRenderer } from '../field-renderers/ModelTraceExplorerFieldRenderer';
import { spanTimeFormatter } from '../timeline-tree/TimelineTree.utils';

const CONNECTOR_WIDTH = 12;
const ROW_HEIGHT = 48;

export const ModelTraceExplorerSummaryIntermediateNode = ({
  node,
  renderMode,
}: {
  node: ModelTraceSpanNode;
  renderMode: 'default' | 'json';
}) => {
  const { theme } = useDesignSystemTheme();
  const [expanded, setExpanded] = useState(false);
  const inputList = useMemo(() => createListFromObject(node.inputs), [node]);
  const outputList = useMemo(() => createListFromObject(node.outputs), [node]);
  const exceptionEvents = getSpanExceptionEvents(node);
  const chatMessageFormat = node.chatMessageFormat;

  const hasException = exceptionEvents.length > 0;
  const containsInputs = inputList.length > 0;
  const containsOutputs = outputList.length > 0;

  const { setSelectedNode, setActiveView, setShowTimelineTreeGantt } = useModelTraceExplorerViewState();

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'row',
        minHeight: ROW_HEIGHT,
        flexShrink: 0,
      }}
    >
      <div css={{ height: ROW_HEIGHT, display: 'flex', alignItems: 'center' }}>
        <Button
          size="small"
          data-testid={`toggle-span-expanded-${node.key}`}
          css={{ flexShrink: 0, marginRight: theme.spacing.xs }}
          icon={expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
          onClick={() => setExpanded(!expanded)}
          componentId="shared.model-trace-explorer.toggle-span"
        />
      </div>
      <div
        css={{
          position: 'relative',
          boxSizing: 'border-box',
          height: '100%',
          borderLeft: `2px solid ${theme.colors.border}`,
          width: CONNECTOR_WIDTH,
        }}
      >
        <div
          css={{
            position: 'absolute',
            left: -2,
            top: 14,
            height: CONNECTOR_WIDTH,
            width: CONNECTOR_WIDTH,
            boxSizing: 'border-box',
            borderBottomLeftRadius: theme.borders.borderRadiusMd,
            borderBottom: `2px solid ${theme.colors.border}`,
            borderLeft: `2px solid ${theme.colors.border}`,
          }}
        />
      </div>
      <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0 }}>
        <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Typography.Text color="secondary" css={{ display: 'inline-flex', alignItems: 'center', height: ROW_HEIGHT }}>
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
                    display: 'flex',
                    flexDirection: 'column',
                    gap: theme.spacing.sm,
                    paddingLeft: theme.spacing.lg,
                    marginBottom: containsOutputs ? 0 : theme.spacing.sm,
                  }}
                >
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
                    display: 'flex',
                    flexDirection: 'column',
                    gap: theme.spacing.sm,
                    paddingLeft: theme.spacing.lg,
                    marginBottom: theme.spacing.sm,
                  }}
                >
                  {outputList.map(({ key, value }) => (
                    <ModelTraceExplorerFieldRenderer
                      key={key}
                      title={key}
                      data={value}
                      renderMode={renderMode}
                      chatMessageFormat={chatMessageFormat}
                    />
                  ))}
                </div>
              </ModelTraceExplorerCollapsibleSection>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
