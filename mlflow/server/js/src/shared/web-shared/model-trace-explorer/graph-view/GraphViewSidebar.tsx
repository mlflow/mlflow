import { useMemo } from 'react';
import { keys, isNil } from 'lodash';

import { Button, CloseIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { ModelSpanType } from '../ModelTrace.types';
import { createListFromObject, getDisplayNameForSpanType, getIconTypeForSpan } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerIcon } from '../ModelTraceExplorerIcon';
import { ModelTraceExplorerCodeSnippet } from '../ModelTraceExplorerCodeSnippet';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';
import { spanTimeFormatter } from '../timeline-tree/TimelineTree.utils';
import { getNodeBackgroundColor } from './GraphView.utils';

interface GraphViewSidebarProps {
  span: ModelTraceSpanNode;
  onClose: () => void;
}

/**
 * Sidebar component that displays span details when a node is selected in the graph view.
 * Shows metadata, inputs, outputs, and attributes in collapsible sections.
 */
export const GraphViewSidebar = ({ span, onClose }: GraphViewSidebarProps) => {
  const { theme } = useDesignSystemTheme();
  const spanType = span.type as ModelSpanType | undefined;
  const iconType = getIconTypeForSpan(spanType ?? 'UNKNOWN');
  const spanTypeName = getDisplayNameForSpanType(spanType ?? ModelSpanType.UNKNOWN);
  const duration = spanTimeFormatter(span.end - span.start);
  const headerBgColor = getNodeBackgroundColor(spanType, theme);

  const inputList = useMemo(() => createListFromObject(span.inputs), [span]);
  const outputList = useMemo(() => createListFromObject(span.outputs), [span]);
  const containsInputs = inputList.length > 0;
  const containsOutputs = outputList.length > 0;
  const containsAttributes = keys(span.attributes).length > 0;

  // Determine status based on span state
  const getSpanStatus = (): string => {
    if (span.events?.some((e) => e.name === 'exception')) {
      return 'Error';
    }
    return 'OK';
  };

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        width: 400,
        borderLeft: `1px solid ${theme.colors.border}`,
        backgroundColor: theme.colors.backgroundPrimary,
        overflow: 'hidden',
        height: '100%',
      }}
    >
      {/* Header with span name, type, and close button */}
      <div
        css={{
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: theme.spacing.sm,
          backgroundColor: headerBgColor,
          borderBottom: `1px solid ${theme.colors.border}`,
          flexShrink: 0,
        }}
      >
        <div
          css={{
            display: 'flex',
            flexDirection: 'row',
            alignItems: 'center',
            gap: theme.spacing.sm,
            overflow: 'hidden',
            flex: 1,
          }}
        >
          <ModelTraceExplorerIcon type={iconType} />
          <Typography.Text
            css={{
              fontWeight: theme.typography.typographyBoldFontWeight,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {span.title}
          </Typography.Text>
          <Typography.Text
            size="sm"
            color="secondary"
            css={{
              backgroundColor: theme.colors.tagDefault,
              padding: `${theme.spacing.xs / 2}px ${theme.spacing.xs}px`,
              borderRadius: theme.borders.borderRadiusSm,
              flexShrink: 0,
            }}
          >
            {spanTypeName}
          </Typography.Text>
        </div>
        <Button
          componentId="shared.model-trace-explorer.graph-view-sidebar-close"
          size="small"
          icon={<CloseIcon />}
          onClick={onClose}
          aria-label="Close sidebar"
          type="tertiary"
        />
      </div>

      {/* Scrollable content area */}
      <div
        css={{
          flex: 1,
          overflowY: 'auto',
          padding: theme.spacing.md,
        }}
      >
        {/* Metadata section */}
        <ModelTraceExplorerCollapsibleSection withBorder sectionKey="metadata" title="Metadata">
          <div
            css={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: theme.spacing.sm,
            }}
          >
            <MetadataItem
              label={
                <FormattedMessage defaultMessage="Duration" description="Label for span duration in sidebar metadata" />
              }
              value={duration}
            />
            <MetadataItem
              label={<FormattedMessage defaultMessage="Type" description="Label for span type in sidebar metadata" />}
              value={spanTypeName}
            />
            <MetadataItem
              label={
                <FormattedMessage defaultMessage="Status" description="Label for span status in sidebar metadata" />
              }
              value={getSpanStatus()}
            />
            <MetadataItem
              label={<FormattedMessage defaultMessage="Span ID" description="Label for span ID in sidebar metadata" />}
              value={String(span.key)}
              truncate
            />
          </div>
        </ModelTraceExplorerCollapsibleSection>

        {/* Inputs section */}
        {containsInputs && (
          <ModelTraceExplorerCollapsibleSection
            withBorder
            css={{ marginTop: theme.spacing.sm }}
            sectionKey="inputs"
            title={
              <FormattedMessage
                defaultMessage="Inputs"
                description="Model trace explorer > graph view sidebar > inputs header"
              />
            }
          >
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
              {inputList.map(({ key, value }, index) => (
                <ModelTraceExplorerCodeSnippet key={key || index} title={key} data={value} />
              ))}
            </div>
          </ModelTraceExplorerCollapsibleSection>
        )}

        {/* Outputs section */}
        {containsOutputs && (
          <ModelTraceExplorerCollapsibleSection
            withBorder
            css={{ marginTop: theme.spacing.sm }}
            sectionKey="outputs"
            title={
              <FormattedMessage
                defaultMessage="Outputs"
                description="Model trace explorer > graph view sidebar > outputs header"
              />
            }
          >
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
              {outputList.map(({ key, value }) => (
                <ModelTraceExplorerCodeSnippet key={key} title={key} data={value} />
              ))}
            </div>
          </ModelTraceExplorerCollapsibleSection>
        )}

        {/* Attributes section */}
        {containsAttributes && !isNil(span.attributes) && (
          <ModelTraceExplorerCollapsibleSection
            withBorder
            css={{ marginTop: theme.spacing.sm }}
            sectionKey="attributes"
            title="Attributes"
          >
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
              {Object.entries(span.attributes).map(([key, value]) => (
                <ModelTraceExplorerCodeSnippet key={key} title={key} data={JSON.stringify(value, null, 2)} />
              ))}
            </div>
          </ModelTraceExplorerCollapsibleSection>
        )}
      </div>
    </div>
  );
};

/**
 * Individual metadata item component for the metadata grid.
 */
const MetadataItem = ({
  label,
  value,
  truncate = false,
}: {
  label: React.ReactNode;
  value: string;
  truncate?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.xs / 2,
      }}
    >
      <Typography.Text size="sm" color="secondary">
        {label}
      </Typography.Text>
      <Typography.Text
        css={{
          fontWeight: theme.typography.typographyBoldFontWeight,
          ...(truncate && {
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            maxWidth: '100%',
          }),
        }}
        title={truncate ? value : undefined}
      >
        {value}
      </Typography.Text>
    </div>
  );
};
