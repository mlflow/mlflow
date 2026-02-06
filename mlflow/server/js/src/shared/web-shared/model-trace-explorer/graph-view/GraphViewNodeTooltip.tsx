import { useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { ModelSpanType } from '../ModelTrace.types';
import { getDisplayNameForSpanType, getIconTypeForSpan } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerIcon } from '../ModelTraceExplorerIcon';
import { spanTimeFormatter } from '../timeline-tree/TimelineTree.utils';

interface GraphViewNodeTooltipProps {
  span: ModelTraceSpanNode;
  onViewDetails?: () => void;
}

/**
 * Tooltip content for a graph node showing span details.
 * Based on TimelineTreeSpanTooltip pattern.
 */
export const GraphViewNodeTooltip = ({ span, onViewDetails }: GraphViewNodeTooltipProps) => {
  const { theme } = useDesignSystemTheme();
  const iconType = getIconTypeForSpan(span.type ?? ModelSpanType.UNKNOWN);
  const spanTypeName = getDisplayNameForSpanType(span.type ?? ModelSpanType.UNKNOWN);
  const duration = spanTimeFormatter(span.end - span.start);

  const tooltipTextPrimary = theme.colors.textPrimary;
  const tooltipTextSecondary = theme.colors.textSecondary;

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.xs,
        maxWidth: 350,
        // Override tooltip container styling to match current theme
        backgroundColor: theme.colors.backgroundPrimary,
        color: tooltipTextPrimary,
        padding: theme.spacing.sm,
        borderRadius: theme.borders.borderRadiusMd,
        boxShadow: theme.shadows.lg,
        margin: -theme.spacing.sm, // Offset the tooltip's internal padding
      }}
    >
      {/* Header with icon and name */}
      <div
        css={{
          display: 'flex',
          flexDirection: 'row',
          gap: theme.spacing.xs,
          alignItems: 'center',
          overflow: 'hidden',
        }}
      >
        <ModelTraceExplorerIcon type={iconType} isInTooltip />
        <span
          css={{
            color: tooltipTextPrimary,
            fontWeight: theme.typography.typographyBoldFontWeight,
            wordBreak: 'break-word',
          }}
        >
          {span.title}
        </span>
      </div>

      {/* Type and duration row */}
      <div
        css={{
          display: 'flex',
          flexDirection: 'row',
          gap: theme.spacing.sm,
          color: tooltipTextSecondary,
          fontSize: theme.typography.fontSizeSm,
        }}
      >
        <span>
          <FormattedMessage
            defaultMessage="Type: {spanType}"
            description="Label showing span type in graph view tooltip"
            values={{ spanType: spanTypeName }}
          />
        </span>
        <span>|</span>
        <span>
          <FormattedMessage
            defaultMessage="Duration: {duration}"
            description="Label showing span duration in graph view tooltip"
            values={{ duration }}
          />
        </span>
      </div>

      {/* Clickable link to view details */}
      <div
        onClick={(e) => {
          e.stopPropagation();
          onViewDetails?.();
        }}
        css={{
          color: theme.colors.actionPrimaryBackgroundDefault,
          fontSize: theme.typography.fontSizeSm,
          fontStyle: 'italic',
          marginTop: theme.spacing.xs,
          cursor: 'pointer',
          '&:hover': {
            textDecoration: 'underline',
          },
        }}
      >
        <FormattedMessage
          defaultMessage="Click to view details"
          description="Clickable link in graph view node tooltip to open sidebar"
        />
      </div>
    </div>
  );
};
