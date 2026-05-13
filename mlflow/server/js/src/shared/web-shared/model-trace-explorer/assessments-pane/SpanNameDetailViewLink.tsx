import { Typography, useDesignSystemTheme } from '@databricks/design-system';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { ModelSpanType } from '../ModelTrace.types';
import { getIconTypeForSpan, getSpanExceptionCount } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerIcon } from '../ModelTraceExplorerIcon';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';

export const SpanNameDetailViewLink = ({ node }: { node: ModelTraceSpanNode }) => {
  const { theme } = useDesignSystemTheme();
  const { setSelectedNode, setActiveView, setShowTimelineTreeGantt } = useModelTraceExplorerViewState();
  const hasException = getSpanExceptionCount(node) > 0;

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        marginRight: theme.spacing.xs,
        gap: theme.spacing.xs,
        '&:hover': {
          textDecoration: 'underline',
          textDecorationColor: hasException
            ? theme.colors.actionDangerDefaultTextDefault
            : theme.colors.actionDefaultTextDefault,
          cursor: 'pointer',
        },
      }}
      onClick={() => {
        setSelectedNode(node);
        setActiveView('detail');
        setShowTimelineTreeGantt(false);
      }}
    >
      <ModelTraceExplorerIcon
        hasException={hasException}
        type={getIconTypeForSpan(node.type ?? ModelSpanType.UNKNOWN)}
      />
      <Typography.Text
        color={hasException ? 'error' : 'primary'}
        css={{ marginLeft: theme.spacing.xs, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
      >
        {node.title}
      </Typography.Text>
    </div>
  );
};
