import { useDesignSystemTheme } from '@databricks/design-system';

import { ExperimentSourceTypeIcon } from './ExperimentSourceTypeIcon';
import type { TraceInfoV3 } from '../../types';
import MlflowUtils from '../../utils/MlflowUtils';
import { NullCell } from '../NullCell';

export const SourceCellRenderer = (props: { traceInfo: TraceInfoV3; isComparing: boolean; disableLinks?: boolean }) => {
  const tags = props.traceInfo.tags;
  const { theme } = useDesignSystemTheme();

  if (!tags) {
    return <NullCell isComparing={props.isComparing} />;
  }

  const sourceType = props.traceInfo.trace_metadata?.[MlflowUtils.sourceTypeTag];

  const sourceLink = MlflowUtils.renderSourceFromMetadata(props.traceInfo);

  return sourceLink && sourceType ? (
    <div
      css={{
        display: 'flex',
        gap: theme.spacing.xs,
        alignItems: 'center',
        // Disable links if the disableLinks prop is true
        ...(props.disableLinks && {
          '& a': {
            pointerEvents: 'none',
            color: 'inherit',
            textDecoration: 'none',
            cursor: 'default',
          },
        }),
      }}
    >
      <ExperimentSourceTypeIcon sourceType={sourceType} css={{ color: theme.colors.textSecondary }} />
      <span css={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{sourceLink}</span>
    </div>
  ) : (
    <NullCell isComparing={props.isComparing} />
  );
};
