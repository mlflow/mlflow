import { Tooltip, useDesignSystemTheme, Typography } from '@databricks/design-system';

import type { AssessmentSource } from '../ModelTrace.types';

export const AssessmentSourceName = ({ source }: { source: AssessmentSource }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <Tooltip componentId="shared.model-trace-explorer.assessment-source-name" content={source.source_id}>
      {/* wrap in span so the tooltip can show up */}
      <span
        css={{
          flexShrink: 1,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          textWrap: 'nowrap',
          marginLeft: theme.spacing.sm,
          minWidth: theme.spacing.md,
        }}
      >
        <Typography.Text>
          <span css={{ color: theme.colors.blue500 }}>{source.source_id}</span>
        </Typography.Text>
      </span>
    </Tooltip>
  );
};
