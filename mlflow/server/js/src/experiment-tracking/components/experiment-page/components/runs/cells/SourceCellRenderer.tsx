import React from 'react';
import Utils from '../../../../../../common/utils/Utils';
import type { RunRowType } from '../../../utils/experimentPage.row-types';
import { useDesignSystemTheme } from '@databricks/design-system';
import { ExperimentSourceTypeIcon } from '../../../../ExperimentSourceTypeIcon';

export const SourceCellRenderer = React.memo(({ value: tags }: { value: RunRowType['tags'] }) => {
  const { theme } = useDesignSystemTheme();
  if (!tags) {
    return <>-</>;
  }
  const sourceType = tags[Utils.sourceTypeTag]?.value || '';

  const sourceLink = Utils.renderSource(tags || {}, undefined, undefined);
  return sourceLink ? (
    <div css={{ display: 'flex', gap: theme.spacing.xs, alignItems: 'center' }}>
      <ExperimentSourceTypeIcon sourceType={sourceType} css={{ color: theme.colors.textSecondary }} />
      <span css={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{sourceLink}</span>
    </div>
  ) : (
    <>-</>
  );
});
