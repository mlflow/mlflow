import React from 'react';
import Utils from '../../../../../../common/utils/Utils';
import { RunRowType } from '../../../utils/experimentPage.row-types';

export const SourceCellRenderer = React.memo(({ value }: { value: RunRowType['tags'] }) => {
  const sourceType = Utils.renderSource(value, undefined, undefined);
  return sourceType ? (
    <>
      {Utils.renderSourceTypeIcon(value)}
      {sourceType}
    </>
  ) : (
    <>-</>
  );
});
