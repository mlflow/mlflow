import { BranchIcon, Tag, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import Utils from '../../../../common/utils/Utils';
import type { KeyValueEntity } from '../../../types';
import { MLFLOW_RUN_GIT_SOURCE_BRANCH_TAG } from '../../../constants';

export const RunViewSourceBox = ({
  runUuid,
  tags,
  search,
}: {
  runUuid: string;
  tags: Record<string, KeyValueEntity>;
  search: string;
}) => {
  const branchName = tags?.[MLFLOW_RUN_GIT_SOURCE_BRANCH_TAG]?.value;
  const runSource = Utils.renderSource(tags, search, runUuid, branchName);

  const { theme } = useDesignSystemTheme();
  return runSource ? (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.sm,
        paddingTop: theme.spacing.sm,
        paddingBottom: theme.spacing.sm,
        flexWrap: 'wrap',
      }}
    >
      <span css={{ color: theme.colors.primary }}>{Utils.renderSourceTypeIconV2(tags)}</span> {runSource}{' '}
      {branchName && (
        <Tooltip title={branchName}>
          <Tag>
            <div css={{ display: 'flex', gap: 4, whiteSpace: 'nowrap' }}>
              <BranchIcon /> {branchName}
            </div>
          </Tag>
        </Tooltip>
      )}
    </div>
  ) : (
    <Typography.Hint>—</Typography.Hint>
  );
};
