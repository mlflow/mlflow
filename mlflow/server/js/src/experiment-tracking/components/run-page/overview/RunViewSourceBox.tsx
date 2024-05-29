import {
  BranchIcon,
  CopyIcon,
  GitCommitIcon,
  Tag,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import Utils from '../../../../common/utils/Utils';
import type { KeyValueEntity } from '../../../types';
import { MLFLOW_RUN_GIT_SOURCE_BRANCH_TAG } from '../../../constants';
import { CopyButton } from 'shared/building_blocks/CopyButton';
import { ExperimentSourceTypeIcon } from '../../ExperimentSourceTypeIcon';

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
  const commitHash = tags?.[Utils.gitCommitTag]?.value;
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
      <ExperimentSourceTypeIcon
        sourceType={tags[Utils.sourceTypeTag]?.value}
        css={{ color: theme.colors.actionPrimaryBackgroundDefault }}
      />
      {runSource}{' '}
      {branchName && (
        <Tooltip title={branchName}>
          <Tag css={{ marginRight: 0 }}>
            <div css={{ display: 'flex', gap: 4, whiteSpace: 'nowrap' }}>
              <BranchIcon /> {branchName}
            </div>
          </Tag>
        </Tooltip>
      )}
      {commitHash && (
        <Tooltip
          dangerouslySetAntdProps={{ overlayStyle: { maxWidth: 'none' } }}
          title={
            <div css={{ display: 'flex', gap: 4, alignItems: 'center' }}>
              {commitHash}
              <CopyButton
                css={{ flex: '0 0 auto' }}
                showLabel={false}
                size="small"
                type="tertiary"
                copyText={commitHash}
                icon={<CopyIcon />}
              />
            </div>
          }
        >
          <Tag css={{ marginRight: 0 }}>
            <div css={{ display: 'flex', gap: 4, whiteSpace: 'nowrap' }}>
              <GitCommitIcon /> {commitHash.slice(0, 7)}
            </div>
          </Tag>
        </Tooltip>
      )}
    </div>
  ) : (
    <Typography.Hint>—</Typography.Hint>
  );
};
