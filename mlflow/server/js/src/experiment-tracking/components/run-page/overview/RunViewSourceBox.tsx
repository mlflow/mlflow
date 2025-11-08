import {
  BranchIcon,
  CopyIcon,
  GitCommitIcon,
  Tag,
  LegacyTooltip,
  Typography,
  useDesignSystemTheme,
  Popover,
} from '@databricks/design-system';
import Utils from '../../../../common/utils/Utils';
import type { KeyValueEntity } from '../../../../common/types';
import { MLFLOW_RUN_GIT_SOURCE_BRANCH_TAG } from '../../../constants';
import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';
import { ExperimentSourceTypeIcon } from '../../ExperimentSourceTypeIcon';

export const RunViewSourceBox = ({
  runUuid,
  tags,
  search,
  className,
}: {
  runUuid: string;
  tags: Record<string, KeyValueEntity>;
  search: string;
  className?: string;
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
      className={className}
    >
      <ExperimentSourceTypeIcon
        sourceType={tags[Utils.sourceTypeTag]?.value}
        css={{ color: theme.colors.actionPrimaryBackgroundDefault }}
      />
      {runSource}{' '}
      {branchName && (
        <LegacyTooltip title={branchName}>
          <Tag
            componentId="codegen_mlflow_app_src_experiment-tracking_components_run-page_overview_runviewsourcebox.tsx_48"
            css={{ marginRight: 0 }}
          >
            <div css={{ display: 'flex', gap: 4, whiteSpace: 'nowrap' }}>
              <BranchIcon /> {branchName}
            </div>
          </Tag>
        </LegacyTooltip>
      )}
      {commitHash && (
        <Popover.Root componentId="mlflow.run_details.overview.source.commit_hash_popover">
          <Popover.Trigger asChild>
            <Tag
              componentId="mlflow.run_details.overview.source.commit_hash"
              css={{ marginRight: 0, '&>div': { paddingRight: 0 } }}
            >
              <div css={{ display: 'flex', gap: theme.spacing.xs, whiteSpace: 'nowrap', alignContent: 'center' }}>
                <GitCommitIcon />
                {commitHash.slice(0, 7)}
              </div>
            </Tag>
          </Popover.Trigger>
          <Popover.Content align="start">
            <Popover.Arrow />
            <div css={{ display: 'flex', gap: theme.spacing.xs, alignItems: 'center' }}>
              {commitHash}
              <CopyButton showLabel={false} size="small" type="tertiary" copyText={commitHash} icon={<CopyIcon />} />
            </div>
          </Popover.Content>
        </Popover.Root>
      )}
    </div>
  ) : (
    <Typography.Hint>—</Typography.Hint>
  );
};
