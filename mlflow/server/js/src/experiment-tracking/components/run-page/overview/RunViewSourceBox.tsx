import {
  BranchIcon,
  CopyIcon,
  GitCommitIcon,
  Tag,
  LegacyTooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import Utils from '../../../../common/utils/Utils';
import type { KeyValueEntity } from '../../../types';
import { MLFLOW_RUN_GIT_SOURCE_BRANCH_TAG } from '../../../constants';
import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';
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
        <LegacyTooltip
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
          <Tag
            componentId="codegen_mlflow_app_src_experiment-tracking_components_run-page_overview_runviewsourcebox.tsx_72"
            css={{ marginRight: 0 }}
          >
            <div css={{ display: 'flex', gap: 4, whiteSpace: 'nowrap' }}>
              <GitCommitIcon /> {commitHash.slice(0, 7)}
            </div>
          </Tag>
        </LegacyTooltip>
      )}
    </div>
  ) : (
    <Typography.Hint>—</Typography.Hint>
  );
};
