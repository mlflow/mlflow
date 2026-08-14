import { FormattedMessage, useIntl } from 'react-intl';
import { Button, FileIcon, InfoSmallIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import Utils from '../../../../common/utils/Utils';
import { EXPERIMENT_PARENT_ID_TAG } from '../../experiment-page/utils/experimentPage.common-utils';

import { RunViewStatusBox } from './RunViewStatusBox';
import { RunViewUserLinkBox } from './RunViewUserLinkBox';
import { RunViewDatasetBox } from './RunViewDatasetBox';
import { RunViewParentRunBox } from './RunViewParentRunBox';
import { RunViewTagsBox } from './RunViewTagsBox';
import { RunViewDescriptionBox } from './RunViewDescriptionBox';
import { RunViewRegisteredModelsBox } from './RunViewRegisteredModelsBox';
import { RunViewSourceBox } from './RunViewSourceBox';
import { DetailsOverviewCopyableIdBox } from '../../DetailsOverviewCopyableIdBox';
import type { RunInfoEntity } from '../../../types';
import type { RunDatasetWithTags } from '../../../types';
import type { UseGetRunQueryResponseRunInfo } from '../hooks/useGetRunQuery';
import type { KeyValueEntity } from '../../../../common/types';
import { type RunPageModelVersionSummary } from '../hooks/useUnifiedRegisteredModelVersionsSummariesForRun';

const { Title } = Typography;

const EmptyValue = () => <Typography.Hint>—</Typography.Hint>;

const PairsContentSection = ({ title, value }: { title: React.ReactNode; value: React.ReactNode }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <tr
      css={{
        display: 'flex',
        minHeight: theme.general.heightSm,
      }}
    >
      <th
        css={{
          flex: `0 0 164px`,
          padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'start',
        }}
      >
        {title}
      </th>
      <td
        css={{
          flex: 1,
          padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'start',
          wordWrap: 'break-word',
          wordBreak: 'break-word',
        }}
      >
        {value}
      </td>
    </tr>
  );
};

export const RunViewDetailsMetadataBox = ({
  runUuid,
  runInfo,
  tags,
  datasets,
  search,
  onRunDataUpdated,
  registeredModelVersionSummaries,
}: {
  runUuid: string;
  runInfo: RunInfoEntity | UseGetRunQueryResponseRunInfo;
  tags: Record<string, KeyValueEntity>;
  datasets?: RunDatasetWithTags[];
  search: string;
  onRunDataUpdated: () => void | Promise<any>;
  registeredModelVersionSummaries: RunPageModelVersionSummary[];
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const parentRunIdTag = tags[EXPERIMENT_PARENT_ID_TAG];
  return (
    <section
      aria-labelledby="Run details section"
      css={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'flex-start',
        gap: theme.spacing.md,
        padding: theme.spacing.md,
        borderRadius: theme.spacing.sm,
        border: `1px solid ${theme.colors.borderDecorative}`,
      }}
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.sm,
        }}
      >
        <InfoSmallIcon
          css={{
            width: theme.spacing.lg,
            height: theme.spacing.lg,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: theme.colors.textSecondary,
            backgroundColor: theme.colors.backgroundSecondary,
            borderRadius: theme.spacing.sm,
          }}
        />
        <Title level={3} withoutMargins>
          <FormattedMessage
            defaultMessage="Run details"
            description="Run page > Overview > Run details section heading"
          />
        </Title>
      </div>
      <table css={{ display: 'flex' }}>
        <tbody>
          <PairsContentSection
            title={
              <Typography.Text bold withoutMargins>
                <FormattedMessage
                  defaultMessage="Description"
                  description="Run page > Overview > Description section > Section title"
                />
              </Typography.Text>
            }
            value={
              <RunViewDescriptionBox
                runUuid={runUuid}
                tags={tags}
                onDescriptionChanged={onRunDataUpdated}
                isFlatLayout
              />
            }
          />
          <PairsContentSection
            title={
              <Typography.Text bold withoutMargins>
                <FormattedMessage
                  defaultMessage="Created at"
                  description="Run page > Overview > Run start time section label"
                />
              </Typography.Text>
            }
            value={runInfo.startTime ? Utils.formatTimestamp(runInfo.startTime, intl) : <EmptyValue />}
          />
          <PairsContentSection
            title={
              <Typography.Text bold withoutMargins>
                <FormattedMessage
                  defaultMessage="Created by"
                  description="Run page > Overview > Run author section label"
                />
              </Typography.Text>
            }
            value={<RunViewUserLinkBox runInfo={runInfo} tags={tags} />}
          />
          <PairsContentSection
            title={
              <Typography.Text bold withoutMargins>
                <FormattedMessage
                  defaultMessage="Status"
                  description="Run page > Overview > Run status section label"
                />
              </Typography.Text>
            }
            value={<RunViewStatusBox status={runInfo.status} useSpinner />}
          />
          <PairsContentSection
            title={
              <Typography.Text bold withoutMargins>
                <FormattedMessage defaultMessage="Run ID" description="Run page > Overview > Run ID section label" />
              </Typography.Text>
            }
            value={<DetailsOverviewCopyableIdBox value={runInfo.runUuid ?? ''} />}
          />
          <PairsContentSection
            title={
              <Typography.Text bold withoutMargins>
                <FormattedMessage
                  defaultMessage="Duration"
                  description="Run page > Overview > Run duration section label"
                />
              </Typography.Text>
            }
            value={Utils.getDuration(runInfo.startTime, runInfo.endTime)}
          />
          <PairsContentSection
            title={
              <Typography.Text bold withoutMargins>
                <FormattedMessage
                  defaultMessage="Dataset used"
                  description="Run page > Overview > Run datasets section label"
                />
              </Typography.Text>
            }
            value={
              datasets?.length ? (
                <RunViewDatasetBox tags={tags} runInfo={runInfo} datasets={datasets} />
              ) : (
                <EmptyValue />
              )
            }
          />
          <PairsContentSection
            title={
              <Typography.Text bold withoutMargins>
                <FormattedMessage
                  defaultMessage="Type"
                  description="Run page > Overview > Experiment ID section label"
                />
              </Typography.Text>
            }
            value={<Typography.Text withoutMargins>Training</Typography.Text>}
          />
          <PairsContentSection
            title={
              <Typography.Text bold withoutMargins>
                <FormattedMessage defaultMessage="Tags" description="Run page > Overview > Run tags section label" />
              </Typography.Text>
            }
            value={
              <RunViewTagsBox
                runUuid={runInfo.runUuid ?? ''}
                tags={tags}
                onTagsUpdated={onRunDataUpdated}
                css={{
                  paddingTop: 0,
                  paddingBottom: 0,
                  '& div': { paddingTop: 0, paddingBottom: 0 },
                  '& button': { paddingTop: 0, paddingBottom: 0 },
                }}
              />
            }
          />
          <PairsContentSection
            title={
              <Typography.Text bold withoutMargins>
                <FormattedMessage
                  defaultMessage="Source"
                  description="Run page > Overview > Run source section label"
                />
              </Typography.Text>
            }
            value={
              <RunViewSourceBox tags={tags} search={search} runUuid={runUuid} css={{ padding: 0 }} hasIcon={false} />
            }
          />
          {parentRunIdTag && (
            <PairsContentSection
              title={
                <Typography.Text bold withoutMargins>
                  <FormattedMessage defaultMessage="Parent run" description="Run page > Overview > Parent run" />
                </Typography.Text>
              }
              value={<RunViewParentRunBox parentRunUuid={parentRunIdTag.value} />}
            />
          )}
          <PairsContentSection
            title={
              <Typography.Text bold withoutMargins>
                <FormattedMessage
                  defaultMessage="Model registered"
                  description="Run page > Overview > Model registered section label"
                />
              </Typography.Text>
            }
            value={
              registeredModelVersionSummaries?.length > 0 ? (
                <RunViewRegisteredModelsBox registeredModelVersionSummaries={registeredModelVersionSummaries} />
              ) : (
                <EmptyValue />
              )
            }
          />
        </tbody>
      </table>
    </section>
  );
};
