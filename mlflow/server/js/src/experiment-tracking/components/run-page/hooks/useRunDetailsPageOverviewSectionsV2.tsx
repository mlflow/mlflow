import { Button, FileIcon, useDesignSystemTheme } from '@databricks/design-system';
import type { AsideSections } from '@databricks/web-shared/utils';
import { KeyValueProperty, NoneCell } from '@databricks/web-shared/utils';
import { FormattedMessage, useIntl } from 'react-intl';
import type { LoggedModelProto, RunDatasetWithTags, RunInfoEntity } from '../../../types';
import type { KeyValueEntity } from '../../../../common/types';
import type { UseGetRunQueryResponseRunInfo } from './useGetRunQuery';
import Utils from '../../../../common/utils/Utils';
import { RunViewTagsBox } from '../overview/RunViewTagsBox';
import { RunViewUserLinkBox } from '../overview/RunViewUserLinkBox';
import { DetailsOverviewCopyableIdBox } from '../../DetailsOverviewCopyableIdBox';
import { RunViewStatusBox } from '../overview/RunViewStatusBox';
import { RunViewParentRunBox } from '../overview/RunViewParentRunBox';
import { EXPERIMENT_PARENT_ID_TAG } from '../../experiment-page/utils/experimentPage.common-utils';
import { RunViewDatasetBoxV2 } from '../overview/RunViewDatasetBoxV2';
import { RunViewSourceBox } from '../overview/RunViewSourceBox';
import { Link, useLocation } from '../../../../common/utils/RoutingUtils';
import { RunViewLoggedModelsBox } from '../overview/RunViewLoggedModelsBox';
import { useMemo } from 'react';
import type { RunPageModelVersionSummary } from './useUnifiedRegisteredModelVersionsSummariesForRun';
import { RunViewRegisteredModelsBox } from '../overview/RunViewRegisteredModelsBox';
import Routes from '../../../routes';
import { RunViewRegisteredPromptsBox } from '../overview/RunViewRegisteredPromptsBox';

enum RunDetailsPageMetadataSections {
  DETAILS = 'DETAILS',
  DATASETS = 'DATASETS',
  TAGS = 'TAGS',
  REGISTERED_MODELS = 'REGISTERED_MODELS',
}

export const useRunDetailsPageOverviewSectionsV2 = ({
  runUuid,
  runInfo,
  tags,
  onTagsUpdated,
  datasets,
  shouldRenderLoggedModelsBox,
  loggedModelsV3,
  registeredModelVersionSummaries,
}: {
  runUuid: string;
  runInfo: RunInfoEntity | UseGetRunQueryResponseRunInfo;
  tags: Record<string, KeyValueEntity>;
  onTagsUpdated: () => void;
  datasets?: RunDatasetWithTags[];
  shouldRenderLoggedModelsBox?: boolean;
  loggedModelsV3: LoggedModelProto[];
  registeredModelVersionSummaries: RunPageModelVersionSummary[];
}): AsideSections => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const { search } = useLocation();
  const loggedModelsFromTags = useMemo(() => Utils.getLoggedModelsFromTags(tags), [tags]);

  const parentRunIdTag = tags[EXPERIMENT_PARENT_ID_TAG];

  const renderPromptMetadataRow = () => {
    return (
      <KeyValueProperty
        keyValue={intl.formatMessage({
          defaultMessage: 'Registered prompts',
          description: 'Run page > Overview > Run prompts section label',
        })}
        value={<RunViewRegisteredPromptsBox tags={tags} runUuid={runUuid} />}
      />
    );
  };

  const detailsContent = runInfo && (
    <>
      <KeyValueProperty
        keyValue={intl.formatMessage({
          defaultMessage: 'Created at',
          description: 'Run page > Overview > Run start time section label',
        })}
        value={runInfo.startTime ? Utils.formatTimestamp(runInfo.startTime, intl) : <NoneCell />}
      />
      <KeyValueProperty
        keyValue={intl.formatMessage({
          defaultMessage: 'Created by',
          description: 'Run page > Overview > Run author section label',
        })}
        value={<RunViewUserLinkBox runInfo={runInfo} tags={tags} />}
      />
      <KeyValueProperty
        keyValue={intl.formatMessage({
          defaultMessage: 'Experiment ID',
          description: 'Run page > Overview > experiment ID section label',
        })}
        value={
          <DetailsOverviewCopyableIdBox
            value={runInfo?.experimentId ?? ''}
            element={
              runInfo?.experimentId ? (
                <Link to={Routes.getExperimentPageRoute(runInfo.experimentId)}>{runInfo?.experimentId}</Link>
              ) : undefined
            }
          />
        }
      />
      <KeyValueProperty
        keyValue={intl.formatMessage({
          defaultMessage: 'Status',
          description: 'Run page > Overview > Run status section label',
        })}
        value={<RunViewStatusBox status={runInfo.status} />}
      />

      <KeyValueProperty
        keyValue={intl.formatMessage({
          defaultMessage: 'Run ID',
          description: 'Run page > Overview > Run ID section label',
        })}
        value={<DetailsOverviewCopyableIdBox value={runInfo.runUuid ?? ''} />}
      />

      <KeyValueProperty
        keyValue={intl.formatMessage({
          defaultMessage: 'Duration',
          description: 'Run page > Overview > Run duration section label',
        })}
        value={Utils.getDuration(runInfo.startTime, runInfo.endTime)}
      />

      {parentRunIdTag && (
        <KeyValueProperty
          keyValue={intl.formatMessage({
            defaultMessage: 'Parent run',
            description: 'Run page > Overview > Parent run',
          })}
          value={<RunViewParentRunBox parentRunUuid={parentRunIdTag.value} />}
        />
      )}
      <KeyValueProperty
        keyValue={intl.formatMessage({
          defaultMessage: 'Source',
          description: 'Run page > Overview > Run source section label',
        })}
        value={
          <RunViewSourceBox
            tags={tags}
            search={search}
            runUuid={runUuid}
            css={{
              paddingTop: theme.spacing.xs,
              paddingBottom: theme.spacing.xs,
            }}
          />
        }
      />
      {shouldRenderLoggedModelsBox && (
        <KeyValueProperty
          keyValue={intl.formatMessage({
            defaultMessage: 'Logged models',
            description: 'Run page > Overview > Run models section label',
          })}
          value={
            <RunViewLoggedModelsBox
              // Pass the run info and logged models
              runInfo={runInfo}
              loggedModels={loggedModelsFromTags}
              loggedModelsV3={loggedModelsV3}
            />
          }
        />
      )}
      {renderPromptMetadataRow()}
    </>
  );

  return [
    {
      id: RunDetailsPageMetadataSections.DETAILS,
      title: intl.formatMessage({
        defaultMessage: 'About this run',
        description: 'Title for the details/metadata section on the run details page',
      }),
      content: detailsContent,
    },
    {
      id: RunDetailsPageMetadataSections.DATASETS,
      title: intl.formatMessage({
        defaultMessage: 'Datasets',
        description: 'Title for the datasets section on the run details page',
      }),
      content: datasets?.length ? (
        <RunViewDatasetBoxV2 tags={tags} runInfo={runInfo} datasets={datasets} />
      ) : (
        <NoneCell />
      ),
    },
    {
      id: RunDetailsPageMetadataSections.TAGS,
      title: intl.formatMessage({
        defaultMessage: 'Tags',
        description: 'Title for the tags section on the run details page',
      }),
      content: <RunViewTagsBox runUuid={runInfo.runUuid ?? ''} tags={tags} onTagsUpdated={onTagsUpdated} />,
    },
    {
      id: RunDetailsPageMetadataSections.REGISTERED_MODELS,
      title: intl.formatMessage({
        defaultMessage: 'Registered models',
        description: 'Title for the registered models section on the run details page',
      }),
      content:
        registeredModelVersionSummaries?.length > 0 ? (
          <RunViewRegisteredModelsBox registeredModelVersionSummaries={registeredModelVersionSummaries} />
        ) : (
          <NoneCell />
        ),
    },
  ];
};
