import { Breadcrumb, LinkIcon, Tag, Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import { CopyButton } from '../../../shared/building_blocks/CopyButton';
import { Link, matchPath, useLocation, useParams } from '../../../common/utils/RoutingUtils';
import { WorkflowType } from '../../../common/contexts/WorkflowTypeContext';
import type { ExperimentEntity } from '../../types';
import { ExperimentPageTabName } from '../../constants';
import Routes, { RoutePaths } from '../../routes';
import { getTabDisplayName } from '../../components/experiment-page/components/header/ExperimentViewHeader.utils';
import { useGetExperimentPageActiveTabByRoute } from '../../components/experiment-page/hooks/useGetExperimentPageActiveTabByRoute';
import { useExperimentBreadcrumbStatus, useExperimentBreadcrumbTitle } from './ExperimentPageRevampContext';

const breadcrumbLabelCss = {
  display: 'inline-flex',
  alignItems: 'center',
  minWidth: 0,
  lineHeight: 'inherit',
} as const;

const ellipsisCss = {
  ...breadcrumbLabelCss,
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
} as const;

interface BreadcrumbDescriptor {
  parentTab: ExperimentPageTabName;
  leafId?: string;
}

const getBreadcrumbDescriptor = (
  pathname: string,
  activeTab: ExperimentPageTabName | undefined,
  params: Record<string, string | undefined>,
): BreadcrumbDescriptor | undefined => {
  if (matchPath(RoutePaths.experimentPageTabDatasetDetail, pathname)) {
    return { parentTab: ExperimentPageTabName.Datasets, leafId: params['datasetId'] };
  }
  if (matchPath(RoutePaths.experimentPageTabSingleChatSession, pathname)) {
    return {
      parentTab:
        activeTab === ExperimentPageTabName.Traces ? ExperimentPageTabName.Traces : ExperimentPageTabName.ChatSessions,
      leafId: params['sessionId'],
    };
  }
  if (matchPath(RoutePaths.experimentPageTabTraceDetail, pathname)) {
    return { parentTab: ExperimentPageTabName.Traces };
  }
  if (matchPath(RoutePaths.experimentPageTabEvaluationRunDetails, pathname)) {
    return { parentTab: ExperimentPageTabName.EvaluationRuns, leafId: params['runUuid'] };
  }
  if (matchPath(RoutePaths.experimentPageTabPromptDetails, pathname)) {
    return { parentTab: ExperimentPageTabName.Prompts, leafId: params['promptName'] };
  }
  return activeTab ? { parentTab: activeTab } : undefined;
};

export interface ExperimentPageBreadcrumbsProps {
  experiment: ExperimentEntity;
}

export const ExperimentPageBreadcrumbs = ({ experiment }: ExperimentPageBreadcrumbsProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { pathname } = useLocation();
  const params = useParams<'datasetId' | 'promptName' | 'runUuid' | 'sessionId'>();
  const { tabName } = useGetExperimentPageActiveTabByRoute();
  const publishedTitle = useExperimentBreadcrumbTitle();
  const descriptor = getBreadcrumbDescriptor(pathname, tabName, params);
  const leaf = descriptor?.leafId
    ? (publishedTitle ?? descriptor.leafId)
    : descriptor?.parentTab === ExperimentPageTabName.ReviewQueue
      ? publishedTitle
      : undefined;
  const experimentName = experiment.name.split('/').pop() ?? experiment.name;
  const runStatus = useExperimentBreadcrumbStatus();

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.sm,
        minWidth: 0,
        flexShrink: 0,
      }}
    >
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, flex: 1, minWidth: 0 }}>
        <nav
          aria-label={intl.formatMessage({
            defaultMessage: 'Breadcrumb',
            description: 'Aria label for experiment page breadcrumb navigation',
          })}
          css={{ flex: '0 1 auto', minWidth: 0, overflow: 'hidden' }}
        >
          <Breadcrumb includeTrailingCaret={false}>
            <Breadcrumb.Item>
              <Link componentId="mlflow.experiment_page.breadcrumb.experiments" to={Routes.experimentsObservatoryRoute}>
                <span css={breadcrumbLabelCss}>
                  {intl.formatMessage({
                    defaultMessage: 'Experiments',
                    description: 'Breadcrumb link to the experiments list',
                  })}
                </span>
              </Link>
            </Breadcrumb.Item>
            <Breadcrumb.Item>
              <Tooltip componentId="mlflow.experiment_page.breadcrumb.experiment_name_tooltip" content={experimentName}>
                <Link
                  componentId="mlflow.experiment_page.breadcrumb.experiment"
                  to={Routes.getExperimentPageRoute(experiment.experimentId)}
                >
                  <span css={[ellipsisCss, { maxWidth: 240 }]}>{experimentName}</span>
                </Link>
              </Tooltip>
            </Breadcrumb.Item>
            {descriptor && (
              <Breadcrumb.Item>
                {leaf ? (
                  <Link
                    componentId="mlflow.experiment_page.breadcrumb.tab"
                    to={Routes.getExperimentPageTabRoute(experiment.experimentId, descriptor.parentTab)}
                  >
                    <span css={breadcrumbLabelCss}>{getTabDisplayName(descriptor.parentTab, WorkflowType.GENAI)}</span>
                  </Link>
                ) : (
                  <span aria-current="page" css={breadcrumbLabelCss}>
                    {getTabDisplayName(descriptor.parentTab, WorkflowType.GENAI)}
                  </span>
                )}
              </Breadcrumb.Item>
            )}
            {leaf && (
              <Breadcrumb.Item>
                <Tooltip componentId="mlflow.experiment_page.breadcrumb.leaf_name_tooltip" content={leaf}>
                  <span aria-current="page" css={[ellipsisCss, { maxWidth: 320 }]}>
                    {leaf}
                  </span>
                </Tooltip>
              </Breadcrumb.Item>
            )}
          </Breadcrumb>
        </nav>
        {runStatus && (
          <div css={{ flexShrink: 0 }}>
            <Tag componentId="mlflow.eval_run_details.status_tag" color="lemon">
              {runStatus}
            </Tag>
          </div>
        )}
      </div>
      <div css={{ flexShrink: 0 }}>
        <CopyButton
          componentId="mlflow.experiment_page.breadcrumb.copy_link"
          copyText={window.location.href}
          showLabel={false}
          type={undefined}
          size="small"
          icon={<LinkIcon />}
          aria-label={intl.formatMessage({
            defaultMessage: 'Copy link to this experiment',
            description: 'Accessible label for the button that copies a link to the current experiment page',
          })}
        />
      </div>
    </div>
  );
};
