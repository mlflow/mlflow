import { ClassNames } from '@emotion/react';
import type { ReactNode } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import {
  InfoTooltip,
  NewWindowIcon,
  TableClockIcon,
  TableIcon,
  Tag,
  Tooltip,
  TruncationTooltip,
  Typography,
  UserIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { Link } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { ExperimentPageTabName } from '../../constants';
import { DEFAULT_INLINE_DATASET_NAME } from '../../types';
import { PROMPT_VERSION_QUERY_PARAM } from '../prompts/utils';
import { SELECTED_SCORER_NAME_QUERY_PARAM_KEY } from '../experiment-scorers/hooks/useSelectedScorerName';
import { getEvaluationDatasetId } from './datasetLineage';
import type { RegisteredScorerReference } from './scorerLineage';
import { formatShortDigest, type EvalRunAgentLineage, type EvalRunScorersDigest } from './evalRunLineage';
import { useGetLoggedModelQuery } from '../../hooks/logged-models/useGetLoggedModelQuery';

export interface RunDetailsStats {
  traces?: number;
  /** When true, only the newest `traces` were loaded, so the true total is higher — shown as "N+". */
  tracesTruncated?: boolean;
  totalTokens?: number;
  totalCost?: number;
  errors?: number;
}

export interface RunDetailsSidebarProps {
  experimentId: string;
  createdBy?: string;
  createdAt?: number;
  dataset?: {
    digest?: string | null;
    name?: string | null;
    profile?: string | null;
    schema?: string | null;
    source?: string | null;
    sourceType?: string | null;
  };
  endpoint?: string;
  /** Prompt versions the run is linked to, from its `mlflow.linkedPrompts` tag. */
  prompts?: RunDetailsPrompt[];
  stats: RunDetailsStats;
  scorers: RunDetailsScorer[];
  tags: { key: string; value: string }[];
  /** Lineage `mlflow.genai.evaluate` records on the run. */
  scorersDigest?: EvalRunScorersDigest;
  agent?: EvalRunAgentLineage;
  gitCommit?: string;
}

export interface RunDetailsScorer {
  name: string;
  displayName: string;
  /** The registered scorer behind every assessment with this name, when there is one. */
  scorerReference?: RegisteredScorerReference;
  /** The scorer's definition digest (`sha256:<hex>`), when all of its assessments agree. */
  digest?: string;
}

export interface RunDetailsPrompt {
  name: string;
  version: string;
}

export interface DetailRowProps {
  label: ReactNode;
  children: ReactNode;
}

/** Shown in a stat slot when the underlying metric is not present on the run. */
const EM_DASH = '—';

const RUN_DETAILS_SIDEBAR_PREFERRED_WIDTH_PX = 320;
const RUN_DETAILS_SIDEBAR_MIN_WIDTH_PX = 260;

const TRUNCATED_TEXT_LAYOUT_STYLE = {
  display: 'block',
  flex: '1 1 auto',
  width: '100%',
  minWidth: 0,
  maxWidth: '100%',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
} as const;

const RUN_DETAILS_SIDEBAR_LAYOUT_STYLE = {
  flex: `0 1 ${RUN_DETAILS_SIDEBAR_PREFERRED_WIDTH_PX}px`,
  marginLeft: 'auto',
  minWidth: `min(${RUN_DETAILS_SIDEBAR_MIN_WIDTH_PX}px, 100%)`,
  maxWidth: '100%',
} as const;

export const DetailRow = ({ label, children }: DetailRowProps) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ display: 'flex', justifyContent: 'space-between', gap: theme.spacing.md }}>
      <Typography.Text color="secondary" size="md">
        {label}
      </Typography.Text>
      <div css={{ flex: '1 1 auto', textAlign: 'right', minWidth: 0, maxWidth: '100%' }}>{children}</div>
    </div>
  );
};

const TruncatedSidebarText = ({ text, title = text }: { text: string; title?: string }) => (
  <span style={TRUNCATED_TEXT_LAYOUT_STYLE} title={title}>
    <Typography.Text size="md">{text}</Typography.Text>
  </span>
);

const ScorersDigestTooltipContent = ({ scorersDigest }: { scorersDigest: EvalRunScorersDigest }) => {
  const { theme } = useDesignSystemTheme();
  if (!scorersDigest.digest) {
    return (
      <FormattedMessage
        defaultMessage="No scorers digest. MLflow could not read any scorer in this run."
        description="Tooltip on the eval run scorers heading when no scorer definition could be hashed"
      />
    );
  }
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, overflowWrap: 'anywhere' }}>
      <span>
        <FormattedMessage
          defaultMessage="Scorers digest: {digest}"
          description="Eval run detail: digest identifying the scorer definitions used by the run"
          values={{ digest: scorersDigest.digest }}
        />
      </span>
      <span>
        {scorersDigest.coverage === 'partial' ? (
          <FormattedMessage
            defaultMessage="A fingerprint of {hashed} of {total} scorers in this run. MLflow could not read the others."
            description="Eval run detail: the scorers digest covers only the scorers whose definition could be hashed"
            values={{ hashed: scorersDigest.hashed, total: scorersDigest.total }}
          />
        ) : (
          <FormattedMessage
            defaultMessage="A fingerprint of all scorers in this run."
            description="Tooltip explaining what the eval run scorers digest identifies"
          />
        )}
      </span>
    </div>
  );
};

const AgentFunctionSidebarValue = ({ agentFunction, digest }: { agentFunction: string; digest?: string }) => {
  if (!digest) {
    return <TruncatedSidebarText text={agentFunction} />;
  }
  return (
    <Tooltip
      componentId="mlflow.eval_run_details.overview.agent_digest_tooltip"
      content={
        <span css={{ overflowWrap: 'anywhere' }}>
          <FormattedMessage
            defaultMessage="{agentFunction} (source digest: {digest})"
            description="Eval run detail: tooltip on the evaluated agent function with the digest of its source code"
            values={{ agentFunction, digest }}
          />
        </span>
      }
    >
      <span style={TRUNCATED_TEXT_LAYOUT_STYLE}>
        <Typography.Text size="md">{agentFunction}</Typography.Text>
      </span>
    </Tooltip>
  );
};

const GIT_SHORT_COMMIT_LENGTH = 7;

const LoggedModelSidebarValue = ({ experimentId, loggedModelId }: { experimentId: string; loggedModelId: string }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { data } = useGetLoggedModelQuery({ loggedModelId });
  const modelName = data?.info?.name ?? loggedModelId;

  return (
    <Link
      componentId="mlflow.eval_run_details.overview.logged_model_link"
      to={Routes.getExperimentLoggedModelDetailsPageRoute(data?.info?.experiment_id ?? experimentId, loggedModelId)}
      target="_blank"
      aria-label={intl.formatMessage(
        {
          defaultMessage: 'View logged model {modelName}',
          description: 'Accessible label for a link that opens the logged model of an evaluation run',
        },
        { modelName },
      )}
      css={{
        ...linkValueCss,
        gap: theme.spacing.xs,
        color: theme.colors.actionTertiaryTextDefault,
      }}
    >
      <TruncationTooltip componentId="mlflow.eval_run_details.overview.logged_model_tooltip">
        <span style={TRUNCATED_TEXT_LAYOUT_STYLE}>{modelName}</span>
      </TruncationTooltip>
      <NewWindowIcon aria-hidden="true" css={{ flexShrink: 0 }} />
    </Link>
  );
};

const AgentSidebarSection = ({ experimentId, agent }: { experimentId: string; agent: EvalRunAgentLineage }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      <Typography.Text bold size="lg">
        <FormattedMessage
          defaultMessage="Agent"
          description="Heading for the evaluated agent section in the evaluation run sidebar"
        />
      </Typography.Text>
      {agent.function && (
        <DetailRow
          label={<FormattedMessage defaultMessage="Function" description="Eval run detail: evaluated agent function" />}
        >
          <AgentFunctionSidebarValue agentFunction={agent.function} digest={agent.digest} />
        </DetailRow>
      )}
      {agent.uri && (
        <DetailRow
          label={
            <FormattedMessage
              defaultMessage="URI"
              description="Eval run detail: URI of the evaluated serving endpoint or app"
            />
          }
        >
          <TruncatedSidebarText text={agent.uri} />
        </DetailRow>
      )}
      {agent.digest && !agent.function && (
        <DetailRow
          label={
            <FormattedMessage
              defaultMessage="Source digest"
              description="Eval run detail: digest of the evaluated agent function source code"
            />
          }
        >
          <Tooltip componentId="mlflow.eval_run_details.overview.agent_digest_tooltip" content={agent.digest}>
            <span>
              <Typography.Text size="md">{formatShortDigest(agent.digest)}</Typography.Text>
            </span>
          </Tooltip>
        </DetailRow>
      )}
      {agent.loggedModelId && (
        <DetailRow
          label={
            <FormattedMessage
              defaultMessage="Logged model"
              description="Eval run detail: logged model linked to the evaluation run"
            />
          }
        >
          <LoggedModelSidebarValue experimentId={experimentId} loggedModelId={agent.loggedModelId} />
        </DetailRow>
      )}
    </div>
  );
};

const linkValueCss = {
  display: 'inline-flex',
  alignItems: 'center',
  minWidth: 0,
  maxWidth: '100%',
} as const;

const DatasetSidebarValue = ({ name, isLink, isUnnamed }: { name: string; isLink: boolean; isUnnamed: boolean }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const displayText = isUnnamed
    ? intl.formatMessage({
        defaultMessage: 'Unnamed dataset',
        description: 'Eval run detail sidebar: label for a dataset logged without a meaningful name',
      })
    : name;
  return (
    <span
      css={{
        ...linkValueCss,
        justifyContent: 'flex-end',
        gap: theme.spacing.xs,
        color: isLink ? theme.colors.actionTertiaryTextDefault : theme.colors.textPrimary,
      }}
    >
      {isUnnamed ? (
        <TableClockIcon aria-hidden="true" css={{ flexShrink: 0 }} />
      ) : (
        <TableIcon aria-hidden="true" css={{ flexShrink: 0 }} />
      )}
      <TruncationTooltip componentId="mlflow.eval_run_details.overview.dataset_tooltip">
        <span style={TRUNCATED_TEXT_LAYOUT_STYLE}>{displayText}</span>
      </TruncationTooltip>
      {isLink && <NewWindowIcon aria-hidden="true" css={{ flexShrink: 0 }} />}
    </span>
  );
};

const DatasetSidebarLink = ({
  experimentId,
  dataset,
}: {
  experimentId: string;
  dataset: NonNullable<RunDetailsSidebarProps['dataset']>;
}) => {
  const intl = useIntl();
  const isUnnamed = !dataset.name || dataset.name === DEFAULT_INLINE_DATASET_NAME;
  const datasetId = getEvaluationDatasetId(dataset.source);
  if (!datasetId) {
    return <DatasetSidebarValue name={dataset.name ?? ''} isLink={false} isUnnamed={isUnnamed} />;
  }
  return (
    <Link
      componentId="mlflow.eval_run_details.overview.dataset_link"
      to={Routes.getExperimentPageDatasetDetailRoute(experimentId, datasetId)}
      target="_blank"
      aria-label={intl.formatMessage(
        {
          defaultMessage: 'View dataset {datasetName}',
          description: 'Accessible label for a link that opens the evaluation dataset an evaluation run used',
        },
        { datasetName: dataset.name ?? datasetId },
      )}
      css={{ display: 'inline-flex', maxWidth: '100%' }}
    >
      <DatasetSidebarValue name={dataset.name ?? ''} isLink isUnnamed={isUnnamed} />
    </Link>
  );
};

const ScorerSidebarValue = ({ name, isLink = false }: { name: string; isLink?: boolean }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <span
      css={{
        ...linkValueCss,
        gap: theme.spacing.xs,
        color: isLink ? theme.colors.actionTertiaryTextDefault : theme.colors.textPrimary,
      }}
    >
      <span style={TRUNCATED_TEXT_LAYOUT_STYLE} title={name}>
        {name}
      </span>
      {isLink && <NewWindowIcon aria-hidden="true" css={{ flexShrink: 0 }} />}
    </span>
  );
};

const ScorerSidebarItem = ({ experimentId, scorer }: { experimentId: string; scorer: RunDetailsScorer }) => {
  const intl = useIntl();
  if (!scorer.scorerReference) {
    return <ScorerSidebarValue name={scorer.displayName} />;
  }
  const searchParams = new URLSearchParams({
    [SELECTED_SCORER_NAME_QUERY_PARAM_KEY]: scorer.scorerReference.scorerName,
  });
  return (
    <Link
      componentId="mlflow.eval_run_details.overview.scorer_link"
      to={{
        pathname: Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Judges),
        search: searchParams.toString(),
      }}
      target="_blank"
      aria-label={intl.formatMessage(
        {
          defaultMessage: 'View scorer {scorerName}',
          description:
            'Accessible label for a link that opens the registered scorer behind an evaluation run assessment',
        },
        { scorerName: scorer.displayName },
      )}
      css={{ display: 'inline-flex', maxWidth: '100%' }}
    >
      <ScorerSidebarValue name={scorer.displayName} isLink />
    </Link>
  );
};

const PromptSidebarLink = ({ experimentId, prompt }: { experimentId: string; prompt: RunDetailsPrompt }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const searchParams = new URLSearchParams({ [PROMPT_VERSION_QUERY_PARAM]: prompt.version });
  return (
    <Link
      componentId="mlflow.eval_run_details.overview.prompt_link"
      to={{
        pathname: Routes.getPromptDetailsPageRoute(encodeURIComponent(prompt.name), experimentId),
        search: searchParams.toString(),
      }}
      target="_blank"
      aria-label={intl.formatMessage(
        {
          defaultMessage: 'View prompt {promptName} version {promptVersion}',
          description: 'Accessible label for a link that opens the prompt version an evaluation run is linked to',
        },
        { promptName: prompt.name, promptVersion: prompt.version },
      )}
      css={{
        ...linkValueCss,
        gap: theme.spacing.xs,
        color: theme.colors.actionTertiaryTextDefault,
      }}
    >
      <TruncationTooltip componentId="mlflow.eval_run_details.overview.prompt_tooltip">
        <span style={TRUNCATED_TEXT_LAYOUT_STYLE}>
          <FormattedMessage
            defaultMessage="{promptName} (v{promptVersion})"
            description="Eval run detail sidebar: a linked prompt and its version"
            values={{ promptName: prompt.name, promptVersion: prompt.version }}
          />
        </span>
      </TruncationTooltip>
      <NewWindowIcon aria-hidden="true" css={{ flexShrink: 0 }} />
    </Link>
  );
};

export const RunDetailsSidebar = ({
  experimentId,
  createdBy,
  createdAt,
  dataset,
  endpoint,
  prompts,
  stats,
  scorers,
  tags,
  scorersDigest,
  agent,
  gitCommit,
}: RunDetailsSidebarProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  return (
    <aside
      style={RUN_DETAILS_SIDEBAR_LAYOUT_STYLE}
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.lg,
      }}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        <Typography.Text bold size="lg">
          <FormattedMessage
            defaultMessage="Run Details"
            description="Heading for the run details sidebar in the evaluation run sidebar"
          />
        </Typography.Text>
        {createdBy && (
          <DetailRow
            label={<FormattedMessage defaultMessage="Created by" description="Eval run detail: who created the run" />}
          >
            <span
              css={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: theme.spacing.xs,
                minWidth: 0,
                maxWidth: '100%',
              }}
            >
              <span css={{ display: 'flex', flexShrink: 0 }}>
                <UserIcon />
              </span>
              <TruncatedSidebarText text={createdBy} />
            </span>
          </DetailRow>
        )}
        {createdAt !== undefined && (
          <DetailRow
            label={
              <FormattedMessage defaultMessage="Date created" description="Eval run detail: when the run was created" />
            }
          >
            <Typography.Text size="md">
              {intl.formatDate(createdAt, { dateStyle: 'medium', timeStyle: 'medium' })}
            </Typography.Text>
          </DetailRow>
        )}
        <DetailRow
          label={<FormattedMessage defaultMessage="Dataset" description="Eval run detail: evaluated dataset name" />}
        >
          {dataset ? (
            <DatasetSidebarLink experimentId={experimentId} dataset={dataset} />
          ) : (
            <Typography.Text size="md">{EM_DASH}</Typography.Text>
          )}
        </DetailRow>
        <DetailRow
          label={
            <FormattedMessage
              defaultMessage="Dataset digest"
              description="Eval run detail: digest identifying the evaluated dataset contents"
            />
          }
        >
          {dataset?.digest ? (
            <TruncatedSidebarText text={dataset.digest} />
          ) : (
            <Typography.Text size="md">{EM_DASH}</Typography.Text>
          )}
        </DetailRow>
        <DetailRow
          label={
            <FormattedMessage
              defaultMessage="Dataset source type"
              description="Eval run detail: the evaluated dataset's source type (e.g. code, local, delta_table)"
            />
          }
        >
          <Typography.Text size="md">{dataset?.sourceType || EM_DASH}</Typography.Text>
        </DetailRow>
        {endpoint && (
          <DetailRow
            label={<FormattedMessage defaultMessage="Endpoint" description="Eval run detail: evaluated endpoint" />}
          >
            <TruncatedSidebarText text={endpoint} />
          </DetailRow>
        )}
        <DetailRow
          label={<FormattedMessage defaultMessage="Prompt" description="Eval run detail: prompt used by the run" />}
        >
          {prompts?.length ? (
            <div
              css={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'flex-end',
                gap: theme.spacing.xs,
                minWidth: 0,
              }}
            >
              {prompts.map((prompt) => (
                <PromptSidebarLink
                  key={`${prompt.name}:${prompt.version}`}
                  experimentId={experimentId}
                  prompt={prompt}
                />
              ))}
            </div>
          ) : (
            <Typography.Text size="md">{EM_DASH}</Typography.Text>
          )}
        </DetailRow>
        {gitCommit && (
          <DetailRow
            label={
              <FormattedMessage
                defaultMessage="Git commit"
                description="Eval run detail: git commit of the run's code"
              />
            }
          >
            <TruncatedSidebarText text={gitCommit.slice(0, GIT_SHORT_COMMIT_LENGTH)} title={gitCommit} />
          </DetailRow>
        )}
      </div>

      {agent && <AgentSidebarSection experimentId={experimentId} agent={agent} />}

      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        <Typography.Text bold size="lg">
          <FormattedMessage
            defaultMessage="Run stats"
            description="Heading for the run statistics section in the evaluation run sidebar"
          />
        </Typography.Text>
        <DetailRow label={<FormattedMessage defaultMessage="Traces" description="Eval run stat: number of traces" />}>
          <Typography.Text size="md">
            {stats.tracesTruncated ? (
              <FormattedMessage
                defaultMessage="{count, number}+"
                description="Eval run trace count shown when only the most recent traces were loaded, so the true total is higher"
                values={{ count: stats.traces ?? 0 }}
              />
            ) : stats.traces !== undefined ? (
              intl.formatNumber(stats.traces)
            ) : (
              EM_DASH
            )}
          </Typography.Text>
        </DetailRow>
        <DetailRow
          label={<FormattedMessage defaultMessage="Total tokens" description="Eval run stat: total tokens used" />}
        >
          <Typography.Text size="md">
            {stats.totalTokens !== undefined ? intl.formatNumber(stats.totalTokens) : EM_DASH}
          </Typography.Text>
        </DetailRow>
        <DetailRow
          label={
            <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}>
              <FormattedMessage defaultMessage="Total cost" description="Eval run stat: total cost" />
              <InfoTooltip
                componentId="mlflow.eval_run_details.run_stats.total_cost.info"
                iconTitle={intl.formatMessage({
                  defaultMessage: 'Estimated cost',
                  description: 'Accessible label for evaluation run estimated cost information',
                })}
                content={intl.formatMessage({
                  defaultMessage: 'This is an estimate of cost and may be inaccurate.',
                  description: 'Tooltip explaining that evaluation run total cost is an estimate',
                })}
              />
            </span>
          }
        >
          <Typography.Text size="md">
            {stats.totalCost !== undefined
              ? intl.formatNumber(stats.totalCost, { style: 'currency', currency: 'USD', maximumFractionDigits: 6 })
              : EM_DASH}
          </Typography.Text>
        </DetailRow>
        <DetailRow label={<FormattedMessage defaultMessage="Errors" description="Eval run stat: number of errors" />}>
          <Typography.Text size="md">
            {stats.errors !== undefined ? intl.formatNumber(stats.errors) : EM_DASH}
          </Typography.Text>
        </DetailRow>
      </div>

      {scorers.length > 0 && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          {scorersDigest ? (
            <Tooltip
              componentId="mlflow.eval_run_details.overview.scorers_digest_tooltip"
              content={<ScorersDigestTooltipContent scorersDigest={scorersDigest} />}
            >
              <span css={{ alignSelf: 'flex-start' }}>
                <Typography.Text bold size="lg">
                  <FormattedMessage
                    defaultMessage="Scorers ({count})"
                    description="Heading for the list of scorers in the evaluation run sidebar, with a count"
                    values={{ count: scorers.length }}
                  />
                </Typography.Text>
              </span>
            </Tooltip>
          ) : (
            <Typography.Text bold size="lg">
              <FormattedMessage
                defaultMessage="Scorers ({count})"
                description="Heading for the list of scorers in the evaluation run sidebar, with a count"
                values={{ count: scorers.length }}
              />
            </Typography.Text>
          )}
          {scorers.map((scorer) => (
            <div
              key={scorer.name}
              css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: theme.spacing.sm }}
            >
              <ScorerSidebarItem experimentId={experimentId} scorer={scorer} />
              {scorer.digest && (
                <Tooltip componentId="mlflow.eval_run_details.overview.scorer_digest_tooltip" content={scorer.digest}>
                  <span css={{ flexShrink: 0 }}>
                    <Typography.Text size="sm" color="secondary">
                      {formatShortDigest(scorer.digest)}
                    </Typography.Text>
                  </span>
                </Tooltip>
              )}
            </div>
          ))}
        </div>
      )}

      {tags.length > 0 && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <Typography.Text bold size="lg">
            <FormattedMessage
              defaultMessage="Tags"
              description="Heading for the tags section in the evaluation run sidebar"
            />
          </Typography.Text>
          <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.xs, minWidth: 0, maxWidth: '100%' }}>
            <ClassNames>
              {({ css }) => {
                const tagLayoutClassName = css({ flex: '0 1 auto', minWidth: 0, maxWidth: '100%' });
                return tags.map((tag) => {
                  const tagText = tag.value ? `${tag.key}: ${tag.value}` : tag.key;
                  return (
                    <Tag
                      key={tag.key}
                      componentId="mlflow.eval_run_details.overview.tag"
                      className={tagLayoutClassName}
                    >
                      <TruncationTooltip componentId="mlflow.eval_run_details.overview.tag_tooltip">
                        <span style={TRUNCATED_TEXT_LAYOUT_STYLE}>{tagText}</span>
                      </TruncationTooltip>
                    </Tag>
                  );
                });
              }}
            </ClassNames>
          </div>
        </div>
      )}
    </aside>
  );
};
