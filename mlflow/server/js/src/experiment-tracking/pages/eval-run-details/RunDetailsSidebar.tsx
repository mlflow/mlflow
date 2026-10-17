import { ClassNames } from '@emotion/react';
import type { ReactNode } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import {
  InfoTooltip,
  Tag,
  TruncationTooltip,
  Typography,
  UserIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';

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
  stats: RunDetailsStats;
  scorers: RunDetailsScorer[];
  tags: { key: string; value: string }[];
}

export interface RunDetailsScorer {
  name: string;
  displayName: string;
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

const TruncatedSidebarText = ({ text }: { text: string }) => (
  <span style={TRUNCATED_TEXT_LAYOUT_STYLE} title={text}>
    <Typography.Text size="md">{text}</Typography.Text>
  </span>
);

const ScorerSidebarValue = ({ name }: { name: string }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <span
      css={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: theme.spacing.xs,
        minWidth: 0,
        maxWidth: '100%',
        color: theme.colors.textPrimary,
      }}
    >
      <span style={TRUNCATED_TEXT_LAYOUT_STYLE} title={name}>
        {name}
      </span>
    </span>
  );
};

export const RunDetailsSidebar = ({
  createdBy,
  createdAt,
  dataset,
  endpoint,
  stats,
  scorers,
  tags,
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
          {dataset?.name ? (
            <TruncatedSidebarText text={dataset.name} />
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
      </div>

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
          <Typography.Text bold size="lg">
            <FormattedMessage
              defaultMessage="Scorers ({count})"
              description="Heading for the list of scorers in the evaluation run sidebar, with a count"
              values={{ count: scorers.length }}
            />
          </Typography.Text>
          {scorers.map((scorer) => (
            <ScorerSidebarValue key={scorer.name} name={scorer.displayName} />
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
