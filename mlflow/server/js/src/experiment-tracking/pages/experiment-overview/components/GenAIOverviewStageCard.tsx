import type { ReactNode } from 'react';
import { Button, Spinner, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useNavigate } from '../../../../common/utils/RoutingUtils';
import type { GenAIOverviewStageStatus } from '../genAIOverview.types';

const OVERVIEW_STAGE_NARROW_MEDIA_QUERY = '@media (max-width: 800px)';
const OVERVIEW_STAGE_ICON_CLASS_NAME = 'genai-overview-stage-icon';

export interface GenAIOverviewStageCardProps {
  componentId: string;
  icon: ReactNode;
  label: ReactNode;
  caption: ReactNode;
  description: ReactNode;
  status: GenAIOverviewStageStatus;
  recommended?: boolean;
  headline?: ReactNode;
  captionControl?: ReactNode;
  artifactHref?: string;
  artifactLabel?: ReactNode;
  artifactAriaLabel?: string;
  activityChart?: ReactNode;
  content?: ReactNode;
  action?: ReactNode;
  expanded?: boolean;
}

const MiniActivityChart = ({ values }: { values: number[] }) => {
  const { theme } = useDesignSystemTheme();
  if (!values.some((value) => value > 0)) return <div css={{ height: 52 }} aria-hidden />;
  const maximum = Math.max(...values, 1);
  return (
    <div css={{ display: 'flex', alignItems: 'flex-end', gap: 3, height: 52 }} aria-hidden>
      {values.map((value, index) => {
        const height = value > 0 ? Math.max((value / maximum) * 100, 8) : 0;
        return (
          <div
            key={index}
            css={{
              flex: 1,
              minWidth: 3,
              height: `${height}%`,
              borderRadius: `${theme.borders.borderRadiusSm}px ${theme.borders.borderRadiusSm}px 0 0`,
              backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
              opacity: 0.55,
            }}
          />
        );
      })}
    </div>
  );
};

export const GenAIOverviewStageCard = ({
  componentId,
  icon,
  label,
  caption,
  description,
  status,
  recommended = false,
  headline,
  captionControl,
  artifactHref,
  artifactLabel,
  artifactAriaLabel,
  activityChart,
  content,
  action,
  expanded = false,
}: GenAIOverviewStageCardProps) => {
  const { theme } = useDesignSystemTheme();
  const navigate = useNavigate();
  const isReady = status.status === 'ready';
  const activity = isReady ? status.activity.map(({ count }) => count) : [];
  const artifactAction = isReady && artifactHref && artifactLabel && (
    <Button
      componentId={`${componentId}.view-artifacts`}
      size="small"
      aria-label={artifactAriaLabel}
      onClick={() => navigate(artifactHref)}
    >
      {artifactLabel}
    </Button>
  );
  const renderedAction = status.status === 'loading' ? null : (action ?? artifactAction);
  const hasHoverTreatment = Boolean(renderedAction);

  return (
    <article
      css={{
        display: 'grid',
        gridTemplateColumns: expanded
          ? 'minmax(0, 1fr)'
          : isReady
            ? renderedAction
              ? 'minmax(200px, 240px) minmax(0, 1fr) auto'
              : 'minmax(200px, 240px) minmax(0, 1fr)'
            : 'minmax(0, 1fr) auto',
        alignItems: 'center',
        gap: theme.spacing.md,
        position: expanded ? 'relative' : undefined,
        minHeight: expanded
          ? theme.spacing.xl * 3
          : isReady
            ? theme.spacing.xl * 3
            : theme.spacing.xl * 2 + theme.spacing.mid,
        padding: expanded ? theme.spacing.md : `${theme.spacing.sm}px ${theme.spacing.lg}px`,
        backgroundColor: theme.colors.backgroundPrimary,
        '&:hover': hasHoverTreatment ? { backgroundColor: theme.colors.actionDefaultBackgroundHover } : undefined,
        [`&:hover .${OVERVIEW_STAGE_ICON_CLASS_NAME}`]: hasHoverTreatment
          ? {
              color: theme.colors.blue500,
              backgroundColor: theme.colors.backgroundPrimary,
            }
          : undefined,
        [OVERVIEW_STAGE_NARROW_MEDIA_QUERY]: {
          gridTemplateColumns: 'minmax(0, 1fr)',
          gap: theme.spacing.md,
          padding: theme.spacing.md,
        },
      }}
    >
      <div
        css={{
          display: 'flex',
          flexDirection: expanded ? 'column' : 'row',
          alignItems: expanded ? 'stretch' : 'center',
          gap: theme.spacing.md,
          minWidth: 0,
        }}
      >
        <div
          css={{
            display: 'flex',
            alignItems: expanded ? 'flex-start' : 'center',
            gap: expanded ? theme.spacing.sm : theme.spacing.md,
            minWidth: 0,
            paddingRight: expanded && renderedAction ? theme.spacing.xl * 5 + theme.spacing.md : undefined,
          }}
        >
          <div
            className={OVERVIEW_STAGE_ICON_CLASS_NAME}
            aria-hidden="true"
            css={{
              display: 'flex',
              flex: `0 0 ${expanded ? theme.spacing.lg : theme.spacing.xl}px`,
              width: expanded ? theme.spacing.lg : theme.spacing.xl,
              height: expanded ? theme.spacing.lg : theme.spacing.xl,
              alignItems: 'center',
              justifyContent: 'center',
              borderRadius: theme.borders.borderRadiusMd,
              color: theme.colors.textSecondary,
              backgroundColor: theme.colors.backgroundSecondary,
              fontSize: expanded ? theme.typography.fontSizeMd : theme.typography.fontSizeLg,
              transition: 'color 120ms ease, background-color 120ms ease',
            }}
          >
            {icon}
          </div>
          <div
            css={{
              display: 'flex',
              flex: 1,
              flexDirection: 'column',
              gap: theme.spacing.xs,
              minWidth: 0,
            }}
          >
            <div css={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: theme.spacing.sm }}>
              <Typography.Text
                color={expanded ? 'primary' : 'secondary'}
                size="sm"
                bold
                css={{
                  textTransform: 'uppercase',
                }}
              >
                {label}
              </Typography.Text>
              {recommended && (
                <Tag componentId={`${componentId}.recommended`} color="indigo">
                  <FormattedMessage defaultMessage="Recommended" description="Recommended overview stage badge" />
                </Tag>
              )}
            </div>
            {status.status === 'loading' ? (
              <Spinner size="small" />
            ) : isReady ? (
              <>
                <div
                  css={{
                    display: 'flex',
                    alignItems: 'baseline',
                    gap: theme.spacing.sm,
                    fontSize: theme.typography.fontSizeXl,
                    lineHeight: 1.1,
                    fontWeight: theme.typography.typographyBoldFontWeight,
                    color: theme.colors.textPrimary,
                  }}
                >
                  {headline ?? status.totalCount.toLocaleString()}
                </div>
                {captionControl ?? (
                  <Typography.Text
                    color="secondary"
                    size="sm"
                    ellipsis
                    title={typeof caption === 'string' ? caption : undefined}
                  >
                    {caption}
                  </Typography.Text>
                )}
              </>
            ) : (
              <Typography.Text color="secondary">{description}</Typography.Text>
            )}
          </div>
        </div>
        {expanded && content}
      </div>
      {isReady && (
        <div
          css={{
            width: '70%',
            minWidth: 0,
            justifySelf: 'start',
            marginLeft: '10%',
            [OVERVIEW_STAGE_NARROW_MEDIA_QUERY]: {
              width: '100%',
              justifySelf: 'stretch',
              marginLeft: 0,
            },
          }}
        >
          {activityChart ?? <MiniActivityChart values={activity} />}
        </div>
      )}
      {renderedAction && (
        <div
          css={{
            justifySelf: 'end',
            whiteSpace: 'nowrap',
            position: expanded ? 'absolute' : undefined,
            top: expanded ? theme.spacing.md : undefined,
            right: expanded ? theme.spacing.md : undefined,
            [OVERVIEW_STAGE_NARROW_MEDIA_QUERY]: { justifySelf: 'start' },
          }}
        >
          {renderedAction}
        </div>
      )}
    </article>
  );
};

export const GenAIOverviewCompactStageCard = ({
  icon,
  label,
  description,
}: {
  icon: ReactNode;
  label: ReactNode;
  description: ReactNode;
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <article
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.xs,
        minWidth: 0,
      }}
    >
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, minWidth: 0 }}>
        <div
          aria-hidden="true"
          css={{
            display: 'flex',
            flex: `0 0 ${theme.spacing.lg}px`,
            width: theme.spacing.lg,
            height: theme.spacing.lg,
            alignItems: 'center',
            justifyContent: 'center',
            borderRadius: theme.borders.borderRadiusMd,
            color: theme.colors.textSecondary,
            backgroundColor: theme.colors.backgroundSecondary,
            fontSize: theme.typography.fontSizeMd,
          }}
        >
          {icon}
        </div>
        <Typography.Text
          color="primary"
          size="sm"
          bold
          css={{
            textTransform: 'uppercase',
          }}
        >
          {label}
        </Typography.Text>
      </div>
      <Typography.Text color="secondary" size="sm">
        {description}
      </Typography.Text>
    </article>
  );
};
