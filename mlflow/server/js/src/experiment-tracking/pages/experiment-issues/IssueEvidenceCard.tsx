import {
  ArrowRightIcon,
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  CodeIcon,
  ModelsIcon,
  TableSkeleton,
  Typography,
  UserSparkleIcon,
  WrenchIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type {
  IssueEvidenceHighlightTone,
  IssueEvidenceStep,
  TraceIssueReference,
} from '../../../shared/web-shared/model-trace-explorer';
import { ModelSpanType } from '../../../shared/web-shared/model-trace-explorer';
import type { ReactNode } from 'react';
import { FormattedMessage, useIntl, type IntlShape } from 'react-intl';
import Utils from '../../../common/utils/Utils';
import type { IssueEvidenceDetail, IssueEvidenceHeader, SpanIdentity } from './issueEvidence.utils';
import { EVIDENCE_NO_ERROR, EVIDENCE_SPAN_SCOPE, MAX_EVIDENCE_FIELDS, toExplorerSpanId } from './issueEvidence.utils';

export interface IssueEvidenceCardProps {
  header: IssueEvidenceHeader;
  issueReference?: TraceIssueReference;
  spans?: SpanIdentity[];
  legacyDetail?: IssueEvidenceDetail;
  legacySpanId?: string;
  isLoadingDetail?: boolean;
  onViewTrace: (spanId?: string) => void;
  isExpanded: boolean;
  onToggleExpanded: () => void;
}

const getEvidenceTraceName = (
  header: IssueEvidenceHeader,
  issueReference: TraceIssueReference | undefined,
  spans: SpanIdentity[],
  legacySpanId: string | undefined,
  fallback: string,
): string => {
  if (header.name) {
    return header.name;
  }
  const spanId = issueReference?.spanId ?? issueReference?.evidence?.steps[0]?.spanId ?? legacySpanId;
  const normalizedSpanId = spanId ? toExplorerSpanId(spanId) : undefined;
  return (
    spans.find((span) => normalizedSpanId && toExplorerSpanId(span.id) === normalizedSpanId)?.name ??
    spans[0]?.name ??
    fallback
  );
};

/**
 * One synthetic step for a reference that carries no structured evidence: everything the trace
 * itself can say about the cited span, so an issue detected before this schema still shows a
 * walkthrough rather than a bare rationale.
 */
const toLegacyEvidenceStep = (
  detail: IssueEvidenceDetail | undefined,
  fallbackSpanId: string,
  intl: IntlShape,
): IssueEvidenceStep => {
  const lines: string[] = [];
  if (!detail) {
    lines.push(
      intl.formatMessage({
        defaultMessage: 'Evidence details could not be loaded for this trace.',
        description: "Experiment issues tab > Legacy evidence step > Trace details couldn't be loaded",
      }),
    );
  } else {
    if (detail.requestPreview) {
      lines.push(
        intl.formatMessage(
          {
            defaultMessage: 'Request: {value}',
            description: "Experiment issues tab > Legacy evidence step > Trace's request preview",
          },
          { value: detail.requestPreview },
        ),
      );
    }
    if (detail.responsePreview) {
      lines.push(
        intl.formatMessage(
          {
            defaultMessage: 'Response: {value}',
            description: "Experiment issues tab > Legacy evidence step > Trace's response preview",
          },
          { value: detail.responsePreview },
        ),
      );
    }
    if (detail.fields.length > 0) {
      const visibleFields = detail.fields
        .slice(0, MAX_EVIDENCE_FIELDS)
        .map(({ key, value }) => (key ? `${key}: ${value}` : value));
      const hiddenFieldCount = detail.fields.length - visibleFields.length;
      if (hiddenFieldCount > 0) {
        visibleFields.push(
          intl.formatMessage(
            {
              defaultMessage: '+{count} more',
              description: 'Experiment issues tab > Legacy evidence step > Count of hidden span input fields',
            },
            { count: hiddenFieldCount },
          ),
        );
      }
      lines.push(
        intl.formatMessage(
          {
            defaultMessage: 'Inputs:\n{fields}',
            description: "Experiment issues tab > Legacy evidence step > Span's input fields",
          },
          { fields: visibleFields.join('\n') },
        ),
      );
    }
    if (detail.errorScope !== EVIDENCE_NO_ERROR) {
      const errorText =
        detail.errorText ??
        (detail.errorScope === EVIDENCE_SPAN_SCOPE
          ? intl.formatMessage({
              defaultMessage: 'This span reported an error with no message or exception recorded.',
              description: "Experiment issues tab > Legacy evidence step > Span's error had no details",
            })
          : intl.formatMessage({
              defaultMessage: 'This trace reported an error with no message recorded.',
              description: "Experiment issues tab > Legacy evidence step > Trace's error had no details",
            }));
      lines.push(
        intl.formatMessage(
          {
            defaultMessage: 'Error: {value}',
            description: "Experiment issues tab > Legacy evidence step > Trace or span's error",
          },
          { value: errorText },
        ),
      );
    }
    if (lines.length === 0) {
      lines.push(
        intl.formatMessage({
          defaultMessage: 'No evidence details were recorded for this trace.',
          description: 'Experiment issues tab > Legacy evidence step > Trace has no displayable details',
        }),
      );
    }
  }

  return {
    spanId: detail?.spanId ?? fallbackSpanId,
    text: lines.join('\n\n'),
    highlights: detail?.errorText ? [detail.errorText] : [],
    highlightTone: detail && detail.errorScope !== EVIDENCE_NO_ERROR ? 'error' : 'warning',
  };
};

const HighlightedText = ({
  text,
  highlights,
  highlightTone,
}: {
  text: string;
  highlights: string[];
  highlightTone: IssueEvidenceHighlightTone;
}) => {
  const { theme } = useDesignSystemTheme();
  const highlightBackgroundColor =
    highlightTone === 'error'
      ? theme.isDarkMode
        ? theme.colors.red800
        : theme.colors.red200
      : highlightTone === 'success'
        ? theme.isDarkMode
          ? theme.colors.green800
          : theme.colors.green200
        : theme.isDarkMode
          ? theme.colors.yellow800
          : theme.colors.yellow200;
  const matches: Array<[number, number]> = [];
  for (const highlight of highlights) {
    if (!highlight) {
      continue;
    }
    let start = 0;
    while (start < text.length) {
      const index = text.indexOf(highlight, start);
      if (index < 0) {
        break;
      }
      matches.push([index, index + highlight.length]);
      start = index + highlight.length;
    }
  }
  matches.sort(([leftStart], [rightStart]) => leftStart - rightStart);
  const mergedMatches: Array<[number, number]> = [];
  for (const match of matches) {
    const previous = mergedMatches[mergedMatches.length - 1];
    if (previous && match[0] <= previous[1]) {
      previous[1] = Math.max(previous[1], match[1]);
    } else {
      mergedMatches.push([...match]);
    }
  }
  const content: ReactNode[] = [];
  let cursor = 0;
  mergedMatches.forEach(([start, end], index) => {
    if (start > cursor) {
      content.push(text.slice(cursor, start));
    }
    content.push(
      <span
        key={`${start}-${end}-${index}`}
        css={{
          backgroundColor: highlightBackgroundColor,
          color: 'inherit',
          borderRadius: theme.borders.borderRadiusSm,
          padding: `0 ${theme.spacing.xs / 2}px`,
        }}
      >
        {text.slice(start, end)}
      </span>,
    );
    cursor = end;
  });
  if (cursor < text.length) {
    content.push(text.slice(cursor));
  }
  return <>{content.length > 0 ? content : text}</>;
};

const EvidenceSpanIcon = ({ type }: { type?: string }) => {
  const { theme } = useDesignSystemTheme();
  const normalizedType = type?.toUpperCase();
  const isLlm = normalizedType === ModelSpanType.LLM || normalizedType === ModelSpanType.CHAT_MODEL;
  const isTool =
    normalizedType === ModelSpanType.TOOL ||
    normalizedType === ModelSpanType.FUNCTION ||
    normalizedType === 'BASH' ||
    normalizedType === 'READ' ||
    normalizedType === 'EDIT';
  const isConversation =
    normalizedType === 'CONVERSATION' ||
    normalizedType === ModelSpanType.CHAIN ||
    normalizedType === ModelSpanType.AGENT;
  const color = isLlm ? theme.colors.purple : isTool ? theme.colors.blue600 : theme.colors.textSecondary;
  const backgroundColor = isLlm
    ? theme.isDarkMode
      ? `${theme.colors.purple}55`
      : `${theme.colors.purple}33`
    : isTool
      ? theme.colors.blue100
      : theme.colors.backgroundSecondary;
  return (
    <span
      aria-hidden
      css={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
        width: theme.spacing.md,
        height: theme.spacing.md,
        borderRadius: theme.borders.borderRadiusSm,
        backgroundColor,
        color,
        fontSize: theme.typography.fontSizeSm,
      }}
    >
      {isLlm ? (
        <ModelsIcon css={{ color }} />
      ) : isTool ? (
        <WrenchIcon css={{ color }} />
      ) : isConversation ? (
        <UserSparkleIcon color="ai" />
      ) : (
        <CodeIcon css={{ color }} />
      )}
    </span>
  );
};

const IssueEvidenceStep = ({
  step,
  spans,
  onViewTrace,
}: {
  step: IssueEvidenceStep;
  spans: SpanIdentity[];
  onViewTrace: (spanId?: string) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const spanId = step.spanId;
  const normalizedSpanId = toExplorerSpanId(spanId);
  const span = spans.find((candidate) => toExplorerSpanId(candidate.id) === normalizedSpanId);
  const spanName = span?.name ?? spanId.slice(0, 8);
  const spanKind = span?.type;
  const mono = step.mono ?? false;

  return (
    <div
      css={{
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm - theme.spacing.xs / 2,
        padding: `${theme.spacing.sm + theme.spacing.xs / 2}px ${theme.spacing.mid}px`,
        backgroundColor: theme.colors.backgroundPrimary,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        // The span keeps its name; "View span" appears at the row's end while the step is hovered or focused.
        '& .issue-evidence-step-action': {
          opacity: 0,
          pointerEvents: 'none',
          transition: `opacity ${theme.animation.transitionDuration}ms ease`,
        },
        '&:hover .issue-evidence-step-action, &:focus-within .issue-evidence-step-action': {
          opacity: 1,
          pointerEvents: 'auto',
        },
      }}
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: theme.spacing.sm,
          minWidth: 0,
        }}
      >
        <span
          css={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: theme.spacing.sm - theme.spacing.xs / 2,
          }}
        >
          <EvidenceSpanIcon type={spanKind} />
          <Typography.Hint
            title={span?.name ?? spanId}
            css={{
              maxWidth: '24ch',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              fontFamily: 'monospace',
            }}
          >
            {spanName}
          </Typography.Hint>
        </span>
        <span className="issue-evidence-step-action">
          <Button
            componentId="mlflow.experiment-issues.evidence.view-span"
            type="tertiary"
            size="small"
            endIcon={<ArrowRightIcon />}
            css={{
              '--db-btn-height-small': `${theme.spacing.md + theme.spacing.xs}px`,
              '&& .anticon svg': {
                width: theme.typography.fontSizeSm,
                height: theme.typography.fontSizeSm,
              },
            }}
            onClick={() => onViewTrace(spanId)}
          >
            <FormattedMessage
              defaultMessage="View span"
              description="Experiment issues tab > Evidence step > Opens the span supporting this step"
            />
          </Button>
        </span>
      </div>
      <div
        css={{
          minWidth: 0,
          whiteSpace: 'pre-wrap',
          overflowWrap: 'anywhere',
          ...(mono
            ? {
                backgroundColor: theme.colors.backgroundSecondary,
                borderRadius: theme.borders.borderRadiusSm + theme.spacing.xs / 2,
                padding: `${theme.spacing.sm}px ${theme.spacing.sm + theme.spacing.xs / 2}px`,
              }
            : {}),
        }}
      >
        <Typography.Text size={mono ? 'sm' : undefined} css={mono ? { fontFamily: 'monospace' } : undefined}>
          <HighlightedText text={step.text} highlights={step.highlights} highlightTone={step.highlightTone} />
        </Typography.Text>
      </div>
    </div>
  );
};

const IssueEvidenceNarrative = ({
  summary,
  steps,
  spans,
  onViewTrace,
}: {
  summary?: string;
  steps: IssueEvidenceStep[];
  spans: SpanIdentity[];
  onViewTrace: (spanId?: string) => void;
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm + theme.spacing.xs / 2,
      }}
    >
      {summary && (
        <Typography.Text css={{ paddingLeft: theme.spacing.md, paddingRight: theme.spacing.md }}>
          {summary}
        </Typography.Text>
      )}
      {steps.length > 0 && (
        <div css={{ position: 'relative', paddingLeft: theme.spacing.md + theme.spacing.xs }}>
          <span
            aria-hidden
            css={{
              position: 'absolute',
              left: theme.spacing.sm - theme.spacing.xs / 4,
              top: theme.spacing.sm + theme.spacing.xs / 2,
              bottom: theme.spacing.sm + theme.spacing.xs / 2,
              width: 1.5,
              backgroundColor: theme.colors.border,
              borderRadius: theme.borders.borderRadiusFull,
            }}
          />
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            {steps.map((step, index) => (
              <div key={`${step.spanId}-${index}`} css={{ position: 'relative' }}>
                <span
                  aria-hidden
                  css={{
                    position: 'absolute',
                    left: -theme.spacing.md - theme.spacing.xs / 8,
                    top: theme.spacing.md - theme.spacing.xs / 4,
                    width: theme.spacing.sm,
                    height: theme.spacing.sm,
                    borderRadius: theme.borders.borderRadiusFull,
                    backgroundColor: theme.colors.backgroundPrimary,
                    border: `1.5px solid ${theme.colors.border}`,
                    zIndex: 1,
                  }}
                />
                <IssueEvidenceStep step={step} spans={spans} onViewTrace={onViewTrace} />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export const IssueEvidenceCard = ({
  header,
  issueReference,
  spans = [],
  legacyDetail,
  legacySpanId,
  isLoadingDetail = false,
  onViewTrace,
  isExpanded,
  onToggleExpanded,
}: IssueEvidenceCardProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const traceName = getEvidenceTraceName(
    header,
    issueReference,
    spans,
    legacySpanId,
    intl.formatMessage({
      defaultMessage: 'Trace',
      description: 'Experiment issues tab > Evidence card > Fallback trace name',
    }),
  );
  const assessmentSteps = issueReference?.evidence?.steps;
  const legacyStep =
    !assessmentSteps && legacySpanId ? toLegacyEvidenceStep(legacyDetail, legacySpanId, intl) : undefined;
  const expansionLabel = isExpanded
    ? intl.formatMessage(
        {
          defaultMessage: 'Collapse evidence for {traceName}',
          description: 'Experiment issues tab > Evidence card > Collapse evidence',
        },
        { traceName },
      )
    : intl.formatMessage(
        {
          defaultMessage: 'Expand evidence for {traceName}',
          description: 'Experiment issues tab > Evidence card > Expand evidence',
        },
        { traceName },
      );

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        minWidth: 0,
        padding: theme.spacing.sm + theme.spacing.xs / 2,
        backgroundColor: theme.colors.backgroundSecondary,
        borderRadius: theme.borders.borderRadiusMd + theme.spacing.xs / 2,
        '& .issue-evidence-trace-action': { display: 'none' },
        '&:hover .issue-evidence-trace-action, &:focus-within .issue-evidence-trace-action': {
          display: 'inline-flex',
        },
      }}
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.sm,
          minWidth: 0,
          paddingLeft: theme.spacing.xs / 2,
          paddingRight: theme.spacing.xs / 2,
        }}
      >
        <Button
          componentId="mlflow.experiment-issues.evidence.expand-toggle"
          type="tertiary"
          size="small"
          icon={
            isExpanded ? (
              <ChevronDownIcon css={{ color: theme.colors.textSecondary }} />
            ) : (
              <ChevronRightIcon css={{ color: theme.colors.textSecondary }} />
            )
          }
          css={{
            '--db-btn-height-small': `${theme.spacing.md + theme.spacing.xs}px`,
          }}
          aria-expanded={isExpanded}
          aria-label={expansionLabel}
          onClick={onToggleExpanded}
        />
        <Typography.Text
          bold
          title={traceName}
          css={{
            minWidth: 0,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {traceName}
        </Typography.Text>
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.sm + theme.spacing.xs,
            marginLeft: 'auto',
            flexShrink: 0,
          }}
        >
          {header.requestTimeMs !== undefined && (
            <Typography.Hint>{Utils.timeSinceStr(header.requestTimeMs)}</Typography.Hint>
          )}
          <span className="issue-evidence-trace-action">
            <Button
              componentId="mlflow.experiment-issues.evidence.view-trace"
              type="link"
              size="small"
              endIcon={<ArrowRightIcon />}
              css={{
                '&& .anticon svg': {
                  width: theme.typography.fontSizeSm,
                  height: theme.typography.fontSizeSm,
                },
              }}
              onClick={() => onViewTrace()}
            >
              <FormattedMessage
                defaultMessage="Open trace"
                description="Experiment issues tab > Evidence card > Opens the full trace"
              />
            </Button>
          </span>
        </div>
      </div>

      {isExpanded &&
        (assessmentSteps ? (
          <IssueEvidenceNarrative
            summary={issueReference?.rationale}
            steps={assessmentSteps}
            spans={spans}
            onViewTrace={onViewTrace}
          />
        ) : isLoadingDetail && legacySpanId ? (
          <TableSkeleton lines={2} />
        ) : legacyStep ? (
          <IssueEvidenceNarrative
            summary={issueReference?.rationale}
            steps={[legacyStep]}
            spans={spans}
            onViewTrace={onViewTrace}
          />
        ) : issueReference?.rationale ? (
          // A reference that named no span explains itself with its rationale alone.
          <IssueEvidenceNarrative
            summary={issueReference.rationale}
            steps={[]}
            spans={spans}
            onViewTrace={onViewTrace}
          />
        ) : (
          <Typography.Text color="secondary" size="sm">
            <FormattedMessage
              defaultMessage="No evidence details were recorded for this trace."
              description="Experiment issues tab > Evidence card > No assessment or legacy evidence"
            />
          </Typography.Text>
        ))}
    </div>
  );
};
