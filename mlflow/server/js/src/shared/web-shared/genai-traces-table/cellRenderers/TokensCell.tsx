import { HoverCard, Tag, Typography } from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';

import { NullCell } from './NullCell';
import { StackedComponents } from './StackedComponents';
import type { TraceInfoV3 } from '../types';

export const TokensCell = (props: {
  currentTraceInfo?: TraceInfoV3;
  otherTraceInfo?: TraceInfoV3;
  isComparing: boolean;
}) => {
  const { currentTraceInfo, otherTraceInfo, isComparing } = props;

  return (
    <StackedComponents
      first={<TokenComponent traceInfo={currentTraceInfo} isComparing={isComparing} />}
      second={isComparing && <TokenComponent traceInfo={otherTraceInfo} isComparing={isComparing} />}
    />
  );
};

const TokenComponent = (props: { traceInfo?: TraceInfoV3; isComparing: boolean }) => {
  const { traceInfo, isComparing } = props;

  const tokenUsage = traceInfo?.trace_metadata?.['mlflow.trace.tokenUsage'];
  const parsedTokenUsage = (() => {
    try {
      return tokenUsage ? JSON.parse(tokenUsage) : {};
    } catch {
      return {};
    }
  })();
  const totalTokens = parsedTokenUsage.total_tokens;
  const inputTokens = parsedTokenUsage.input_tokens;
  const outputTokens = parsedTokenUsage.output_tokens;

  const intl = useIntl();

  if (!totalTokens) {
    return <NullCell isComparing={isComparing} />;
  }

  return (
    <HoverCard
      trigger={
        <Tag css={{ width: 'fit-content', maxWidth: '100%' }} componentId="mlflow.genai-traces-table.tokens">
          <span
            css={{
              display: 'block',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {totalTokens}
          </span>
        </Tag>
      }
      content={
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {totalTokens && (
            <div
              css={{
                display: 'flex',
                flexDirection: 'row',
              }}
            >
              <div
                css={{
                  width: '35%',
                }}
              >
                <Typography.Text>
                  {intl.formatMessage({
                    defaultMessage: 'Total',
                    description: 'Label for the total tokensin the tooltip for the tokens cell.',
                  })}
                </Typography.Text>
              </div>
              <div>
                <Typography.Text color="secondary">{totalTokens}</Typography.Text>
              </div>
            </div>
          )}
          {inputTokens && (
            <div
              css={{
                display: 'flex',
                flexDirection: 'row',
              }}
            >
              <div
                css={{
                  width: '35%',
                }}
              >
                <Typography.Text>
                  {intl.formatMessage({
                    defaultMessage: 'Input',
                    description: 'Label for the input tokens in the tooltip for the tokens cell.',
                  })}
                </Typography.Text>
              </div>
              <div>
                <Typography.Text color="secondary">{inputTokens}</Typography.Text>
              </div>
            </div>
          )}
          {outputTokens && (
            <div
              css={{
                display: 'flex',
                flexDirection: 'row',
              }}
            >
              <div
                css={{
                  width: '35%',
                }}
              >
                <Typography.Text>
                  {intl.formatMessage({
                    defaultMessage: 'Output',
                    description: 'Label for the output tokens in the tooltip for the tokens cell.',
                  })}
                </Typography.Text>
              </div>
              <div>
                <Typography.Text color="secondary">{outputTokens}</Typography.Text>
              </div>
            </div>
          )}
        </div>
      }
    />
  );
};
