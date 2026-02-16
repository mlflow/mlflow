import { HoverCard, Tag, Typography } from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';
import { getTraceTokenUsage, type ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';

import { NullCell } from './NullCell';
import { StackedComponents } from './StackedComponents';

export const TokensCell = (props: {
  currentTraceInfo?: ModelTraceInfoV3;
  otherTraceInfo?: ModelTraceInfoV3;
  isComparing: boolean;
}) => {
  const { currentTraceInfo, otherTraceInfo, isComparing } = props;

  const currentTokenUsage = currentTraceInfo ? getTraceTokenUsage(currentTraceInfo) : undefined;
  const otherTokenUsage = otherTraceInfo ? getTraceTokenUsage(otherTraceInfo) : undefined;

  return (
    <StackedComponents
      first={
        <TokenComponent
          inputTokens={currentTokenUsage?.input_tokens}
          outputTokens={currentTokenUsage?.output_tokens}
          totalTokens={currentTokenUsage?.total_tokens}
          isComparing={isComparing}
        />
      }
      second={
        isComparing && (
          <TokenComponent
            inputTokens={otherTokenUsage?.input_tokens}
            outputTokens={otherTokenUsage?.output_tokens}
            totalTokens={otherTokenUsage?.total_tokens}
            isComparing={isComparing}
          />
        )
      }
    />
  );
};

export const TokenComponent = ({
  inputTokens,
  outputTokens,
  totalTokens,
  isComparing,
}: {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  isComparing: boolean;
}) => {
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
