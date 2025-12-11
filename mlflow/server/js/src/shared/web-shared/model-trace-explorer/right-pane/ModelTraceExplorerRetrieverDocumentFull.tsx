import { Button, ChevronUpIcon, FileDocumentIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { GenAIMarkdownRenderer } from '../../genai-markdown-renderer';
import { KeyValueTag } from '../key-value-tag/KeyValueTag';

export function ModelTraceExplorerRetrieverDocumentFull({
  text,
  metadataTags,
  setExpanded,
  logDocumentClick,
}: {
  text: string;
  metadataTags: { key: string; value: string }[];
  setExpanded: (expanded: boolean) => void;
  logDocumentClick?: (action: string) => void;
}) {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', flexDirection: 'column' }}>
      <div
        role="button"
        onClick={() => {
          setExpanded(false);
          logDocumentClick?.('collapse');
        }}
        css={{
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'center',
          cursor: 'pointer',
          padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
          height: theme.typography.lineHeightBase,
          boxSizing: 'content-box',
          '&:hover': {
            backgroundColor: theme.colors.backgroundSecondary,
          },
        }}
      >
        <FileDocumentIcon />
      </div>
      <div css={{ padding: theme.spacing.md, paddingBottom: 0 }}>
        <GenAIMarkdownRenderer>{text}</GenAIMarkdownRenderer>
      </div>
      <div css={{ padding: theme.spacing.md, paddingTop: 0 }}>
        {metadataTags.map(({ key, value }) => (
          <KeyValueTag key={key} itemKey={key} itemValue={value} />
        ))}
      </div>
      <Button
        css={{ width: '100%', padding: theme.spacing.sm }}
        componentId="shared.model-trace-explorer.retriever-document-collapse"
        icon={<ChevronUpIcon />}
        type="tertiary"
        onClick={() => setExpanded(false)}
      >
        <FormattedMessage
          defaultMessage="See less"
          description="Model trace explorer > selected span > code snippet > see less button"
        />
      </Button>
    </div>
  );
}
