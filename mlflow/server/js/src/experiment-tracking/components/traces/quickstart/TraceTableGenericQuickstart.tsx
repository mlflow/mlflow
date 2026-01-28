import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { CodeSnippet, SnippetCopyAction } from '@databricks/web-shared/snippet';

import { QUICKSTART_CONTENT } from './TraceTableQuickstart.utils';

export const TraceTableGenericQuickstart = ({
  flavorName,
  baseComponentId,
}: {
  flavorName: keyof typeof QUICKSTART_CONTENT;
  baseComponentId: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const { getContent, getCodeSource, minVersion } = QUICKSTART_CONTENT[flavorName];
  const content = getContent(baseComponentId);
  const code = getCodeSource();
  return (
    <div>
      <Typography.Paragraph color="secondary" css={{ maxWidth: 600 }}>
        {content}
      </Typography.Paragraph>
      <div css={{ position: 'relative' }}>
        <SnippetCopyAction
          componentId="mlflow.traces.empty_state.example_code_copy"
          css={{ position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs }}
          copyText={code}
        />
        <CodeSnippet language="python" showLineNumbers>
          {code}
        </CodeSnippet>
      </div>
    </div>
  );
};
