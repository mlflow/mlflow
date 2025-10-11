import { CopyIcon, Typography, useDesignSystemTheme, Alert } from '@databricks/design-system';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';

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
      <Typography.Text css={{ maxWidth: 800 }} color="secondary">
        {content}
      </Typography.Text>
      <div css={{ position: 'relative', width: 'min-content' }}>
        <CopyButton
          componentId={`${baseComponentId}.traces_table.${flavorName}_quickstart_snippet_copy`}
          css={{ zIndex: 1, position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs }}
          showLabel={false}
          copyText={code}
          icon={<CopyIcon />}
        />
        <CodeSnippet
          showLineNumbers
          theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
          style={{
            padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
            marginTop: theme.spacing.md,
          }}
          language="python"
        >
          {code}
        </CodeSnippet>
      </div>
    </div>
  );
};
