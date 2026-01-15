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
    <div css={{ maxWidth: 600 }}>
      <Typography.Text color="secondary">
        {content}
      </Typography.Text>
      <div css={{ position: 'relative', maxWidth: '100%' }}>
        <CopyButton
          componentId="codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_components_traces_quickstart_tracetablegenericquickstart_25"
          css={{ zIndex: 1, position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs }}
          showLabel={false}
          copyText={code}
          icon={<CopyIcon />}
        />
        <CodeSnippet showLineNumbers language="python" theme={theme.isDarkMode ? 'duotoneDark' : 'light'}>
          {code}
        </CodeSnippet>
      </div>
    </div>
  );
};
