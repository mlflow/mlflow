import { CopyIcon, Typography, useDesignSystemTheme, Alert } from '@databricks/design-system';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';

import { QUICKSTART_CONTENT } from './TraceTableQuickstart.utils';
import { FormattedMessage } from 'react-intl';
import { useTracesViewTableNoTracesQuickstartContext } from './TracesViewTableNoTracesQuickstartContext';

export const TraceTableGenericQuickstart = ({
  flavorName,
  baseComponentId,
}: {
  flavorName: keyof typeof QUICKSTART_CONTENT;
  baseComponentId: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const { getContent, getCodeSource, minVersion } = QUICKSTART_CONTENT[flavorName];
  const { displayVersionWarnings = true } = useTracesViewTableNoTracesQuickstartContext();
  const content = getContent(baseComponentId);
  const versionCheck = `import mlflow
from packaging.version import Version

assert Version(mlflow.__version__) >= Version("${minVersion}"), (
  "This feature requires MLflow version ${minVersion} or newer. "
  "Please run '%pip install -U mlflow' in a notebook cell, "
  "and restart the kernel when the command finishes."
)

`;

  const code = versionCheck + getCodeSource();

  const alertContent = (
    <FormattedMessage
      defaultMessage="This example requires MLflow version {minVersion} or newer. Please run {installCommand} in a notebook cell if your MLflow version is older than this, and restart the kernel when the command finishes."
      description="Alert description informing the user of how to upgrade MLflow to the minimum required version"
      values={{
        minVersion,
        installCommand: <Typography.Text code>%pip install -U mlflow</Typography.Text>,
      }}
    />
  );

  return (
    <div>
      {displayVersionWarnings && (
        <Alert
          componentId={`${baseComponentId}.traces_table.${flavorName}_quickstart_alert`}
          css={{ marginBottom: theme.spacing.md, maxWidth: 800 }}
          closable={false}
          message={
            <FormattedMessage
              defaultMessage="Requires MLflow >= {minVersion}"
              description="Alert title informing the user of the minimum required MLflow version to run the code example"
              values={{ minVersion }}
            />
          }
          description={alertContent}
          type="info"
        />
      )}
      <Typography.Text css={{ maxWidth: 800 }}>{content}</Typography.Text>
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
