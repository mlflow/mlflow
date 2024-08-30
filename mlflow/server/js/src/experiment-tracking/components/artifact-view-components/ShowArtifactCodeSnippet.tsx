import React from 'react';
import { CopyIcon, useDesignSystemTheme } from '@databricks/design-system';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';

export const ShowArtifactCodeSnippet = ({ code }: { code: string }): React.ReactElement => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ position: 'relative' }}>
      <CopyButton
        css={{ zIndex: 1, position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs }}
        showLabel={false}
        copyText={code}
        icon={<CopyIcon />}
      />
      <CodeSnippet
        language="python"
        showLineNumbers={false}
        style={{
          padding: theme.spacing.sm,
          color: theme.colors.textPrimary,
          backgroundColor: theme.colors.backgroundSecondary,
          whiteSpace: 'pre-wrap',
        }}
        wrapLongLines
      >
        {code}
      </CodeSnippet>
    </div>
  );
};
