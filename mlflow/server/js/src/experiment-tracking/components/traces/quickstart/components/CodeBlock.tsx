import { CopyIcon, type ThemeType } from '@databricks/design-system';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';

export const CodeBlock = ({
  theme,
  code,
  language,
  componentId,
}: {
  theme: ThemeType;
  code: string;
  language: string;
  componentId: string;
}) => {
  return (
    <div
      css={{
        position: 'relative',
        borderRadius: theme.borders.borderRadiusMd,
        overflow: 'hidden',
        border: `1px solid ${theme.colors.border}`,
      }}
    >
      <CopyButton
        componentId={componentId}
        css={{ zIndex: 1, position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs }}
        showLabel={false}
        copyText={code}
        icon={<CopyIcon />}
      />
      <CodeSnippet
        showLineNumbers
        language={language === 'bash' ? 'text' : language === 'typescript' ? 'javascript' : 'python'}
        theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
        style={{
          fontSize: 12,
          overflow: 'auto',
        }}
      >
        {code}
      </CodeSnippet>
    </div>
  );
};
