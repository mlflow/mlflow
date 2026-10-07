import { CopyIcon, useDesignSystemTheme } from '@databricks/design-system';
import { CodeSnippet } from '@databricks/web-shared/snippet';

import { CopyButton } from '../../shared/building_blocks/CopyButton';
import { overlayButtonStyles } from '../styles';

export type SnippetFormat = 'cli' | 'python';

export const CopyableSnippet = ({
  componentId,
  code,
  format,
  copyLabel,
}: {
  componentId: string;
  code: string;
  format: SnippetFormat;
  copyLabel: string;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ position: 'relative' }}>
      <CopyButton
        componentId={componentId}
        showLabel={false}
        copyText={code}
        icon={<CopyIcon />}
        aria-label={copyLabel}
        css={overlayButtonStyles(theme)}
      />
      <CodeSnippet
        language={format === 'python' ? 'python' : 'text'}
        theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
        style={{ padding: theme.spacing.sm, paddingRight: theme.spacing.xl + theme.spacing.sm }}
      >
        {code}
      </CodeSnippet>
    </div>
  );
};
