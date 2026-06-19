import { CopyIcon, useDesignSystemTheme } from '@databricks/design-system';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { CopyButton } from '../../../../shared/building_blocks/CopyButton';
import { prettyPrintJson } from '../utils';

interface Props {
  /** Raw JSON string (e.g. tool-call arguments or a JSON-format reply). */
  raw: string;
}

/**
 * Renders a JSON string as a syntax-highlighted code block with line numbers and
 * a copy button. The JSON is pretty-printed when parseable and shown verbatim
 * otherwise. The snippet background is transparent so the surrounding assistant
 * card background shows through (the shared `CodeSnippet` prism theme would
 * otherwise paint its own opaque background).
 */
export const JsonCodeBlock = ({ raw }: Props) => {
  const { theme } = useDesignSystemTheme();
  const code = prettyPrintJson(raw);

  return (
    <div css={{ position: 'relative' }} data-testid="mlflow.playground.json_code_block">
      <div css={{ position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs, zIndex: 1 }}>
        <CopyButton
          componentId="mlflow.playground.json_code_block.copy"
          copyText={code}
          showLabel={false}
          size="small"
          icon={<CopyIcon />}
        />
      </div>
      <CodeSnippet
        theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
        language="json"
        showLineNumbers
        style={{
          margin: 0,
          padding: 0,
          // Transparent so the assistant card background shows through instead of
          // the prism theme's own (dark "purple") background.
          backgroundColor: 'transparent',
          // Cap the height so a large JSON payload scrolls within the card instead
          // of stretching it; the prism theme already sets `overflow: auto` on the
          // <pre>. Mirrors GenAIMarkdownRenderer's code-block max height.
          maxHeight: 640,
        }}
      >
        {code}
      </CodeSnippet>
    </div>
  );
};
