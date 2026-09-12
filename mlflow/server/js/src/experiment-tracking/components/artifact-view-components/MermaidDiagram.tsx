import { useEffect, useState, type ReactNode } from 'react';
import { Alert, Spinner, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import sanitize from '../../../shared/web-shared/html-content/sanitize';
import { CodeBlock } from '../../../shared/web-shared/genai-markdown-renderer/GenAIMarkdownRenderer';

let diagramId = 0;

const MermaidDiagram = ({ children }: { children?: ReactNode }) => {
  const source = String(children).replace(/\n$/, '');
  const { theme } = useDesignSystemTheme();
  const [id] = useState(() => `mlflow-mermaid-${++diagramId}`);
  const [svg, setSvg] = useState<string>();
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setSvg(undefined);
    setHasError(false);

    import('mermaid')
      .then(async ({ default: mermaid }) => {
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: 'strict',
          secure: [
            'securityLevel',
            'secure',
            'startOnLoad',
            'maxTextSize',
            'maxEdges',
            'suppressErrorRendering',
            'htmlLabels',
            'dompurifyConfig',
            'theme',
            'themeCSS',
            'themeVariables',
            'darkMode',
            'fontFamily',
            'altFontFamily',
          ],
          htmlLabels: false,
          suppressErrorRendering: true,
          theme: theme.isDarkMode ? 'dark' : 'default',
        });
        return mermaid.render(id, source);
      })
      .then(({ svg }) => {
        if (!cancelled) {
          setSvg(sanitize(svg));
        }
      })
      .catch(() => {
        if (!cancelled) {
          setHasError(true);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [id, source, theme.isDarkMode]);

  if (hasError) {
    return (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        <Alert
          componentId="mlflow.artifact_view.mermaid_render_error"
          type="error"
          closable={false}
          message={
            <FormattedMessage
              defaultMessage="Unable to render Mermaid diagram. Showing source instead."
              description="Error shown when a Mermaid diagram in a Markdown artifact cannot be rendered"
            />
          }
        />
        <CodeBlock language="text">{source}</CodeBlock>
      </div>
    );
  }

  if (!svg) {
    return (
      <div css={{ display: 'flex', justifyContent: 'center', padding: theme.spacing.md }}>
        <Spinner size="small" />
      </div>
    );
  }

  return (
    <div
      data-testid="mermaid-diagram"
      css={{
        display: 'flex',
        justifyContent: 'center',
        maxWidth: '100%',
        '& svg': { maxWidth: '100%', height: 'auto' },
      }}
      // eslint-disable-next-line react/no-danger
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
};

export default MermaidDiagram;
