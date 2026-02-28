import { useState, useEffect } from 'react';
import {
  MarkdownIcon,
  SegmentedControlButton,
  SegmentedControlGroup,
  TextBoxIcon,
  Tooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { coy as lightStyle, atomDark as darkStyle } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import { getArtifactContent } from '../../../common/utils/ArtifactUtils';
import { ArtifactViewSkeleton } from './ArtifactViewSkeleton';
import { ArtifactViewErrorState } from './ArtifactViewErrorState';
import type { LoggedModelArtifactViewerProps } from './ArtifactViewComponents.types';
import { fetchArtifactUnified } from './utils/fetchArtifactUnified';
import { GenAIMarkdownRenderer } from '../../../shared/web-shared/genai-markdown-renderer';

interface ShowArtifactMarkdownViewProps extends LoggedModelArtifactViewerProps {
  runUuid: string;
  path: string;
}

const ShowArtifactMarkdownView = ({
  runUuid,
  path,
  isLoggedModelsMode,
  loggedModelId,
  experimentId,
  entityTags,
}: ShowArtifactMarkdownViewProps) => {
  const { theme } = useDesignSystemTheme();

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error>();
  const [mdContent, setMdContent] = useState<string>('');
  const [showRendered, setShowRendered] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(undefined);
    fetchArtifactUnified(
      {
        experimentId,
        runUuid,
        path,
        isLoggedModelsMode,
        loggedModelId,
        entityTags,
      },
      getArtifactContent,
    )
      .then((text) => {
        if (cancelled) return;
        setMdContent(text as string);
        setLoading(false);
      })
      .catch((error: Error) => {
        if (cancelled) return;
        setMdContent('');
        setError(error);
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [experimentId, runUuid, path, isLoggedModelsMode, loggedModelId, entityTags]);

  if (loading) {
    return <ArtifactViewSkeleton className="artifact-markdown-view-loading" />;
  }
  if (error) {
    return <ArtifactViewErrorState className="artifact-markdown-view-error" />;
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', width: '100%', height: '100%' }}>
      <div css={{ display: 'flex', padding: theme.spacing.xs }}>
        <SegmentedControlGroup
          name="markdown-render-mode"
          size="small"
          componentId="mlflow.artifact_view.markdown_render_mode"
          value={showRendered}
          onChange={(event) => setShowRendered(event.target.value)}
        >
          <SegmentedControlButton data-testid="markdown-view-source-button" value={false}>
            <Tooltip
              componentId="mlflow.artifact_view.markdown_source_tooltip"
              content={
                <FormattedMessage
                  defaultMessage="View source"
                  description="Tooltip for button that shows raw markdown source text"
                />
              }
            >
              <div css={{ display: 'flex', alignItems: 'center' }}>
                <TextBoxIcon css={{ fontSize: theme.typography.fontSizeLg }} />
              </div>
            </Tooltip>
          </SegmentedControlButton>
          <SegmentedControlButton data-testid="markdown-view-rendered-button" value>
            <Tooltip
              componentId="mlflow.artifact_view.markdown_rendered_tooltip"
              content={
                <FormattedMessage
                  defaultMessage="View rendered"
                  description="Tooltip for button that shows rendered markdown"
                />
              }
            >
              <div css={{ display: 'flex', alignItems: 'center' }}>
                <MarkdownIcon css={{ fontSize: theme.typography.fontSizeLg }} />
              </div>
            </Tooltip>
          </SegmentedControlButton>
        </SegmentedControlGroup>
      </div>
      <div css={{ flex: 1, overflow: 'auto', padding: theme.spacing.md }}>
        {showRendered ? (
          <GenAIMarkdownRenderer>{mdContent}</GenAIMarkdownRenderer>
        ) : (
          <SyntaxHighlighter
            language="markdown"
            style={theme.isDarkMode ? darkStyle : lightStyle}
            customStyle={{
              fontFamily: 'Source Code Pro, Menlo, monospace',
              fontSize: theme.typography.fontSizeMd,
              overflow: 'auto',
              marginTop: 0,
              width: '100%',
              height: '100%',
              padding: theme.spacing.xs,
              border: 'none',
            }}
          >
            {mdContent}
          </SyntaxHighlighter>
        )}
      </div>
    </div>
  );
};

export default ShowArtifactMarkdownView;
