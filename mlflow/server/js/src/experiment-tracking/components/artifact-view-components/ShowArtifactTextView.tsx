import React, { Component } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { coy as style, atomDark as darkStyle } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import { getLanguage } from '../../../common/utils/FileUtils';
import { getArtifactContent } from '../../../common/utils/ArtifactUtils';
import './ShowArtifactTextView.css';
import type { DesignSystemHocProps } from '@databricks/design-system';
import { WithDesignSystemThemeHoc } from '@databricks/design-system';
import { ArtifactViewSkeleton } from './ArtifactViewSkeleton';
import { ArtifactViewErrorState } from './ArtifactViewErrorState';
import type { LoggedModelArtifactViewerProps } from './ArtifactViewComponents.types';
import { fetchArtifactUnified } from './utils/fetchArtifactUnified';

const LARGE_ARTIFACT_SIZE = 1024 * 1024;
// Beyond this size, skip the syntax highlighter entirely and render as
// plain text in a <pre> block. Prism creates a DOM node per token, so
// multi-megabyte files can freeze the browser tab.
const MAX_HIGHLIGHTER_SIZE = 1024 * 1024;

type SyntaxHighlighterErrorBoundaryProps = {
  fallback: React.ReactNode;
  children: React.ReactNode;
};

type SyntaxHighlighterErrorBoundaryState = {
  hasError: boolean;
};

/**
 * Catches render errors from the syntax highlighter (e.g. files with
 * problematic content or that are too large for Prism to tokenize) and
 * falls back to a plain-text rendering instead of crashing the page.
 */
export class SyntaxHighlighterErrorBoundary extends Component<
  SyntaxHighlighterErrorBoundaryProps,
  SyntaxHighlighterErrorBoundaryState
> {
  state: SyntaxHighlighterErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback;
    }
    return this.props.children;
  }
}

type Props = DesignSystemHocProps & {
  runUuid: string;
  path: string;
  size?: number;
  getArtifact?: (...args: any[]) => any;
} & LoggedModelArtifactViewerProps;

type State = {
  loading?: boolean;
  error?: Error;
  text?: string;
  path?: string;
};

class ShowArtifactTextView extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.fetchArtifacts = this.fetchArtifacts.bind(this);
  }

  static defaultProps = {
    getArtifact: fetchArtifactUnified,
  };

  state = {
    loading: true,
    error: undefined,
    text: undefined,
    path: undefined,
  };

  componentDidMount() {
    this.fetchArtifacts();
  }

  componentDidUpdate(prevProps: Props) {
    if (this.props.path !== prevProps.path || this.props.runUuid !== prevProps.runUuid) {
      this.fetchArtifacts();
    }
  }

  render() {
    if (this.state.loading || this.state.path !== this.props.path) {
      return <ArtifactViewSkeleton className="artifact-text-view-loading" />;
    }
    if (this.state.error) {
      return <ArtifactViewErrorState className="artifact-text-view-error" />;
    } else {
      const textLength = this.state.text ? (this.state.text as string).length : 0;
      const isVeryLargeFile = textLength > MAX_HIGHLIGHTER_SIZE;
      const isLargeFile = (this.props.size || 0) > LARGE_ARTIFACT_SIZE;
      const language = isLargeFile || isVeryLargeFile ? 'text' : getLanguage(this.props.path);
      const { theme } = this.props.designSystemThemeApi;

      const overrideStyles = {
        fontFamily: 'Source Code Pro,Menlo,monospace',
        fontSize: theme.typography.fontSizeMd,
        overflow: 'auto',
        marginTop: '0',
        width: '100%',
        height: '100%',
        padding: theme.spacing.xs,
        borderColor: theme.colors.borderDecorative,
        border: 'none',
      };
      const renderedContent = this.state.text ? prettifyArtifactText(language, this.state.text) : this.state.text;

      const plainTextFallback = (
        <pre
          style={{
            ...overrideStyles,
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-all',
            color: theme.colors.textPrimary,
            backgroundColor: theme.colors.backgroundPrimary,
            margin: 0,
          }}
        >
          {renderedContent ?? ''}
        </pre>
      );

      if (isVeryLargeFile) {
        return (
          <div className="mlflow-ShowArtifactPage">
            <div className="text-area-border-box">{plainTextFallback}</div>
          </div>
        );
      }

      const syntaxStyle = theme.isDarkMode ? darkStyle : style;

      return (
        <div className="mlflow-ShowArtifactPage">
          <div className="text-area-border-box">
            <SyntaxHighlighterErrorBoundary fallback={plainTextFallback}>
              <SyntaxHighlighter language={language} style={syntaxStyle} customStyle={overrideStyles}>
                {renderedContent ?? ''}
              </SyntaxHighlighter>
            </SyntaxHighlighterErrorBoundary>
          </div>
        </div>
      );
    }
  }

  /** Fetches artifacts and updates component state with the result */
  fetchArtifacts() {
    this.setState({ loading: true });
    const { isLoggedModelsMode, loggedModelId, path, runUuid, experimentId, entityTags } = this.props;

    this.props
      .getArtifact?.({ isLoggedModelsMode, loggedModelId, path, runUuid, experimentId, entityTags }, getArtifactContent)
      .then((text: string) => {
        this.setState({ text: text, loading: false });
      })
      .catch((error: Error) => {
        this.setState({ error: error, loading: false });
      });
    this.setState({ path: this.props.path });
  }
}

export function prettifyArtifactText(language: string, rawText: string) {
  if (language === 'json') {
    try {
      const parsedJson = JSON.parse(rawText);
      return JSON.stringify(parsedJson, null, 2);
    } catch (e) {
      // No-op
    }
    return rawText;
  }
  return rawText;
}
export default React.memo(WithDesignSystemThemeHoc(ShowArtifactTextView));
