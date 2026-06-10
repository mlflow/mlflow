import React, { Component } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { coy as style, atomDark as darkStyle } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import { getExtension, getLanguage } from '../../../common/utils/FileUtils';
import { getArtifactContent } from '../../../common/utils/ArtifactUtils';
import './ShowArtifactTextView.css';
import type { DesignSystemHocProps } from '@databricks/design-system';
import { WithDesignSystemThemeHoc } from '@databricks/design-system';
import { ArtifactViewSkeleton } from './ArtifactViewSkeleton';
import { ArtifactViewErrorState } from './ArtifactViewErrorState';
import type { LoggedModelArtifactViewerProps } from './ArtifactViewComponents.types';
import { fetchArtifactUnified } from './utils/fetchArtifactUnified';
import { virtualizedRenderer } from './artifactTextVirtualizedRenderer';

const LARGE_ARTIFACT_SIZE = 100 * 1024;
export const VERY_LARGE_ARTIFACT_SIZE = 5 * 1024 * 1024;
// react-syntax-highlighter's processLines() flattens its line array with [].concat(...lines), which
// exceeds the JS engine's max-arguments limit (~130K) and crashes. Byte size is only a loose proxy for
// line count, so a small-but-line-heavy file can crash below VERY_LARGE_ARTIFACT_SIZE. Gate on line
// count too, well below the crash limit, so the virtualized renderer also catches those files.
export const VERY_LARGE_ARTIFACT_LINE_COUNT = 50 * 1000;

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
      const isLargeFile = (this.props.size || 0) > LARGE_ARTIFACT_SIZE;
      const language = isLargeFile ? 'text' : getLanguage(this.props.path);
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
      const renderedContent = this.state.text
        ? prettifyArtifactText(language, this.state.text, this.props.path)
        : this.state.text;

      const syntaxStyle = theme.isDarkMode ? darkStyle : style;

      const isVeryLargeFile =
        (this.props.size || 0) > VERY_LARGE_ARTIFACT_SIZE ||
        countLines(renderedContent) > VERY_LARGE_ARTIFACT_LINE_COUNT;

      return (
        <div className="mlflow-ShowArtifactPage">
          <div className="text-area-border-box">
            <SyntaxHighlighter
              language={language}
              style={syntaxStyle}
              customStyle={overrideStyles}
              renderer={isVeryLargeFile ? virtualizedRenderer() : undefined}
            >
              {renderedContent ?? ''}
            </SyntaxHighlighter>
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

function countLines(text?: string) {
  if (!text) {
    return 0;
  }
  let count = 1;
  let index = text.indexOf('\n');
  while (index !== -1) {
    count += 1;
    index = text.indexOf('\n', index + 1);
  }
  return count;
}

export function prettifyArtifactText(language: string, rawText: string, path?: string) {
  if (path && getExtension(path).toLowerCase() === 'jsonl') {
    return rawText;
  }
  if (language === 'json') {
    try {
      const parsedJson = JSON.parse(rawText);
      return JSON.stringify(parsedJson, null, 2);
    } catch (e) {
      return rawText;
    }
  }
  return rawText;
}
export default React.memo(WithDesignSystemThemeHoc(ShowArtifactTextView));
