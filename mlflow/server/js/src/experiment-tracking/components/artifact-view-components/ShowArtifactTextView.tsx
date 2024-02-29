import React, { Component } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { coy as style, atomDark as darkStyle } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import { getLanguage } from '../../../common/utils/FileUtils';
import { getArtifactContent, getArtifactLocationUrl } from '../../../common/utils/ArtifactUtils';
import './ShowArtifactTextView.css';
import { DesignSystemHocProps, WithDesignSystemThemeHoc } from '@databricks/design-system';
import { shouldEnableLoggedArtifactTableView } from 'common/utils/FeatureUtils';

const LARGE_ARTIFACT_SIZE = 100 * 1024;

type Props = DesignSystemHocProps & {
  runUuid: string;
  path: string;
  size?: number;
  getArtifact?: (...args: any[]) => any;
};

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
    getArtifact: getArtifactContent,
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
      return <div className="artifact-text-view-loading">Loading...</div>;
    }
    if (this.state.error) {
      return <div className="artifact-text-view-error">Oops we couldn't load your file because of an error.</div>;
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
        border: shouldEnableLoggedArtifactTableView() ? 'none' : 'inherit',
      };
      const renderedContent = this.state.text ? prettifyArtifactText(language, this.state.text) : this.state.text;

      const syntaxStyle = theme.isDarkMode ? darkStyle : style;

      return (
        <div className="ShowArtifactPage">
          <div className="text-area-border-box">
            <SyntaxHighlighter language={language} style={syntaxStyle} customStyle={overrideStyles}>
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
    const artifactLocation = getArtifactLocationUrl(this.props.path, this.props.runUuid);
    this.props
      .getArtifact?.(artifactLocation)
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
