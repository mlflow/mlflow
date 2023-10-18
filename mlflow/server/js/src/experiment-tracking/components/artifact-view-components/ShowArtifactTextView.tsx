/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { coy as style } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import { getLanguage } from '../../../common/utils/FileUtils';
import { getArtifactContent, getArtifactLocationUrl } from '../../../common/utils/ArtifactUtils';
import './ShowArtifactTextView.css';

const LARGE_ARTIFACT_SIZE = 100 * 1024;

type OwnProps = {
  runUuid: string;
  path: string;
  size?: number;
  getArtifact?: (...args: any[]) => any;
};

type State = any;

type Props = OwnProps & typeof ShowArtifactTextView.defaultProps;

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
      return <div className='artifact-text-view-loading'>Loading...</div>;
    }
    if (this.state.error) {
      return (
        <div className='artifact-text-view-error'>
          Oops we couldn't load your file because of an error.
        </div>
      );
    } else {
      const isLargeFile = (this.props.size || 0) > LARGE_ARTIFACT_SIZE;
      const language = isLargeFile ? 'text' : getLanguage(this.props.path);
      const overrideStyles = {
        fontFamily: 'Source Code Pro,Menlo,monospace',
        fontSize: '13px',
        overflow: 'auto',
        marginTop: '0',
        width: '100%',
        height: '100%',
        padding: '5px',
      };
      const renderedContent = ShowArtifactTextView.prettifyText(language, this.state.text);
      return (
        <div className='ShowArtifactPage'>
          <div className='text-area-border-box'>
            <SyntaxHighlighter language={language} style={style} customStyle={overrideStyles}>
              {renderedContent}
            </SyntaxHighlighter>
          </div>
        </div>
      );
    }
  }

  static prettifyText(language: any, rawText: any) {
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

  /** Fetches artifacts and updates component state with the result */
  fetchArtifacts() {
    this.setState({ loading: true });
    const artifactLocation = getArtifactLocationUrl(this.props.path, this.props.runUuid);
    this.props
      .getArtifact(artifactLocation)
      .then((text: any) => {
        this.setState({ text: text, loading: false });
      })
      .catch((error: any) => {
        this.setState({ error: error, loading: false });
      });
    this.setState({ path: this.props.path });
  }
}

export default ShowArtifactTextView;
