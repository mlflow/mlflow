/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import { getArtifactContent } from '../../../common/utils/ArtifactUtils';
import './ShowArtifactHtmlView.css';
import Iframe from 'react-iframe';
import { ArtifactViewSkeleton } from './ArtifactViewSkeleton';
import { ArtifactViewErrorState } from './ArtifactViewErrorState';
import type { LoggedModelArtifactViewerProps } from './ArtifactViewComponents.types';
import { fetchArtifactUnified, type FetchArtifactUnifiedFn } from './utils/fetchArtifactUnified';

type ShowArtifactHtmlViewState = {
  loading: boolean;
  error?: any;
  html: string;
  path: string;
};

type ShowArtifactHtmlViewProps = {
  runUuid: string;
  path: string;
  getArtifact: FetchArtifactUnifiedFn;
} & LoggedModelArtifactViewerProps;

class ShowArtifactHtmlView extends Component<ShowArtifactHtmlViewProps, ShowArtifactHtmlViewState> {
  constructor(props: ShowArtifactHtmlViewProps) {
    super(props);
    this.fetchArtifacts = this.fetchArtifacts.bind(this);
  }

  private artifactRequestId = 0;

  componentWillUnmount() {
    this.artifactRequestId += 1;
  }

  static defaultProps = {
    getArtifact: fetchArtifactUnified,
  };

  state = {
    loading: true,
    error: undefined,
    html: '',
    path: '',
  };

  componentDidMount() {
    this.fetchArtifacts();
  }

  componentDidUpdate(prevProps: ShowArtifactHtmlViewProps) {
    if (
      this.props.path !== prevProps.path ||
      this.props.runUuid !== prevProps.runUuid ||
      this.props.artifactRootUri !== prevProps.artifactRootUri ||
      this.props.isLoggedModelsMode !== prevProps.isLoggedModelsMode ||
      this.props.loggedModelId !== prevProps.loggedModelId
    ) {
      this.fetchArtifacts();
    }
  }

  render() {
    if (this.state.loading || this.state.path !== this.props.path) {
      return <ArtifactViewSkeleton className="artifact-html-view-loading" />;
    }
    if (this.state.error) {
      return <ArtifactViewErrorState error={this.state.error} className="artifact-html-view-error" />;
    } else {
      return (
        <div className="mlflow-artifact-html-view">
          <Iframe
            url=""
            src={this.getBlobURL(this.state.html, 'text/html')}
            width="100%"
            height="100%"
            id="html"
            className="mlflow-html-iframe"
            display="block"
            position="relative"
            sandbox="allow-scripts"
          />
        </div>
      );
    }
  }

  getBlobURL = (code: string, type: string) => {
    const blob = new Blob([code], { type });
    return URL.createObjectURL(blob);
  };

  /** Fetches artifacts and updates component state with the result */
  fetchArtifacts() {
    const requestId = ++this.artifactRequestId;
    const { artifactRootUri, path, runUuid, isLoggedModelsMode, loggedModelId } = this.props;
    this.setState({ loading: true, error: undefined, html: '', path });

    this.props
      .getArtifact?.({ artifactRootUri, path, runUuid, isLoggedModelsMode, loggedModelId }, getArtifactContent)
      .then((html: string) => {
        if (requestId !== this.artifactRequestId) return;
        this.setState({ html, error: undefined, loading: false, path });
      })
      .catch((error: Error) => {
        if (requestId !== this.artifactRequestId) return;
        this.setState({ error, loading: false, path });
      });
  }
}

export default ShowArtifactHtmlView;
