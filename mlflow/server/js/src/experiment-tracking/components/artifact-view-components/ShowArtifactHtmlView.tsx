/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import {
  getArtifactContent,
  getArtifactLocationUrl,
  getLoggedModelArtifactLocationUrl,
} from '../../../common/utils/ArtifactUtils';
import './ShowArtifactHtmlView.css';
import Iframe from 'react-iframe';
import { ArtifactViewSkeleton } from './ArtifactViewSkeleton';
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
    if (this.props.path !== prevProps.path || this.props.runUuid !== prevProps.runUuid) {
      this.fetchArtifacts();
    }
  }

  render() {
    if (this.state.loading || this.state.path !== this.props.path) {
      return <ArtifactViewSkeleton className="artifact-html-view-loading" />;
    }
    if (this.state.error) {
      // eslint-disable-next-line no-console -- TODO(FEINF-3587)
      console.error('Unable to load HTML artifact, got error ' + this.state.error);
      return <div className="artifact-html-view-error">Oops we couldn't load your file because of an error.</div>;
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
    const { path, runUuid, isLoggedModelsMode, loggedModelId, experimentId, entityTags } = this.props;

    this.props
      .getArtifact?.({ path, runUuid, isLoggedModelsMode, loggedModelId, experimentId, entityTags }, getArtifactContent)
      .then((html: string) => {
        this.setState({ html: html, loading: false, path: this.props.path });
      })
      .catch((error: Error) => {
        this.setState({ error: error, loading: false, path: this.props.path });
      });
  }
}

export default ShowArtifactHtmlView;
