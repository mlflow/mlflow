/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { Pagination, Spinner } from '@databricks/design-system';
import { getArtifactBytesContent } from '../../../common/utils/ArtifactUtils';
import './ShowArtifactPdfView.css';
import Utils from '../../../common/utils/Utils';
import { ErrorWrapper } from '../../../common/utils/ErrorWrapper';
import { ArtifactViewSkeleton } from './ArtifactViewSkeleton';
import { ArtifactViewErrorState } from './ArtifactViewErrorState';
import type { LoggedModelArtifactViewerProps } from './ArtifactViewComponents.types';
import { fetchArtifactUnified, type FetchArtifactUnifiedFn } from './utils/fetchArtifactUnified';
import { setupReactPDFWorker } from './utils/setupReactPDFWorker';

setupReactPDFWorker(pdfjs);

type Props = {
  runUuid: string;
  path: string;
  getArtifact: FetchArtifactUnifiedFn<ArrayBuffer>;
} & LoggedModelArtifactViewerProps;

type State = any;

class ShowArtifactPdfView extends Component<Props, State> {
  state = {
    loading: true,
    error: undefined,
    pdfData: undefined,
    currentPage: 1,
    numPages: 1,
  };

  private artifactRequestId = 0;

  componentWillUnmount() {
    this.artifactRequestId += 1;
  }

  static defaultProps = {
    getArtifact: fetchArtifactUnified,
  };

  /** Fetches artifacts and updates component state with the result */
  fetchPdf() {
    const requestId = ++this.artifactRequestId;
    const { artifactRootUri, path, runUuid, isLoggedModelsMode, loggedModelId } = this.props;
    this.setState({ loading: true, error: undefined, pdfData: undefined, currentPage: 1, numPages: 1 });

    this.props
      .getArtifact?.({ artifactRootUri, path, runUuid, isLoggedModelsMode, loggedModelId }, getArtifactBytesContent)
      .then((artifactPdfData: any) => {
        if (requestId !== this.artifactRequestId) return;
        this.setState({ pdfData: { data: artifactPdfData }, error: undefined, loading: false });
      })
      .catch((error: any) => {
        if (requestId !== this.artifactRequestId) return;
        this.setState({ error: error, loading: false });
      });
  }

  componentDidMount() {
    this.fetchPdf();
  }

  resetPDFState() {
    this.setState({
      pdfData: undefined,
      error: undefined,
      loading: true,
      currentPage: 1,
      numPages: 1,
    });
  }

  componentDidUpdate(prevProps: Props) {
    if (
      this.props.path !== prevProps.path ||
      this.props.runUuid !== prevProps.runUuid ||
      this.props.artifactRootUri !== prevProps.artifactRootUri ||
      this.props.isLoggedModelsMode !== prevProps.isLoggedModelsMode ||
      this.props.loggedModelId !== prevProps.loggedModelId
    ) {
      this.resetPDFState();
      this.fetchPdf();
    }
  }

  onDocumentLoadSuccess = ({ numPages }: any) => {
    this.setState({ numPages });
  };

  onDocumentLoadError = (error: any) => {
    Utils.logErrorAndNotifyUser(new ErrorWrapper(error));
  };

  onPageChange = (newPageNumber: any, itemsPerPage: any) => {
    this.setState({ currentPage: newPageNumber });
  };

  renderPdf = () => {
    return (
      <React.Fragment>
        <div className="mlflow-pdf-viewer">
          <div className="mlflow-paginator">
            <Pagination
              // @ts-expect-error TS(2322): Type '{ simple: true; currentPageIndex: number; nu... Remove this comment to see the full error message
              simple
              currentPageIndex={this.state.currentPage}
              numTotal={this.state.numPages}
              pageSize={1}
              onChange={this.onPageChange}
              /*
               * Currently DuBois pagination does not natively support
               * "simple" mode which is required here, hence `dangerouslySetAntdProps`
               */
              dangerouslySetAntdProps={{ simple: true }}
            />
          </div>
          <div className="mlflow-document">
            <Document
              file={this.state.pdfData}
              onLoadSuccess={this.onDocumentLoadSuccess}
              onLoadError={this.onDocumentLoadError}
              loading={<Spinner />}
            >
              <Page
                pageNumber={this.state.currentPage}
                loading={<Spinner />}
                renderTextLayer={false}
                renderAnnotationLayer={false}
              />
            </Document>
          </div>
        </div>
      </React.Fragment>
    );
  };

  render() {
    if (this.state.loading) {
      return <ArtifactViewSkeleton className="artifact-pdf-view-loading" />;
    }
    if (this.state.error) {
      return <ArtifactViewErrorState error={this.state.error} className="artifact-pdf-view-error" />;
    } else {
      return <div className="mlflow-pdf-outer-container">{this.renderPdf()}</div>;
    }
  }
}

export default ShowArtifactPdfView;
