/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
// @ts-expect-error TS(7016): Could not find a declaration file for module 'reac... Remove this comment to see the full error message
import { Document, Page, pdfjs } from 'react-pdf';
import { Pagination, Spinner } from '@databricks/design-system';
import { getArtifactBytesContent, getArtifactLocationUrl } from '../../../common/utils/ArtifactUtils';
import './ShowArtifactPdfView.css';
import Utils from '../../../common/utils/Utils';
import { ErrorWrapper } from '../../../common/utils/ErrorWrapper';
import { ArtifactViewSkeleton } from './ArtifactViewSkeleton';
import { ArtifactViewErrorState } from './ArtifactViewErrorState';

// See: https://github.com/wojtekmaj/react-pdf/blob/master/README.md#enable-pdfjs-worker for how
// workerSrc is supposed to be specified.
pdfjs.GlobalWorkerOptions.workerSrc = `./static-files/pdf.worker.js`;

type OwnProps = {
  runUuid: string;
  path: string;
  getArtifact?: (...args: any[]) => any;
};

type State = any;

type Props = OwnProps & typeof ShowArtifactPdfView.defaultProps;

class ShowArtifactPdfView extends Component<Props, State> {
  state = {
    loading: true,
    error: undefined,
    pdfData: undefined,
    currentPage: 1,
    numPages: 1,
  };

  static defaultProps = {
    getArtifact: getArtifactBytesContent,
  };

  /** Fetches artifacts and updates component state with the result */
  fetchPdf() {
    const artifactLocation = getArtifactLocationUrl(this.props.path, this.props.runUuid);
    this.props
      .getArtifact(artifactLocation)
      .then((artifactPdfData: any) => {
        this.setState({ pdfData: { data: artifactPdfData }, loading: false });
      })
      .catch((error: any) => {
        this.setState({ error: error, loading: false });
      });
  }

  componentDidMount() {
    this.fetchPdf();
  }

  componentDidUpdate(prevProps: Props) {
    if (this.props.path !== prevProps.path || this.props.runUuid !== prevProps.runUuid) {
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
        <div className="pdf-viewer">
          <div className="paginator">
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
          <div className="document">
            <Document
              file={this.state.pdfData}
              onLoadSuccess={this.onDocumentLoadSuccess}
              onLoadError={this.onDocumentLoadError}
              loading={<Spinner />}
            >
              <Page pageNumber={this.state.currentPage} loading={<Spinner />} />
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
      return <ArtifactViewErrorState className="artifact-pdf-view-error" />;
    } else {
      return <div className="pdf-outer-container">{this.renderPdf()}</div>;
    }
  }
}

export default ShowArtifactPdfView;
