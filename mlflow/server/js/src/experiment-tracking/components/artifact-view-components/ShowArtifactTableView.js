import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { getSrc } from './ShowArtifactPage';
import { getArtifactContent } from '../../../common/utils/ArtifactUtils';
import { MLFlowAgGridLoader } from 'src/common/components/ag-grid/AgGridLoader';
import { Spinner } from '../../../common/components/Spinner';
import Papa from 'papaparse';

class ShowArtifactTableView extends Component {
  constructor(props) {
    super(props);
    this.fetchArtifacts = this.fetchArtifacts.bind(this);
  }

  static MAX_ROW_LENGTH = 500;

  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    path: PropTypes.string.isRequired,
    getArtifact: PropTypes.func,
  };

  static defaultProps = {
    getArtifact: getArtifactContent,
  };

  state = {
    loading: true,
    error: undefined,
    data: undefined,
    headers: undefined,
    originalRowLength: 0,
    text: undefined,
  };

  componentDidMount() {
    this.fetchArtifacts();
  }

  componentDidUpdate(prevProps) {
    if (this.props.path !== prevProps.path || this.props.runUuid !== prevProps.runUuid) {
      this.setState({ data: [], headers: [], originalRowLength: 0 });
      this.fetchArtifacts();
    }
  }

  render() {
    if (this.state.loading) {
      return <div className='artifact-text-view-loading'>Loading...</div>;
    }
    if (this.state.error) {
      return (
        <div className='artifact-text-view-error'>
          Oops we couldn't load your file because of an error.
        </div>
      );
    }

    if (this.state.data) {
      // const { theme } = designSystemThemeApi; // Do we not need this?
      const agGridOverrides = {
        '--ag-border-color': 'rgba(0, 0, 0, 0.06)',
        '--ag-header-foreground-color': '#20272e',
        // '&.ag-header': {
        //   position: 'static',
        //   top: 0,
        //   zIndex: 1,
        // },
        // '&.ag-header': {
        //   top: 0,
        //   position: 'fixed',
        //   width: 'auto',
        //   display: 'table',
        //   zIndex: 99,
        // },
        // '&.ag-root': {
        //   overflow: 'scroll',
        // },
        // '&.ag-root-wrapper': {
        //   border: '0',
        //   borderRadius: '4px',
        //   overflow: 'scroll',
        // },
        overflow: 'scroll',
        // So scrolling horizontally on the trcakpad doesn't make browser go back a page
        'overscroll-behavior-x': 'contain',
      };

      // eslint-disable-next-line max-len
      const rowPreviewMessage = `Previewing the first ${this.state.data.length} rows out of ${this.state.originalRowLength}`;

      return (
        <div id='grid-parent-scrollable' className='ag-theme-balham' css={agGridOverrides}>
          <span style={{ display: 'flex', justifyContent: 'center' }}>{rowPreviewMessage}</span>
          <MLFlowAgGridLoader
            columnDefs={this.getColumnDefs(this.state.headers)}
            defaultColDef={{ suppressMenu: true }}
            loadingOverlayComponentParams={{ showImmediately: true }}
            loadingOverlayComponent='loadingOverlayComponent'
            components={{ loadingOverlayComponent: Spinner }}
            domLayout='autoHeight'
            enableCellTextSelection
            enableSele
            rowData={this.state.data}
            suppressColumnVirtualisation
            alwaysShowHorizontalScroll
            // onGridReady={(params) => params.api.sizeColumnsToFit()}
            onFirstDataRendered={(params) => {
              const allColumnIds = [];
              params.columnApi.getAllColumns().forEach((column) => {
                allColumnIds.push(column.getId());
              });
              params.columnApi.autoSizeColumns(allColumnIds, false);
            }}
          />
        </div>
      );
    } else {
      return (
        <div className='ShowArtifactPage'>
          <div className='text-area-border-box'>{this.state.text}</div>
        </div>
      );
    }
  }

  /** Fetches artifacts and updates component state with the result */
  fetchArtifacts() {
    const artifactLocation = getSrc(this.props.path, this.props.runUuid);
    this.props
      .getArtifact(artifactLocation)
      .then((text) => {
        try {
          const result = Papa.parse(text, { header: true });
          let dataPreview;
          let originalRowLength = result.data.length;
          if (result.data.length > ShowArtifactTableView.MAX_ROW_LENGTH) {
            dataPreview = result.data.slice(0, ShowArtifactTableView.MAX_ROW_LENGTH);
          } else {
            dataPreview = result.data;
            // Removes possible parsed empty line at end of file
            if (Object.values(dataPreview[dataPreview.length - 1]).map((value) => value === '')) {
              dataPreview.pop();
              originalRowLength -= 1;
            }
          }
          this.setState({
            data: dataPreview,
            headers: result.meta.fields,
            originalRowLength: originalRowLength,
            loading: false,
          });
        } catch (_) {
          this.setState({ text: text, loading: false });
        }
      })
      .catch((error) => {
        this.setState({ error: error, loading: false });
      });
  }

  getColumnDefs(fields) {
    return fields.map((field) => {
      return {
        headerName: field,
        field: field,
        suppressSizeToFit: true,
        sortable: true,
        resizable: true,
        maxWidth: 360,
      };
    });
  }
}

export default ShowArtifactTableView;
