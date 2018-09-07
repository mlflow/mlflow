import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { getSrc } from './ShowArtifactPage';
import { CSRF_HEADER_NAME, getCsrfToken } from '../../setupCsrf';
import Papa from 'papaparse';
import ReactDataGrid from 'react-data-grid';
import { Data, Toolbar } from 'react-data-grid-addons';
import { Alert } from 'react-bootstrap';
import bytes from 'bytes';
import sizeof from 'object-sizeof';
import './ShowArtifactCsvView.css';

const MAX_DATA_TO_DOWNLOAD = 1024 * 1024 * 1024; // 1 GB
const MAX_DATA_TO_PARSE = 1024 * 100; // 100 KB

const DEFAULT_COLUMN_PROPS = {
  filterable: true,
  sortable: true,
};

class ShowArtifactCsvView extends Component {
  constructor(props) {
    super(props);
    this.fetchArtifacts = this.fetchArtifacts.bind(this);
    this.getRowAt = this.getRowAt.bind(this);
    this.handleParsingComplete = this.handleParsingComplete.bind(this);
    this.handleStep = this.handleStep.bind(this);
    this.handleFilterChange = this.handleFilterChange.bind(this);
    this.onClearFilters = this.onClearFilters.bind(this);
    this.handleGridSort = this.handleGridSort.bind(this);
    this.handleParseError = this.handleParseError.bind(this);
    this.handlePreviewWarningDismissed = this.handlePreviewWarningDismissed.bind(this);
  }

  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    path: PropTypes.string.isRequired,
    fileSize: PropTypes.number.isRequired,
  };

  state = {
    fileTooLarge: false,
    error: undefined,
    data: [],
    renderedData: undefined,
    sortColumn: undefined,
    sortDirection: undefined,
    filters: {},
    numRowsRead: 0,
    approximateSize: 0,
    isPreview: false,
    previewWarningDismissed: false,
    readingFinished: false,
  };

  prepareNewFile() {
    if (this.props.fileSize > MAX_DATA_TO_DOWNLOAD) {
      this.setState({fileTooLarge: true});
    } else {
      this.fetchArtifacts();
    }
  }

  componentWillMount() {
    this.prepareNewFile();
  }

  componentDidUpdate(prevProps) {
    if (this.props.path !== prevProps.path || this.props.runUuid !== prevProps.runUuid) {
      // eslint-disable-next-line react/no-did-update-set-state
      this.setState({
        fileTooLarge: false,
        error: undefined,
        data: [],
        renderedData: undefined,
        sortColumn: undefined,
        sortDirection: undefined,
        filters: {},
        numRowsRead: 0,
        approximateSize: 0,
        isPreview: false,
        previewWarningDismissed: false,
        readingFinished: false,
      });
      this.prepareNewFile();
    }
  }

  getColumnDefinitions(fieldNames) {
    return fieldNames.map((fieldName) => {
      return {
        ...DEFAULT_COLUMN_PROPS,
        key: fieldName,
        name: fieldName,
      };
    });
  }

  setRenderedData() {
    const selectedRows = Data.Selectors.getRows({
      rows: this.state.data,
      sortDirection: this.state.sortDirection,
      sortColumn: this.state.sortColumn,
      filters: this.state.filters,
    });
    this.setState({ renderedData: selectedRows });
  }

  handleFilterChange = (filter) => {
    const newFilters = Object.assign({}, this.state.filters);
    if (filter.filterTerm) {
      newFilters[filter.column.key] = filter;
    } else {
      delete newFilters[filter.column.key];
    }
    this.setState({ filters: newFilters }, this.setRenderedData);
  };

  onClearFilters = () => {
    this.setState({ filters: {} }, this.setRenderedData);
  };

  handleGridSort = (sortColumn, sortDirection) => {
    this.setState({ sortColumn: sortColumn, sortDirection: sortDirection }, this.setRenderedData);
  };

  getRowAt(index) {
    if (index < 0 || index > this.state.renderedData.length) {
      return undefined;
    }

    return this.state.renderedData[index];
  }

  handleParsingComplete() {
    this.setState({ readingFinished: true }, this.setRenderedData);
  }

  handleStep(results, parser) {
    const newNumRowsRead = this.state.numRowsRead + results.data.length;
    const newApproximateSize = this.state.approximateSize + sizeof(results.data);
    if (this.state.numRowsRead === 0) {
      this.setState({
        columns: this.getColumnDefinitions(results.meta.fields),
        approximateSize: newApproximateSize,
        data: [...this.state.data, ...results.data],
        numRowsRead: newNumRowsRead,
      });
    } else {
      this.setState({
        approximateSize: newApproximateSize,
        data: [...this.state.data, ...results.data],
        numRowsRead: newNumRowsRead,
      });
    }
    if (newApproximateSize >= MAX_DATA_TO_PARSE) {
      parser.abort();
      this.setState({ isPreview: true }, this.setRenderedData);
    }
  }

  handleParseError(error) {
    this.setState({error: error});
  }

  handlePreviewWarningDismissed() {
    this.setState({ previewWarningDismissed: true });
  }

  render() {
    if (this.state.error) {
      return (
        <div>
          <em>Oops we couldn't load your file because of an error parsing the CSV contents</em>:
          {this.state.error.message} at row {this.state.error.row}
        </div>
      );
    }
    if (this.state.fileTooLarge) {
      return (
        <div>
          File too large to download: {bytes(this.props.fileSize)} exceeds file size
          limit of {bytes(MAX_DATA_TO_DOWNLOAD)}.
        </div>
      );
    }
    if (this.state.numRowsRead === 0 || this.state.renderedData === undefined) {
      return (
        <div>
          Loading...
        </div>
      );
    }
    let dataGridContainerHeight = `100%`;
    const showPreviewWarning = this.state.isPreview && !this.state.previewWarningDismissed;
    if (showPreviewWarning) {
      dataGridContainerHeight = `90%`;
    }
    return (
      <div className="data-grid-outer-container">
        {showPreviewWarning ?
          <Alert className="data-grid-preview-warning"
                 onDismiss={this.handlePreviewWarningDismissed}>
              <h4>Only showing the first {this.state.data.length} rows.</h4>
          </Alert>
          :
          null
        }
        <div className="data-grid-container" style={{height: dataGridContainerHeight}}>
          <ReactDataGrid
            columns={this.state.columns}
            toolbar={<Toolbar enableFilter/>}
            rowGetter={this.getRowAt}
            rowsCount={this.state.renderedData.length}
            onAddFilter={this.handleFilterChange}
            onClearFilters={this.onClearFilters}
            onGridSort={this.handleGridSort}
            rowScrollTimeout={200}/>
        </div>
      </div>
    );
  }

  fetchArtifacts() {
    Papa.parse(getSrc(this.props.path, this.props.runUuid), {
      complete: this.handleParsingComplete,
      download: true,
      downloadRequestHeaders: { [CSRF_HEADER_NAME]: getCsrfToken() },
      error: this.handleParseError,
      dynamicTyping: true,
      header: true,
      skipEmptyLines: 'greedy',
      step: this.handleStep,
    });
  }
}

export default ShowArtifactCsvView;
