import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { getSrc } from './ShowArtifactPage';
import { getArtifactContent } from '../../../common/utils/ArtifactUtils';
import { Table } from '@databricks/design-system';
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
      this.setState({ data: undefined, headers: undefined, originalRowLength: 0, loading: true });
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
      // eslint-disable-next-line max-len
      const rowPreviewMessage = `Previewing the first ${this.state.data.length} rows out of ${this.state.originalRowLength}`;

      const columns = this.state.headers.map((f) => ({
        title: f,
        dataIndex: f,
        key: f,
        sorter: (a, b) => {
          if (isNaN(a)) {
            return a[f].localeCompare(b[f]);
          } else {
            return a[f] - b[f];
          }
        },
        width: 200,
        ellipsis: {
          showTitle: true,
        },
      }));

      return (
        <div css={{ overscrollBehaviorX: 'contain', overflowX: 'scroll', margin: 10 }}>
          <span style={{ display: 'flex', justifyContent: 'center' }}>{rowPreviewMessage}</span>
          <Table
            columns={columns}
            dataSource={this.state.data}
            pagination={false}
            sticky
            scroll={{ x: 'min-content', y: true }}
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
}

export default ShowArtifactTableView;
