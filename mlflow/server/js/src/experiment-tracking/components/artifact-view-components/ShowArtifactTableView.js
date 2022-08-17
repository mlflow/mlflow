import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { getSrc } from './ShowArtifactPage';
import { getArtifactContent } from '../../../common/utils/ArtifactUtils';
import Papa from 'papaparse';
// import './ShowArtifactTextView.css';

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
    text: undefined,
  };

  componentDidMount() {
    console.log('TABLE')
    this.fetchArtifacts();
  }

  componentDidUpdate(prevProps) {
    if (this.props.path !== prevProps.path || this.props.runUuid !== prevProps.runUuid) {
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
      
      return (<div></div>);
    } else {
      return (
        <div className='ShowArtifactPage'>
          <div className='text-area-border-box'>
            {this.state.text}
          </div>
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
          const result = Papa.parse(text, {header: true,})

          let data;
          if (result.data.length > this.MAX_ROW_LENGTH) {
            data = result.data.slice(0, this.MAX_ROW_LENGTH)
          } else {
            data = result.data;
          }
          console.log(data[0])
          this.setState({ data: data, headers: result.meta.fields, loading: false })
        } catch (error) {
          console.log(error)
          this.setState({ text: text, loading: false });
        }
      })
      .catch((error) => {
        this.setState({ error: error, loading: false });
      });
  }

  getColumnDefs() {
    const {
      metricKeyList,
      paramKeyList,
      categorizedUncheckedKeys,
      visibleTagKeyList,
      orderByKey,
      orderByAsc,
      onSortBy,
      onExpand,
      designSystemThemeApi,
    } = this.props;
    const { theme } = designSystemThemeApi;
    const getStyle = (key) =>
      key === this.props.orderByKey ? { backgroundColor: theme.colors.blue100 } : {};
    const headerStyle = (key) => getStyle(key);
    const cellStyle = (params) => getStyle(params.colDef.headerComponentParams.canonicalSortKey);

    return [];
  }
}

export default ShowArtifactTableView;
