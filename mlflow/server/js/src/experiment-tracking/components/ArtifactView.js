import React, { Component } from 'react';
import PropTypes from 'prop-types';
import {
  getBasename, getExtension, IMAGE_EXTENSIONS,
  TEXT_EXTENSIONS,
} from '../../common/utils/FileUtils';
import { ArtifactNode as ArtifactUtils, ArtifactNode } from '../utils/ArtifactUtils';
import { decorators, Treebeard } from 'react-treebeard';
import bytes from 'bytes';
import RegisterModelButton from '../../model-registry/components/RegisterModelButton';
import ShowArtifactPage, { getSrc } from './artifact-view-components/ShowArtifactPage';
import {
  ModelVersionStatus,
  ModelVersionStatusIcons,
  DefaultModelVersionStatusMessages, modelVersionStatusIconTooltips,
} from '../../model-registry/constants';
import Utils from '../../common/utils/Utils';
import _ from 'lodash';
import { getModelVersionPageURL } from '../../model-registry/routes';
import { Tooltip } from 'antd';

import './ArtifactView.css';
import spinner from '../../common/static/mlflow-spinner.png';

export class ArtifactView extends Component {
  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    // The root artifact node.
    artifactNode: PropTypes.instanceOf(ArtifactNode).isRequired,
    artifactRootUri: PropTypes.string.isRequired,
    listArtifactsApi: PropTypes.func.isRequired,
    modelVersionsBySource: PropTypes.object.isRequired,
    handleActiveNodeChange: PropTypes.func.isRequired,
  };

  state = {
    activeNodeId: undefined,
    toggledNodeIds: {},
    requestedNodeIds: new Set(),
  };

  renderModelVersionInfoSection() {
    const { runUuid, modelVersionsBySource } = this.props;
    const { activeNodeId } = this.state;
    const activeNodeRealPath = this.getActiveNodeRealPath();
    const existingModelVersions = modelVersionsBySource[activeNodeRealPath];

    return existingModelVersions ? (
      <ModelVersionInfoSection modelVersion={_.last(existingModelVersions)}/>
    ) : (
      <RegisterModelButton
        runUuid={runUuid}
        modelPath={activeNodeRealPath}
        disabled={activeNodeId === undefined}
      />
    );
  }

  renderPathAndSizeInfo() {
    return (
      <div className='artifact-info-left'>
        <div className='artifact-info-path'>
          <label>Full Path:</label> {this.getActiveNodeRealPath()}
        </div>
        <div className='artifact-info-size'>
          <label>Size:</label> {this.getActiveNodeSize()}
        </div>
      </div>
    );
  }

  renderDownloadLink() {
    const { runUuid } = this.props;
    const { activeNodeId } = this.state;
    return (
      <div className='artifact-info-link'>
        <a href={getSrc(activeNodeId, runUuid)} title='Download artifact'>
          <i className='fas fa-download' />
        </a>
      </div>
    );
  }

  renderArtifactInfo() {
    return (
      <div className='artifact-info'>
        {this.renderPathAndSizeInfo()}
        <div className='artifact-info-right'>
          {this.activeNodeIsDirectory()
            ? (Utils.isModelRegistryEnabled() ? this.renderModelVersionInfoSection() : null)
            : this.renderDownloadLink()}
        </div>
      </div>
    );
  }

  onToggleTreebeard = (dataNode, toggled) => {
    const { id, loading } = dataNode;
    const newRequestedNodeIds = new Set(this.state.requestedNodeIds);
    // - loading indicates that this node is a directory and has not been loaded yet.
    // - requestedNodeIds keeps track of in flight requests.
    if (loading && !this.state.requestedNodeIds.has(id)) {
      this.props.listArtifactsApi(this.props.runUuid, id);
    }
    this.setState({
      activeNodeId: id,
      toggledNodeIds: {
        ...this.state.toggledNodeIds,
        [id]: toggled,
      },
      requestedNodeIds: newRequestedNodeIds,
    });
  };

  getTreebeardData = (artifactNode) => {
    const isRoot = artifactNode.isRoot;
    if (isRoot) {
      if (artifactNode.children) {
        return Object.values(artifactNode.children).map((c) => this.getTreebeardData(c));
      }
      // This case should never happen since we should never call this function on an empty root.
      throw Error("unreachable code.");
    }

    let id;
    let name;
    let toggled;
    let children;
    let active;

    if (artifactNode.fileInfo) {
      const path = artifactNode.fileInfo.path;
      id = path;
      name = getBasename(path);
    }

    const toggleState = this.state.toggledNodeIds[id];
    if (toggleState) {
      toggled = toggleState;
    }

    if (artifactNode.children) {
      children = Object.values(artifactNode.children).map((c) => this.getTreebeardData(c));
    }

    if (this.state.activeNodeId === id) {
      active = true;
    }

    const loading = artifactNode.children !== undefined && !artifactNode.isLoaded;

    return {
      id,
      name,
      toggled,
      children,
      active,
      loading,
    };
  };

  getActiveNodeRealPath() {
    if (this.state.activeNodeId) {
      return `${this.props.artifactRootUri}/${this.state.activeNodeId}`;
    }
    return this.props.artifactRootUri;
  }

  getActiveNodeSize() {
    if (this.state.activeNodeId) {
      const node = ArtifactUtils.findChild(this.props.artifactNode, this.state.activeNodeId);
      const size = node.fileInfo.file_size || '0';
      return bytes(parseInt(size, 10));
    }
    return bytes(0);
  }

  activeNodeIsDirectory() {
    if (this.state.activeNodeId) {
      const node = ArtifactUtils.findChild(this.props.artifactNode, this.state.activeNodeId);
      return node.fileInfo.is_dir;
    } else {
      // No node is highlighted so we're displaying the root, which is a directory.
      return true;
    }
  }

  componentDidUpdate(prevProps, prevState) {
    const { activeNodeId } = this.state;
    if (prevState.activeNodeId !== activeNodeId) {
      this.props.handleActiveNodeChange(this.activeNodeIsDirectory());
    }
  }

  render() {
    if (ArtifactUtils.isEmpty(this.props.artifactNode)) {
      return <NoArtifactView />;
    }
    return (
      <div>
        <div className="artifact-view">
          <div className="artifact-left">
            <Treebeard
              data={this.getTreebeardData(this.props.artifactNode)}
              onToggle={this.onToggleTreebeard}
              style={TREEBEARD_STYLE}
              decorators={decorators}
            />
          </div>
          <div className="artifact-right">
            {this.state.activeNodeId ? this.renderArtifactInfo() : null}
            <ShowArtifactPage runUuid={this.props.runUuid} path={this.state.activeNodeId}/>
          </div>
        </div>
      </div>
    );
  }
}

function ModelVersionInfoSection(props) {
  const { modelVersion } = props;
  const { name, version, status, status_message } = modelVersion;

  const modelVersionLink = (
    <a
      href={getModelVersionPageURL(name, version)}
      className='model-version-link'
      title={`${name}, v${version}`}
      target='_blank'
    >
      <span className='model-name'>{name}</span>
      <span>,&nbsp;v{version}&nbsp;</span>
      <i className='fas fa-external-link-alt'/>
    </a>
  );

  return (
    <div className='model-version-info'>
      <div className='model-version-link-section'>
        <Tooltip title={status_message || modelVersionStatusIconTooltips[status]}>
          {ModelVersionStatusIcons[status]}
        </Tooltip>
        {modelVersionLink}
      </div>
      <div className='model-version-status-text'>
        {status === ModelVersionStatus.READY ? (
          <React.Fragment>
            Registered on {' '}
            {Utils.formatTimestamp(modelVersion.creation_timestamp, 'yyyy/mm/dd')}
          </React.Fragment>
        ) : (status_message || DefaultModelVersionStatusMessages[status])}
      </div>
    </div>
  );
}

ModelVersionInfoSection.propTypes = { modelVersion: PropTypes.object.isRequired };

function NoArtifactView() {
  return (
    <div className="empty-artifact-outer-container">
      <div className="empty-artifact-container">
        <div>
          {/* TODO: put a nice image here */}
        </div>
        <div>
          <div className="no-artifacts">No Artifacts Recorded</div>
          <div className="no-artifacts-info">
            Use the log artifact APIs to store file outputs from MLflow runs.
          </div>
        </div>
      </div>
    </div>
  );
}

const TREEBEARD_STYLE = {
  tree: {
    base: {
      listStyle: 'none',
      margin: 0,
      padding: 0,
      backgroundColor: '#FAFAFA',
      fontSize: '14px',
      maxWidth: '500px',
      height: '556px',
      overflow: 'scroll',
    },
    node: {
      base: {
        position: 'relative',
      },
      link: {
        cursor: 'pointer',
        position: 'relative',
        padding: '0px 5px',
        display: 'block',
      },
      activeLink: {
        background: '#c7c2d0',
      },
      toggle: {
        base: {
          position: 'relative',
          display: 'inline-block',
          verticalAlign: 'top',
          marginLeft: '-5px',
          height: '24px',
          width: '24px',
        },
        wrapper: {
          position: 'absolute',
          top: '50%',
          left: '50%',
          margin: '-12px 0 0 -4px',
          height: '14px',
        },
        height: 7,
        width: 7,
        arrow: {
          fill: '#7a7a7a',
          strokeWidth: 0,
        },
      },
      header: {
        base: {
          display: 'inline-block',
          verticalAlign: 'top',
          color: '#333',
        },
        connector: {
          width: '2px',
          height: '12px',
          borderLeft: 'solid 2px black',
          borderBottom: 'solid 2px black',
          position: 'absolute',
          top: '0px',
          left: '-21px',
        },
        title: {
          lineHeight: '24px',
          verticalAlign: 'middle',
        },
      },
      subtree: {
        listStyle: 'none',
        paddingLeft: '19px',
      },
    },
  },
};

// eslint-disable-next-line react/prop-types
decorators.Header = ({style, node}) => {
  let iconType;
  if (node.children) {
    iconType = 'folder';
  } else {
    const extension = getExtension(node.name);
    if (IMAGE_EXTENSIONS.has(extension)) {
      iconType = 'file-image';
    } else if (TEXT_EXTENSIONS.has(extension)) {
      iconType = 'file-code';
    } else {
      iconType = 'file-alt';
    }
  }
  const iconClass = `fa fa-${iconType}`;

  // Add margin-left to the non-directory nodes to align the arrow, icons, and texts.
  const iconStyle = node.children
    ? { marginRight: '5px' }
    : { marginRight: '5px', marginLeft: '19px' };

  return (
    <div style={style.base}>
      <div style={style.title}>
        <i className={iconClass} style={iconStyle}/>
        {node.name}
      </div>
    </div>
  );
};

// eslint-disable-next-line react/prop-types
decorators.Loading = ({style}) => {
  return (
    <div style={style}>
      <img alt="" className="loading-spinner" src={spinner}/>
      {' '}loading...
    </div>
  );
};
