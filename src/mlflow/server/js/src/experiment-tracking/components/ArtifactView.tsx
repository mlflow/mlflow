/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import { connect } from 'react-redux';
import { injectIntl, FormattedMessage } from 'react-intl';
import { getBasename } from '../../common/utils/FileUtils';
import { ArtifactNode as ArtifactUtils, ArtifactNode } from '../utils/ArtifactUtils';
// @ts-expect-error TS(7016): Could not find a declaration file for module 'byte... Remove this comment to see the full error message
import bytes from 'bytes';
import { RegisterModel } from '../../model-registry/components/RegisterModel';
import ShowArtifactPage from './artifact-view-components/ShowArtifactPage';
import {
  ModelVersionStatus,
  ModelVersionStatusIcons,
  DefaultModelVersionStatusMessages,
  modelVersionStatusIconTooltips,
} from '../../model-registry/constants';
import Utils from '../../common/utils/Utils';
import _ from 'lodash';
import { ModelRegistryRoutes } from '../../model-registry/routes';
import { DesignSystemHocProps, Tooltip, Typography, WithDesignSystemThemeHoc } from '@databricks/design-system';
import './ArtifactView.css';

import { getArtifactRootUri, getArtifacts } from '../reducers/Reducers';
import { getAllModelVersions } from '../../model-registry/reducers';
import { listArtifactsApi } from '../actions';
import { MLMODEL_FILE_NAME } from '../constants';
import { getArtifactLocationUrl } from '../../common/utils/ArtifactUtils';
import { ArtifactViewTree } from './ArtifactViewTree';
import { shouldEnableDeepLearningUI, shouldEnableLoggedArtifactTableView } from '../../common/utils/FeatureUtils';
import { useDesignSystemTheme } from '@databricks/design-system';
import { Button } from '@databricks/design-system';
import { CopyIcon } from '@databricks/design-system';
import { DownloadIcon } from '@databricks/design-system';
import { Checkbox } from '@databricks/design-system';
import { getLoggedTablesFromTags } from 'common/utils/TagUtils';

const { Text } = Typography;

type ArtifactViewImplProps = DesignSystemHocProps & {
  runUuid: string;
  initialSelectedArtifactPath?: string;
  artifactNode: any; // TODO: PropTypes.instanceOf(ArtifactNode)
  artifactRootUri: string;
  listArtifactsApi: (...args: any[]) => any;
  modelVersionsBySource: any;
  handleActiveNodeChange: (...args: any[]) => any;
  runTags?: any;
  modelVersions?: any[];
  intl: {
    formatMessage: (...args: any[]) => any;
  };
  getCredentialsForArtifactReadApi: (...args: any[]) => any;

  /**
   * If true, the artifact browser will try to use all available height
   */
  useAutoHeight?: boolean;
};

type ArtifactViewImplState = any;

export class ArtifactViewImpl extends Component<ArtifactViewImplProps, ArtifactViewImplState> {
  state = {
    activeNodeId: undefined,
    toggledNodeIds: {},
    requestedNodeIds: new Set(),
    viewAsTable: true,
  };

  getExistingModelVersions() {
    const { modelVersionsBySource } = this.props;
    const activeNodeRealPath = Utils.normalize(this.getActiveNodeRealPath());
    return modelVersionsBySource[activeNodeRealPath];
  }

  renderRegisterModelButton() {
    const { runUuid } = this.props;
    const { activeNodeId } = this.state;
    const activeNodeRealPath = this.getActiveNodeRealPath();
    const usingNewRunViewPage = shouldEnableDeepLearningUI();
    return (
      <RegisterModel
        runUuid={runUuid}
        modelPath={activeNodeRealPath}
        modelRelativePath={String(activeNodeId)}
        disabled={activeNodeId === undefined}
        showButton
        buttonType={usingNewRunViewPage ? undefined : 'primary'}
      />
    );
  }

  renderModelVersionInfoSection(existingModelVersions: any) {
    return <ModelVersionInfoSection modelVersion={_.last(existingModelVersions)} />;
  }

  renderPathAndSizeInfo() {
    // We will only be in this function if this.state.activeNodeId is defined
    const node = ArtifactUtils.findChild(this.props.artifactNode, this.state.activeNodeId);
    const activeNodeRealPath = this.getActiveNodeRealPath();

    return (
      <div className="artifact-info-left">
        <div className="artifact-info-path">
          <label>
            <FormattedMessage
              defaultMessage="Full Path:"
              // eslint-disable-next-line max-len
              description="Label to display the full path of where the artifact of the experiment runs is located"
            />
          </label>{' '}
          {/* @ts-expect-error TS(2322): Type '{ children: string; className: string; ellip... Remove this comment to see the full error message */}
          <Text className="artifact-info-text" ellipsis copyable>
            {activeNodeRealPath}
          </Text>
        </div>
        {node.fileInfo.is_dir === false ? (
          <div className="artifact-info-size">
            <label>
              <FormattedMessage
                defaultMessage="Size:"
                description="Label to display the size of the artifact of the experiment"
              />
            </label>{' '}
            {bytes(this.getActiveNodeSize())}
          </div>
        ) : null}
      </div>
    );
  }

  renderPathAndSizeInfoV2() {
    // We will only be in this function if this.state.activeNodeId is defined
    const node = ArtifactUtils.findChild(this.props.artifactNode, this.state.activeNodeId);
    const activeNodeRealPath = this.getActiveNodeRealPath();

    const { theme } = this.props.designSystemThemeApi;

    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.sm,
        }}
      >
        <div
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
            textOverflow: 'ellipsis',
          }}
        >
          <Typography.Text bold size="lg">
            {this.state.activeNodeId}
          </Typography.Text>
          {node.fileInfo.is_dir === false && (
            <Typography.Text color="secondary">{bytes(this.getActiveNodeSize())}</Typography.Text>
          )}
        </div>
        <div
          style={{
            display: 'inline-flex',
            gap: theme.spacing.sm,
          }}
        >
          <Typography.Text
            color="secondary"
            dangerouslySetAntdProps={{
              copyable: {
                text: activeNodeRealPath,
                icon: <CopyIcon />,
                tooltips: [
                  <FormattedMessage
                    defaultMessage="Copy path"
                    description="Tooltip displayed on hover over the copy icon"
                  />,
                  <FormattedMessage
                    defaultMessage="Path copied"
                    description="Tooltip displayed after path was successfully copied to clipboard"
                  />,
                ],
              },
            }}
          >
            <FormattedMessage
              defaultMessage="Path:"
              // eslint-disable-next-line max-len
              description="Label to display the full path of where the artifact of the experiment runs is located"
            />{' '}
            {activeNodeRealPath}
          </Typography.Text>
        </div>
      </div>
    );
  }

  onDownloadClick(runUuid: any, artifactPath: any) {
    window.location.href = getArtifactLocationUrl(artifactPath, runUuid);
  }

  renderDownloadLink() {
    const { runUuid } = this.props;
    const { activeNodeId } = this.state;
    return (
      <div className="artifact-info-link">
        {/* eslint-disable-next-line jsx-a11y/anchor-is-valid */}
        <a
          onClick={() => this.onDownloadClick(runUuid, activeNodeId)}
          title={this.props.intl.formatMessage({
            defaultMessage: 'Download artifact',
            description: 'Link to download the artifact of the experiment',
          })}
        >
          <i className="fas fa-download" />
        </a>
      </div>
    );
  }

  renderControls() {
    const { runUuid } = this.props;
    const { activeNodeId } = this.state;
    return (
      <div style={{ display: 'flex', alignItems: 'flex-start' }}>
        <div style={{ display: 'inline-flex', alignItems: 'center' }}>
          {this.shouldShowViewAsTableCheckbox && (
            <Checkbox
              isChecked={this.state.viewAsTable}
              onChange={() =>
                this.setState({
                  viewAsTable: !this.state.viewAsTable,
                })
              }
            >
              <FormattedMessage
                defaultMessage="View as table"
                description="Experiment tracking > Artifact view > View as table checkbox"
              />
            </Checkbox>
          )}
          <Tooltip
            arrowPointAtCenter
            placement="topLeft"
            title={this.props.intl.formatMessage({
              defaultMessage: 'Download artifact',
              description: 'Link to download the artifact of the experiment',
            })}
          >
            <Button
              componentId="codegen_mlflow_app_src_experiment-tracking_components_artifactview.tsx_337"
              icon={<DownloadIcon />}
              onClick={() => this.onDownloadClick(runUuid, activeNodeId)}
            />
          </Tooltip>
        </div>
      </div>
    );
  }

  renderArtifactInfo() {
    const existingModelVersions = this.getExistingModelVersions();
    let toRender;
    if (existingModelVersions && Utils.isModelRegistryEnabled()) {
      // note that this case won't trigger for files inside a registered model/model version folder
      // React searches for existing model versions under the path of the file, which won't exist.
      toRender = this.renderModelVersionInfoSection(existingModelVersions);
    } else if (this.activeNodeCanBeRegistered() && Utils.isModelRegistryEnabled()) {
      toRender = this.renderRegisterModelButton();
    } else if (this.activeNodeIsDirectory()) {
      toRender = null;
    } else {
      toRender = this.renderDownloadLink();
    }
    const { theme } = this.props.designSystemThemeApi;
    return (
      <div
        css={{
          height: theme.general.heightBase + theme.spacing.md,
          backgroundColor: theme.colors.backgroundSecondary,
          padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
          whiteSpace: 'nowrap',
          display: 'flex',
        }}
      >
        {this.renderPathAndSizeInfo()}
        <div className="artifact-info-right">{toRender}</div>
      </div>
    );
  }

  renderArtifactInfoV2() {
    const existingModelVersions = this.getExistingModelVersions();
    let toRender;
    if (existingModelVersions && Utils.isModelRegistryEnabled()) {
      // note that this case won't trigger for files inside a registered model/model version folder
      // React searches for existing model versions under the path of the file, which won't exist.
      toRender = this.renderModelVersionInfoSection(existingModelVersions);
    } else if (this.activeNodeCanBeRegistered() && Utils.isModelRegistryEnabled()) {
      toRender = this.renderRegisterModelButton();
    } else if (this.activeNodeIsDirectory()) {
      toRender = null;
    } else {
      toRender = this.renderControls();
    }
    const { theme } = this.props.designSystemThemeApi;
    return (
      <>
        <div
          css={{
            padding: `${theme.spacing.sm}px ${theme.spacing.sm}px ${theme.spacing.sm}px ${theme.spacing.md}px`,
            whiteSpace: 'nowrap',
            display: 'flex',
            justifyContent: 'space-between',
          }}
        >
          {this.renderPathAndSizeInfoV2()}
          {toRender}
        </div>
      </>
    );
  }

  onToggleTreebeard = (
    dataNode: {
      id: string;
      loading: boolean;
    },
    toggled: boolean,
  ) => {
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

  getTreebeardData = (artifactNode: any): any => {
    const { isRoot } = artifactNode;
    if (isRoot) {
      if (artifactNode.children) {
        return Object.values(artifactNode.children).map((c) => this.getTreebeardData(c));
      }
      // This case should never happen since we should never call this function on an empty root.
      throw Error('unreachable code.');
    }

    let id;
    let name;
    let toggled;
    let children;
    let active;

    if (artifactNode.fileInfo) {
      const { path } = artifactNode.fileInfo;
      id = path;
      name = getBasename(path);
    }

    // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
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
      return parseInt(size, 10);
    }
    return 0;
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

  activeNodeCanBeRegistered() {
    if (this.state.activeNodeId) {
      const node = ArtifactUtils.findChild(this.props.artifactNode, this.state.activeNodeId);
      if (node && node.children && MLMODEL_FILE_NAME in node.children) {
        return true;
      }
    }
    return false;
  }

  componentDidUpdate(prevProps: ArtifactViewImplProps, prevState: ArtifactViewImplState) {
    const { activeNodeId } = this.state;
    if (prevState.activeNodeId !== activeNodeId) {
      this.props.handleActiveNodeChange(this.activeNodeIsDirectory());
    }
  }

  componentDidMount() {
    if (this.props.initialSelectedArtifactPath) {
      const artifactPathParts = this.props.initialSelectedArtifactPath.split('/');
      if (artifactPathParts) {
        try {
          // Check if valid artifactId was supplied in URL. If not, don't select
          // or expand anything.
          ArtifactUtils.findChild(this.props.artifactNode, this.props.initialSelectedArtifactPath);
        } catch (err) {
          console.error(err);
          return;
        }
      }
      let pathSoFar = '';
      const toggledArtifactState = {
        activeNodeId: this.props.initialSelectedArtifactPath,
        toggledNodeIds: {},
      };
      artifactPathParts.forEach((part) => {
        pathSoFar += part;
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        toggledArtifactState['toggledNodeIds'][pathSoFar] = true;
        pathSoFar += '/';
      });
      this.setArtifactState(toggledArtifactState);
    }
  }

  setArtifactState(artifactState: any) {
    this.setState(artifactState);
  }

  get shouldShowViewAsTableCheckbox() {
    return (
      this.state.activeNodeId &&
      this.props.runTags &&
      getLoggedTablesFromTags(this.props.runTags).includes(this.state.activeNodeId)
    );
  }

  render() {
    if (ArtifactUtils.isEmpty(this.props.artifactNode)) {
      return <NoArtifactView useAutoHeight={this.props.useAutoHeight} />;
    }
    const { theme } = this.props.designSystemThemeApi;

    if (shouldEnableLoggedArtifactTableView()) {
      return (
        <div
          className="artifact-view"
          css={{
            flex: this.props.useAutoHeight ? 1 : 'unset',
            height: this.props.useAutoHeight ? 'auto' : undefined,
          }}
        >
          <div
            style={{
              minWidth: '200px',
              maxWidth: '400px',
              flex: 1,
              whiteSpace: 'nowrap',
              borderRight: `1px solid ${theme.colors.borderDecorative}`,
            }}
          >
            <ArtifactViewTree
              data={this.getTreebeardData(this.props.artifactNode)}
              onToggleTreebeard={this.onToggleTreebeard}
            />
          </div>
          <div className="artifact-right">
            {this.state.activeNodeId ? this.renderArtifactInfoV2() : null}
            <ShowArtifactPage
              runUuid={this.props.runUuid}
              path={this.state.activeNodeId}
              isDirectory={this.activeNodeIsDirectory()}
              size={this.getActiveNodeSize()}
              runTags={this.props.runTags}
              artifactRootUri={this.props.artifactRootUri}
              modelVersions={this.props.modelVersions}
              showArtifactLoggedTableView={this.state.viewAsTable && this.shouldShowViewAsTableCheckbox}
            />
          </div>
        </div>
      );
    }

    return (
      <div
        className="artifact-view"
        css={{
          border: `1px solid ${theme.colors.borderDecorative}`,
          flex: this.props.useAutoHeight ? 1 : 'unset',
          height: this.props.useAutoHeight ? 'auto' : undefined,
        }}
      >
        <div className="artifact-left">
          <ArtifactViewTree
            data={this.getTreebeardData(this.props.artifactNode)}
            onToggleTreebeard={this.onToggleTreebeard}
          />
        </div>
        <div className="artifact-right">
          {this.state.activeNodeId ? this.renderArtifactInfo() : null}
          <ShowArtifactPage
            runUuid={this.props.runUuid}
            path={this.state.activeNodeId}
            isDirectory={this.activeNodeIsDirectory()}
            size={this.getActiveNodeSize()}
            runTags={this.props.runTags}
            artifactRootUri={this.props.artifactRootUri}
            modelVersions={this.props.modelVersions}
          />
        </div>
      </div>
    );
  }
}

const mapStateToProps = (state: any, ownProps: any) => {
  const { runUuid } = ownProps;
  const { apis } = state;
  const artifactNode = getArtifacts(runUuid, state);
  const artifactRootUri = getArtifactRootUri(runUuid, state);
  const modelVersions = getAllModelVersions(state);
  const modelVersionsWithNormalizedSource = _.flatMap(modelVersions, (version) => {
    // @ts-expect-error TS(2698): Spread types may only be created from object types... Remove this comment to see the full error message
    return { ...version, source: Utils.normalize((version as any).source) };
  });
  const modelVersionsBySource = _.groupBy(modelVersionsWithNormalizedSource, 'source');
  return { artifactNode, artifactRootUri, modelVersions, modelVersionsBySource, apis };
};

const mapDispatchToProps = {
  listArtifactsApi,
};

export const ArtifactView = connect(
  mapStateToProps,
  mapDispatchToProps,
  // @ts-expect-error TS(2769): No overload matches this call.
)(WithDesignSystemThemeHoc(injectIntl(ArtifactViewImpl)));

type ModelVersionInfoSectionProps = {
  modelVersion: any;
};

function ModelVersionInfoSection(props: ModelVersionInfoSectionProps) {
  const { modelVersion } = props;
  const { name, version, status, status_message } = modelVersion;

  // eslint-disable-next-line prefer-const
  let mvPageRoute = Utils.getIframeCorrectedRoute(ModelRegistryRoutes.getModelVersionPageRoute(name, version));
  const modelVersionLink = (
    <Tooltip title={`${name} version ${version}`}>
      <a href={mvPageRoute} className="model-version-link" target="_blank" rel="noreferrer">
        <span className="model-name">{name}</span>
        <span>,&nbsp;v{version}&nbsp;</span>
        <i className="fas fa-external-link-o" />
      </a>
    </Tooltip>
  );

  return (
    <div className="model-version-info">
      <div className="model-version-link-section">
        <Tooltip title={status_message || modelVersionStatusIconTooltips[status]}>
          <div>{ModelVersionStatusIcons[status]}</div>
        </Tooltip>
        {modelVersionLink}
      </div>
      <div className="model-version-status-text">
        {status === ModelVersionStatus.READY ? (
          <React.Fragment>
            <FormattedMessage
              defaultMessage="Registered on {registeredDate}"
              description="Label to display at what date the model was registered"
              values={{
                registeredDate: Utils.formatTimestamp(modelVersion.creation_timestamp, 'yyyy/mm/dd'),
              }}
            />
          </React.Fragment>
        ) : (
          status_message || DefaultModelVersionStatusMessages[status]
        )}
      </div>
    </div>
  );
}

function NoArtifactView({ useAutoHeight }: { useAutoHeight?: boolean }) {
  return (
    <div
      className="empty-artifact-outer-container"
      css={{
        flex: useAutoHeight ? 1 : 'unset',
        height: useAutoHeight ? 'auto' : undefined,
      }}
    >
      <div className="empty-artifact-container">
        <div>{/* TODO: put a nice image here */}</div>
        <div>
          <div className="no-artifacts">
            <FormattedMessage
              defaultMessage="No Artifacts Recorded"
              description="Empty state string when there are no artifacts record for the experiment"
            />
          </div>
          <div className="no-artifacts-info">
            <FormattedMessage
              defaultMessage="Use the log artifact APIs to store file outputs from MLflow runs."
              // eslint-disable-next-line max-len
              description="Information in the empty state explaining how one could log artifacts output files for the experiment runs"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
