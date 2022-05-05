import React, { Component, createRef } from 'react';
import PropTypes from 'prop-types';
import { FormattedMessage, injectIntl } from 'react-intl';
import { getExperiment, getParams, getRunInfo, getRunTags } from '../reducers/Reducers';
import { connect } from 'react-redux';
import './RunView.css';
import { HtmlTableView } from './HtmlTableView';
import { Link } from 'react-router-dom';
import Routes from '../routes';
import ArtifactPage from './ArtifactPage';
import { getLatestMetrics } from '../reducers/MetricReducer';
import { Experiment } from '../sdk/MlflowMessages';
import Utils from '../../common/utils/Utils';
import { NOTE_CONTENT_TAG, NoteInfo } from '../utils/NoteUtils';
import { RenameRunModal } from './modals/RenameRunModal';
import { EditableTagsTableView } from '../../common/components/EditableTagsTableView';
import { Button, Descriptions, message } from 'antd';
import { CollapsibleSection } from '../../common/components/CollapsibleSection';
import { EditableNote } from '../../common/components/EditableNote';
import { setTagApi, deleteTagApi } from '../actions';
import { PageHeader, OverflowMenu } from '../../shared/building_blocks/PageHeader';

export class RunViewImpl extends Component {
  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    run: PropTypes.object.isRequired,
    experiment: PropTypes.instanceOf(Experiment).isRequired,
    experimentId: PropTypes.string.isRequired,
    comparedExperimentIds: PropTypes.arrayOf(PropTypes.string),
    hasComparedExperimentsBefore: PropTypes.bool,
    params: PropTypes.object.isRequired,
    tags: PropTypes.object.isRequired,
    latestMetrics: PropTypes.object.isRequired,
    getMetricPagePath: PropTypes.func.isRequired,
    runDisplayName: PropTypes.string.isRequired,
    runName: PropTypes.string.isRequired,
    handleSetRunTag: PropTypes.func.isRequired,
    setTagApi: PropTypes.func.isRequired,
    deleteTagApi: PropTypes.func.isRequired,
    modelVersions: PropTypes.arrayOf(PropTypes.object),
    intl: PropTypes.shape({ formatMessage: PropTypes.func.isRequired }).isRequired,
  };

  state = {
    showRunRenameModal: false,
    showNoteEditor: false,
    showTags: Utils.getVisibleTagValues(this.props.tags).length > 0,
    isTagsRequestPending: false,
  };

  formRef = createRef();

  componentDidMount() {
    const pageTitle = `${this.props.runDisplayName} - MLflow Run`;
    Utils.updatePageTitle(pageTitle);
  }

  handleRenameRunClick = () => {
    this.setState({ showRunRenameModal: true });
  };

  hideRenameRunModal = () => {
    this.setState({ showRunRenameModal: false });
  };

  handleAddTag = (values) => {
    const form = this.formRef.current;
    const { runUuid } = this.props;

    this.setState({ isTagsRequestPending: true });
    this.props
      .setTagApi(runUuid, values.name, values.value)
      .then(() => {
        this.setState({ isTagsRequestPending: false });
        form.resetFields();
      })
      .catch((ex) => {
        this.setState({ isTagsRequestPending: false });
        console.error(ex);
        const errorMessage = (
          <FormattedMessage
            defaultMessage='Failed to add tag. Error: {errorTrace}'
            description='Error message when add to tag feature fails'
            values={{ errorTrace: ex.getUserVisibleError() }}
          />
        );
        message.error(errorMessage);
      });
  };

  handleSaveEdit = ({ name, value }) => {
    const { runUuid } = this.props;
    return this.props.setTagApi(runUuid, name, value).catch((ex) => {
      console.error(ex);
      const errorMessage = (
        <FormattedMessage
          defaultMessage='Failed to set tag. Error: {errorTrace}'
          description='Error message when updating or setting a tag feature fails'
          values={{ errorTrace: ex.getUserVisibleError() }}
        />
      );
      message.error(errorMessage);
    });
  };

  handleDeleteTag = ({ name }) => {
    const { runUuid } = this.props;
    return this.props.deleteTagApi(runUuid, name).catch((ex) => {
      console.error(ex);
      const errorMessage = (
        <FormattedMessage
          defaultMessage='Failed to delete tag. Error: {errorTrace}'
          description='Error message when deleting a tag feature fails'
          values={{ errorTrace: ex.getUserVisibleError() }}
        />
      );
      message.error(errorMessage);
    });
  };

  getRunCommand() {
    const { tags, params } = this.props;
    let runCommand = null;
    const sourceName = Utils.getSourceName(tags);
    const sourceVersion = Utils.getSourceVersion(tags);
    const entryPointName = Utils.getEntryPointName(tags);
    const backend = Utils.getBackend(tags);
    if (Utils.getSourceType(tags) === 'PROJECT') {
      runCommand = 'mlflow run ' + shellEscape(sourceName);
      if (sourceVersion && sourceVersion !== 'latest') {
        runCommand += ' -v ' + shellEscape(sourceVersion);
      }
      if (entryPointName && entryPointName !== 'main') {
        runCommand += ' -e ' + shellEscape(entryPointName);
      }
      if (backend) {
        runCommand += ' -b ' + shellEscape(backend);
      }
      Object.values(params)
        .sort()
        .forEach((p) => {
          runCommand += ' -P ' + shellEscape(p.key + '=' + p.value);
        });
    }
    return runCommand;
  }

  handleCancelEditNote = () => {
    this.setState({ showNoteEditor: false });
  };

  handleSubmitEditNote = (note) => {
    return this.props.handleSetRunTag(NOTE_CONTENT_TAG, note).then(() => {
      this.setState({ showNoteEditor: false });
    });
  };

  startEditingDescription = (e) => {
    e.stopPropagation();
    this.setState({ showNoteEditor: true });
  };

  static getRunStatusDisplayName(status) {
    return status !== 'RUNNING' ? status : 'UNFINISHED';
  }

  handleCollapseChange() {}

  renderSectionTitle(title, count = 0) {
    if (count === 0) {
      return title;
    }

    return (
      <>
        {title} ({count})
      </>
    );
  }

  getExperimentPageLink() {
    return this.props.hasComparedExperimentsBefore ? (
      <Link to={Routes.getCompareExperimentsPageRoute(this.props.comparedExperimentIds)}>
        <FormattedMessage
          defaultMessage='Displaying Runs from {numExperiments} Experiments'
          // eslint-disable-next-line max-len
          description='Breadcrumb nav item to link to the compare-experiments page on compare runs page'
          values={{
            numExperiments: this.props.comparedExperimentIds.length,
          }}
        />
      </Link>
    ) : (
      <Link
        to={Routes.getExperimentPageRoute(this.props.experiment.experiment_id)}
        data-test-id='experiment-runs-link'
      >
        {this.props.experiment.getName()}
      </Link>
    );
  }

  render() {
    const {
      runUuid,
      run,
      params,
      tags,
      latestMetrics,
      getMetricPagePath,
      modelVersions,
    } = this.props;
    const { showNoteEditor, isTagsRequestPending } = this.state;
    const noteInfo = NoteInfo.fromTags(tags);
    const startTime = run.getStartTime() ? Utils.formatTimestamp(run.getStartTime()) : '(unknown)';
    const duration = Utils.getDuration(run.getStartTime(), run.getEndTime());
    const status = RunViewImpl.getRunStatusDisplayName(run.getStatus());
    const lifecycleStage = run.getLifecycleStage();
    const queryParams = window.location && window.location.search ? window.location.search : '';
    const runCommand = this.getRunCommand();
    const noteContent = noteInfo && noteInfo.content;
    const breadcrumbs = [this.getExperimentPageLink(), this.props.runDisplayName];
    /* eslint-disable prefer-const */
    let feedbackForm;
    const plotTitle = this.props.intl.formatMessage({
      defaultMessage: 'Plot chart',
      description: 'Link to the view the plot chart for the experiment run',
    });

    return (
      <div className='RunView'>
        <PageHeader
          title={<span data-test-id='runs-header'>{this.props.runDisplayName}</span>}
          breadcrumbs={breadcrumbs}
          feedbackForm={feedbackForm}
        >
          <OverflowMenu
            menu={[
              {
                id: 'overflow-rename-button',
                onClick: this.handleRenameRunClick,
                itemName: (
                  <FormattedMessage
                    defaultMessage='Rename'
                    description='Menu item to rename an experiment run'
                  />
                ),
              },
            ]}
          />
        </PageHeader>
        <div className='header-container'>
          <RenameRunModal
            runUuid={runUuid}
            onClose={this.hideRenameRunModal}
            runName={this.props.runName}
            isOpen={this.state.showRunRenameModal}
          />
        </div>

        {/* Metadata List */}
        <Descriptions className='metadata-list'>
          <Descriptions.Item
            label={this.props.intl.formatMessage({
              defaultMessage: 'Date',
              description: 'Label for displaying the start time of the experiment ran',
            })}
          >
            {startTime}
          </Descriptions.Item>
          <Descriptions.Item
            label={this.props.intl.formatMessage({
              defaultMessage: 'Source',
              description: 'Label for displaying source notebook of the experiment run',
            })}
          >
            <div style={{ display: 'flex', alignItems: 'center' }}>
              {Utils.renderSourceTypeIcon(tags)}
              {Utils.renderSource(tags, queryParams, runUuid)}
            </div>
          </Descriptions.Item>
          {Utils.getSourceVersion(tags) ? (
            <Descriptions.Item
              label={this.props.intl.formatMessage({
                defaultMessage: 'Git Commit',
                description: 'Label for displaying the tag or the commit hash of the git commit',
              })}
            >
              {Utils.renderVersion(tags, false)}
            </Descriptions.Item>
          ) : null}
          {Utils.getSourceType(tags) === 'PROJECT' ? (
            <Descriptions.Item
              label={this.props.intl.formatMessage({
                defaultMessage: 'Entry Point',
                description: 'Label for displaying entry point of the project',
              })}
            >
              {Utils.getEntryPointName(tags) || 'main'}
            </Descriptions.Item>
          ) : null}
          <Descriptions.Item
            label={this.props.intl.formatMessage({
              defaultMessage: 'User',
              description: 'Label for displaying the user who created the experiment run',
            })}
          >
            {Utils.getUser(run, tags)}
          </Descriptions.Item>
          {duration ? (
            <Descriptions.Item
              label={this.props.intl.formatMessage({
                defaultMessage: 'Duration',
                description: 'Label for displaying the duration of the experiment run',
              })}
            >
              {duration}
            </Descriptions.Item>
          ) : null}
          <Descriptions.Item
            label={this.props.intl.formatMessage({
              defaultMessage: 'Status',
              description:
                // eslint-disable-next-line max-len
                'Label for displaying status of the experiment run to see if its running or finished',
            })}
          >
            {status}
          </Descriptions.Item>
          {lifecycleStage ? (
            <Descriptions.Item
              label={this.props.intl.formatMessage({
                defaultMessage: 'Lifecycle Stage',
                description:
                  // eslint-disable-next-line max-len
                  'Label for displaying lifecycle stage of the experiment run to see if its active or deleted',
              })}
            >
              {lifecycleStage}
            </Descriptions.Item>
          ) : null}
          {tags['mlflow.parentRunId'] !== undefined ? (
            <Descriptions.Item
              label={this.props.intl.formatMessage({
                defaultMessage: 'Parent Run',
                description:
                  'Label for displaying a link to the parent experiment run if any present',
              })}
            >
              <Link
                to={Routes.getRunPageRoute(
                  this.props.experimentId,
                  tags['mlflow.parentRunId'].value,
                )}
              >
                {tags['mlflow.parentRunId'].value}
              </Link>
            </Descriptions.Item>
          ) : null}
          {tags['mlflow.databricks.runURL'] !== undefined ? (
            <Descriptions.Item
              label={this.props.intl.formatMessage({
                defaultMessage: 'Job Output',
                description: 'Label for displaying the output logs for the experiment run job',
              })}
            >
              {/* Reported during ESLint upgrade */}
              {/* eslint-disable-next-line react/jsx-no-target-blank */}
              <a
                href={Utils.setQueryParams(tags['mlflow.databricks.runURL'].value, queryParams)}
                target='_blank'
              >
                <FormattedMessage
                  defaultMessage='Logs'
                  description='Link to the logs for the job output'
                />
              </a>
            </Descriptions.Item>
          ) : null}
        </Descriptions>

        {/* Page Sections */}
        <div className='RunView-info'>
          {runCommand ? (
            <CollapsibleSection
              title={this.props.intl.formatMessage({
                defaultMessage: 'Run Command',
                description:
                  // eslint-disable-next-line max-len
                  'Label for the collapsible area to display the run command used for the experiment run',
              })}
              onChange={this.handleCollapseChange('runCommand')}
              data-test-id='run-command-section'
            >
              <textarea className='run-command text-area' readOnly value={runCommand} />
            </CollapsibleSection>
          ) : null}
          <CollapsibleSection
            title={
              <span>
                <FormattedMessage
                  defaultMessage='Description'
                  description='Label for the notes editable content for the experiment run'
                />
                {!showNoteEditor && (
                  <>
                    {' '}
                    <Button
                      type='link'
                      onClick={this.startEditingDescription}
                      data-test-id='edit-description-button'
                    >
                      <FormattedMessage
                        defaultMessage='Edit'
                        // eslint-disable-next-line max-len
                        description='Text for the edit button next to the description section title on the run view'
                      />
                    </Button>
                  </>
                )}
              </span>
            }
            forceOpen={showNoteEditor}
            defaultCollapsed={!noteContent}
            onChange={this.handleCollapseChange('notes')}
            data-test-id='run-notes-section'
          >
            <EditableNote
              defaultMarkdown={noteContent}
              onSubmit={this.handleSubmitEditNote}
              onCancel={this.handleCancelEditNote}
              showEditor={showNoteEditor}
            />
          </CollapsibleSection>
          <CollapsibleSection
            defaultCollapsed
            title={this.renderSectionTitle(
              this.props.intl.formatMessage({
                defaultMessage: 'Parameters',
                description:
                  // eslint-disable-next-line max-len
                  'Label for the collapsible area to display the parameters used during the experiment run',
              }),
              getParamValues(params).length,
            )}
            onChange={this.handleCollapseChange('parameters')}
            data-test-id='run-parameters-section'
          >
            <HtmlTableView
              testId='params-table'
              columns={[
                {
                  title: this.props.intl.formatMessage({
                    defaultMessage: 'Name',
                    description:
                      // eslint-disable-next-line max-len
                      'Column title for name column for displaying the params name for the experiment run',
                  }),
                  dataIndex: 'name',
                },
                {
                  title: this.props.intl.formatMessage({
                    defaultMessage: 'Value',
                    description:
                      // eslint-disable-next-line max-len
                      'Column title for value column for displaying the value of the params for the experiment run ',
                  }),
                  dataIndex: 'value',
                },
              ]}
              values={getParamValues(params)}
            />
          </CollapsibleSection>
          <CollapsibleSection
            defaultCollapsed
            title={this.renderSectionTitle(
              this.props.intl.formatMessage({
                defaultMessage: 'Metrics',
                description:
                  // eslint-disable-next-line max-len
                  'Label for the collapsible area to display the output metrics after the experiment run',
              }),
              getMetricValues(latestMetrics, getMetricPagePath, plotTitle).length,
            )}
            onChange={this.handleCollapseChange('metrics')}
            data-test-id='run-metrics-section'
          >
            <HtmlTableView
              testId='metrics-table'
              columns={[
                {
                  title: this.props.intl.formatMessage({
                    defaultMessage: 'Name',
                    description:
                      // eslint-disable-next-line max-len
                      'Column title for name column for displaying the metrics name for the experiment run',
                  }),
                  dataIndex: 'name',
                },
                {
                  title: this.props.intl.formatMessage({
                    defaultMessage: 'Value',
                    description:
                      // eslint-disable-next-line max-len
                      'Column title for value column for displaying the value of the metrics for the experiment run ',
                  }),
                  dataIndex: 'value',
                },
              ]}
              values={getMetricValues(latestMetrics, getMetricPagePath, plotTitle)}
            />
          </CollapsibleSection>
          <div data-test-id='tags-section'>
            <CollapsibleSection
              title={this.renderSectionTitle(
                this.props.intl.formatMessage({
                  defaultMessage: 'Tags',
                  description:
                    'Label for the collapsible area to display the tags for the experiment run',
                }),
                Utils.getVisibleTagValues(tags).length,
              )}
              defaultCollapsed
              onChange={this.handleCollapseChange('tags')}
              data-test-id='run-tags-section'
            >
              <EditableTagsTableView
                innerRef={this.formRef}
                handleAddTag={this.handleAddTag}
                handleDeleteTag={this.handleDeleteTag}
                handleSaveEdit={this.handleSaveEdit}
                tags={tags}
                isRequestPending={isTagsRequestPending}
              />
            </CollapsibleSection>
          </div>
          <CollapsibleSection
            title={this.props.intl.formatMessage({
              defaultMessage: 'Artifacts',
              description:
                // eslint-disable-next-line max-len
                'Label for the collapsible area to display the artifacts page',
            })}
            onChange={this.handleCollapseChange('artifacts')}
            data-test-id='run-artifacts-section'
          >
            <ArtifactPage runUuid={runUuid} modelVersions={modelVersions} runTags={tags} />
          </CollapsibleSection>
        </div>
      </div>
    );
  }

  handleSubmittedNote(err) {
    if (err) {
      // Do nothing; error is handled by the note editor view
    } else {
      // Successfully submitted note, close the editor
      this.setState({ showNotesEditor: false });
    }
  }
}

const mapStateToProps = (state, ownProps) => {
  const { comparedExperimentIds, hasComparedExperimentsBefore } = state.compareExperiments;
  const { runUuid, experimentId } = ownProps;
  const run = getRunInfo(runUuid, state);
  const experiment = getExperiment(experimentId, state);
  const params = getParams(runUuid, state);
  const tags = getRunTags(runUuid, state);
  const latestMetrics = getLatestMetrics(runUuid, state);
  const runDisplayName = Utils.getRunDisplayName(tags, runUuid);
  const runName = Utils.getRunName(tags, runUuid);
  return {
    run,
    experiment,
    params,
    tags,
    latestMetrics,
    runDisplayName,
    runName,
    comparedExperimentIds,
    hasComparedExperimentsBefore,
  };
};
const mapDispatchToProps = { setTagApi, deleteTagApi };

export const RunViewImplWithIntl = injectIntl(RunViewImpl);
export const RunView = connect(mapStateToProps, mapDispatchToProps)(RunViewImplWithIntl);

// Private helper functions.

const getParamValues = (params) => {
  return Object.values(params)
    .sort()
    .map((p, index) => ({ key: `params-${index}`, name: p.getKey(), value: p.getValue() }));
};

const getMetricValues = (latestMetrics, getMetricPagePath, plotTitle) => {
  return Object.values(latestMetrics)
    .sort()
    .map(({ key, value }, index) => {
      return {
        key: `metrics-${index}`,
        name: (
          <Link to={getMetricPagePath(key)} title={plotTitle}>
            {key}
            <i className='fas fa-chart-line' style={{ paddingLeft: '6px' }} />
          </Link>
        ),
        value: <span title={value}>{Utils.formatMetric(value)}</span>,
      };
    });
};

const shellEscape = (str) => {
  if (/["\r\n\t ]/.test(str)) {
    return '"' + str.replace(/"/g, '\\"') + '"';
  }
  return str;
};
