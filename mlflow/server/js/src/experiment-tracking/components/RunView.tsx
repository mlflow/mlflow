/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component, createRef } from 'react';
import { FormattedMessage, injectIntl } from 'react-intl';
import {
  getExperiment,
  getParams,
  getRunInfo,
  getRunTags,
  getRunDatasets,
} from '../reducers/Reducers';
import { connect } from 'react-redux';
import './RunView.css';
import { HtmlTableView } from './HtmlTableView';
import { Link } from '../../common/utils/RoutingUtils';
import Routes from '../routes';
import ArtifactPage from './ArtifactPage';
import { getLatestMetrics } from '../reducers/MetricReducer';
import { Experiment } from '../sdk/MlflowMessages';
import Utils from '../../common/utils/Utils';
import { capitalizeFirstLetter } from '../../common/utils/StringUtils';
import { NOTE_CONTENT_TAG, NoteInfo } from '../utils/NoteUtils';
import { RenameRunModal } from './modals/RenameRunModal';
import { EditableTagsTableView } from '../../common/components/EditableTagsTableView';
import { Button, withNotifications } from '@databricks/design-system';
import { CollapsibleSection } from '../../common/components/CollapsibleSection';
import { EditableNote } from '../../common/components/EditableNote';
import { setTagApi, deleteTagApi } from '../actions';
import { PageHeader, OverflowMenu } from '../../shared/building_blocks/PageHeader';
import { Descriptions } from '../../common/components/Descriptions';
import { ExperimentViewDatasetWithContext } from './experiment-page/components/runs/ExperimentViewDatasetWithContext';
import { RunDatasetWithTags } from 'experiment-tracking/types';
import {
  DatasetWithRunType,
  ExperimentViewDatasetDrawer,
} from './experiment-page/components/runs/ExperimentViewDatasetDrawer';
import { shouldEnableExperimentDatasetTracking } from '../../common/utils/FeatureUtils';

type RunViewImplProps = {
  runUuid: string;
  run: any;
  experiment: any; // TODO: PropTypes.instanceOf(Experiment)
  experimentId: string;
  comparedExperimentIds?: string[];
  hasComparedExperimentsBefore?: boolean;
  params: any;
  tags: any;
  latestMetrics: any;
  datasets: any;
  getMetricPagePath: (...args: any[]) => any;
  runDisplayName: string;
  runName: string;
  handleSetRunTag: (...args: any[]) => any;
  setTagApi: (...args: any[]) => any;
  deleteTagApi: (...args: any[]) => any;
  intl: {
    formatMessage: (...args: any[]) => any;
  };
  notificationContextHolder: React.ReactNode;
  notificationAPI: any;
};

type RunViewImplState = any;

export class RunViewImpl extends Component<RunViewImplProps, RunViewImplState> {
  state = {
    showRunRenameModal: false,
    showNoteEditor: false,
    showTags: Utils.getVisibleTagValues(this.props.tags).length > 0,
    isTagsRequestPending: false,
    isDrawerOpen: false,
    selectedDatasetWithRun: null,
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

  setIsDrawerOpen = (isDrawerOpen: boolean) => {
    this.setState({ isDrawerOpen });
  };

  setSelectedDatasetWithRun = (datasetWithRun: DatasetWithRunType) => {
    this.setState({ selectedDatasetWithRun: datasetWithRun });
  };

  handleAddTag = (values: any) => {
    const form = this.formRef.current;
    const { runUuid, notificationAPI, intl } = this.props;

    this.setState({ isTagsRequestPending: true });
    this.props
      .setTagApi(runUuid, values.name, values.value)
      .then(() => {
        this.setState({ isTagsRequestPending: false });
        (form as any).resetFields();
      })
      .catch((ex: any) => {
        this.setState({ isTagsRequestPending: false });
        console.error(ex);
        const errorMessage = intl.formatMessage(
          {
            defaultMessage: 'Failed to add tag. Error: {errorTrace}',
            description: 'Error message when add to tag feature fails',
          },
          { errorTrace: ex.getMessageField() },
        );
        notificationAPI.error({ message: errorMessage });
      });
  };

  handleSaveEdit = ({ name, value }: any) => {
    const { runUuid, notificationAPI, intl } = this.props;

    return this.props.setTagApi(runUuid, name, value).catch((ex: any) => {
      console.error(ex);
      const errorMessage = intl.formatMessage(
        {
          defaultMessage: 'Failed to set tag. Error: {errorTrace}',
          description: 'Error message when updating or setting a tag feature fails',
        },
        { errorTrace: ex.getMessageField() },
      );
      notificationAPI.error({ message: errorMessage });
    });
  };

  handleDeleteTag = ({ name }: any) => {
    const { runUuid, notificationAPI, intl } = this.props;
    return this.props.deleteTagApi(runUuid, name).catch((ex: any) => {
      console.error(ex);
      const errorMessage = intl.formatMessage(
        {
          defaultMessage: 'Failed to delete tag. Error: {errorTrace}',
          description: 'Error message when deleting a tag feature fails',
        },
        { errorTrace: ex.getMessageField() },
      );
      notificationAPI.error({ message: errorMessage });
    });
  };

  getRunCommand() {
    const { tags, params } = this.props;
    let runCommand: any = null;
    const sourceName = Utils.getSourceName(tags);
    const sourceVersion = Utils.getSourceVersion(tags);
    const entryPointName = Utils.getEntryPointName(tags);
    const backend = Utils.getBackend(tags);

    if (Utils.getSourceType(tags) === 'PIPELINE') {
      const profileName = Utils.getPipelineProfileName(tags);
      const stepName = Utils.getPipelineStepName(tags);
      runCommand = '';
      if (sourceName) {
        const repoName = Utils.dropExtension(Utils.baseName(sourceName));
        runCommand += `git clone ${sourceName}\n`;
        runCommand += `cd ${repoName}\n`;
      }

      if (sourceVersion) {
        runCommand += `git checkout ${sourceVersion}\n`;
      }

      runCommand += `mlflow recipes run -p ${shellEscape(profileName)}`;

      if (stepName) {
        runCommand += ' -s ' + shellEscape(stepName);
      }
    }

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
          runCommand += ' -P ' + shellEscape((p as any).key + '=' + (p as any).value);
        });
    }
    return runCommand;
  }

  handleCancelEditNote = () => {
    this.setState({ showNoteEditor: false });
  };

  handleSubmitEditNote = (note: any) => {
    return this.props.handleSetRunTag(NOTE_CONTENT_TAG, note).then(() => {
      this.setState({ showNoteEditor: false });
    });
  };

  startEditingDescription = (e: any) => {
    e.stopPropagation();
    this.setState({ showNoteEditor: true });
  };

  static getRunStatusDisplayName(status: any) {
    return status !== 'RUNNING' ? status : 'UNFINISHED';
  }

  handleCollapseChange(_sectionName: any) {
    return undefined;
  }

  renderSectionTitle(title: any, count = 0) {
    if (count === 0) {
      return title;
    }

    return (
      <>
        {title} ({count})
      </>
    );
  }

  renderUserIdLink = (run: any, tags: any, experiment: any) => {
    const user = Utils.getUser(run, tags);
    return <Link to={Routes.searchRunsByUser(experiment.experiment_id, user)}>{user}</Link>;
  };

  renderLifecycleLink = (experiment: any) => {
    const lifecycleStage = this.props.run.getLifecycleStage();
    return (
      <Link
        to={Routes.searchRunsByLifecycleStage(
          experiment.experiment_id,
          capitalizeFirstLetter(lifecycleStage),
        )}
      >
        {lifecycleStage}
      </Link>
    );
  };

  getExperimentPageLink() {
    return this.props.hasComparedExperimentsBefore ? (
      <Link to={Routes.getCompareExperimentsPageRoute(this.props.comparedExperimentIds)}>
        <FormattedMessage
          defaultMessage='Displaying Runs from {numExperiments} Experiments'
          // eslint-disable-next-line max-len
          description='Breadcrumb nav item to link to the compare-experiments page on compare runs page'
          values={{
            // @ts-expect-error TS(2532): Object is possibly 'undefined'.
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
      experiment,
      params,
      tags,
      latestMetrics,
      datasets,
      getMetricPagePath,
      notificationContextHolder,
      runName,
      experimentId,
    } = this.props;
    const { showNoteEditor, isTagsRequestPending, isDrawerOpen, selectedDatasetWithRun } =
      this.state;
    const noteInfo = NoteInfo.fromTags(tags);
    const startTime = run.getStartTime() ? Utils.formatTimestamp(run.getStartTime()) : '(unknown)';
    const duration = Utils.getDuration(run.getStartTime(), run.getEndTime());
    const status = RunViewImpl.getRunStatusDisplayName(run.getStatus());
    const lifecycleStage = run.getLifecycleStage();
    const queryParams = window.location && window.location.search ? window.location.search : '';
    const runCommand = this.getRunCommand();
    const noteContent = noteInfo && noteInfo.content;
    const breadcrumbs = [this.getExperimentPageLink()];
    const plotTitle = this.props.intl.formatMessage({
      defaultMessage: 'Plot chart',
      description: 'Link to the view the plot chart for the experiment run',
    });

    return (
      <div className='RunView'>
        <PageHeader
          title={<span data-test-id='runs-header'>{this.props.runDisplayName}</span>}
          breadcrumbs={breadcrumbs}
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
            // @ts-expect-error TS(2322): Type '{ runUuid: string; onClose: () => void; runN... Remove this comment to see the full error message
            runUuid={runUuid}
            onClose={this.hideRenameRunModal}
            runName={this.props.runName}
            isOpen={this.state.showRunRenameModal}
          />
        </div>

        {/* Metadata List */}
        {/* @ts-expect-error TS(2322): Type '{ children: (Element | null)[]; className: s... Remove this comment to see the full error message */}
        <Descriptions className='metadata-list'>
          <Descriptions.Item
            label={this.props.intl.formatMessage({
              defaultMessage: 'Run ID',
              description: 'Label for displaying the ID of the experiment run',
            })}
          >
            {runUuid}
          </Descriptions.Item>
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
            {this.renderUserIdLink(run, tags, experiment)}
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
              {this.renderLifecycleLink(experiment)}
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
              <textarea
                css={styles.runCommandArea}
                // Setting row count basing on the number of line breaks
                rows={(runCommand.match(/\n/g) || []).length + 1}
                value={runCommand}
              />
            </CollapsibleSection>
          ) : null}
          <CollapsibleSection
            title={
              <span className='RunView-editDescriptionHeader'>
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
              // @ts-expect-error TS(2322): Type '{ defaultMarkdown: any; onSubmit: (note: any... Remove this comment to see the full error message
              defaultMarkdown={noteContent}
              onSubmit={this.handleSubmitEditNote}
              onCancel={this.handleCancelEditNote}
              showEditor={showNoteEditor}
            />
          </CollapsibleSection>
          {shouldEnableExperimentDatasetTracking() && (
            <CollapsibleSection
              defaultCollapsed
              title={this.renderSectionTitle(
                this.props.intl.formatMessage({
                  defaultMessage: 'Datasets',
                  description:
                    // eslint-disable-next-line max-len
                    'Label for the collapsible area to display the datasets used during the experiment run',
                }),
                datasets ? datasets.length : 0,
              )}
              onChange={this.handleCollapseChange('parameters')}
              data-test-id='run-parameters-section'
            >
              <div css={{ marginLeft: '16px' }}>
                {datasets &&
                  datasets.map((dataset: RunDatasetWithTags) => (
                    <div
                      key={`${dataset.dataset.name}-${dataset.dataset.digest}`}
                      css={{
                        display: 'flex',
                        flexDirection: 'column',
                        justifyContent: 'center',
                      }}
                    >
                      <Button
                        type='link'
                        css={{
                          textAlign: 'left',
                        }}
                        onClick={() => {
                          this.setSelectedDatasetWithRun({
                            datasetWithTags: dataset,
                            runData: {
                              experimentId: experimentId,
                              runUuid: runUuid,
                              runName: runName,
                              datasets: datasets,
                              tags: tags,
                            },
                          });
                          this.setIsDrawerOpen(true);
                        }}
                      >
                        <ExperimentViewDatasetWithContext
                          datasetWithTags={dataset}
                          displayTextAsLink
                        />
                      </Button>
                    </div>
                  ))}
              </div>
            </CollapsibleSection>
          )}
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
                // @ts-expect-error TS(2322): Type '{ innerRef: RefObject<unknown>; handleAddTag... Remove this comment to see the full error message
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
            <ArtifactPage runUuid={runUuid} runTags={tags} />
          </CollapsibleSection>
        </div>
        {notificationContextHolder}
        {selectedDatasetWithRun && (
          <ExperimentViewDatasetDrawer
            isOpen={isDrawerOpen}
            setIsOpen={this.setIsDrawerOpen}
            selectedDatasetWithRun={selectedDatasetWithRun}
            setSelectedDatasetWithRun={this.setSelectedDatasetWithRun}
          />
        )}
      </div>
    );
  }

  handleSubmittedNote(err: any) {
    if (err) {
      // Do nothing; error is handled by the note editor view
    } else {
      // Successfully submitted note, close the editor
      this.setState({ showNotesEditor: false });
    }
  }
}

const mapStateToProps = (state: any, ownProps: any) => {
  const { comparedExperimentIds, hasComparedExperimentsBefore } = state.compareExperiments;
  const { runUuid, experimentId } = ownProps;
  const run = getRunInfo(runUuid, state);
  const experiment = getExperiment(experimentId, state);
  const params = getParams(runUuid, state);
  const tags = getRunTags(runUuid, state);
  const latestMetrics = getLatestMetrics(runUuid, state);
  const datasets = getRunDatasets(runUuid, state);
  const runDisplayName = Utils.getRunDisplayName(run, runUuid);
  // @ts-expect-error TS(2554): Expected 1 arguments, but got 2.
  const runName = Utils.getRunName(run, runUuid);
  return {
    run,
    experiment,
    params,
    tags,
    latestMetrics,
    datasets,
    runDisplayName,
    runName,
    comparedExperimentIds,
    hasComparedExperimentsBefore,
  };
};
const mapDispatchToProps = { setTagApi, deleteTagApi };

// @ts-expect-error TS(2769): No overload matches this call.
export const RunViewImplWithIntl = withNotifications(injectIntl(RunViewImpl));
export const RunView = connect(mapStateToProps, mapDispatchToProps)(RunViewImplWithIntl);

// Private helper functions.

const getParamValues = (params: any) => {
  return Object.values(params)
    .sort()
    .map((p, index) => ({
      key: `params-${index}`,
      name: (p as any).getKey(),
      value: (p as any).getValue(),
    }));
};

const getMetricValues = (latestMetrics: any, getMetricPagePath: any, plotTitle: any) => {
  return (
    Object.values(latestMetrics)
      .sort()
      // @ts-expect-error TS(2345): Argument of type '({ key, value }: { key: any; val... Remove this comment to see the full error message
      .map(({ key, value }, index) => {
        return {
          key: `metrics-${index}`,
          name: (
            <Link to={getMetricPagePath(key)} title={plotTitle}>
              {key}
              <i className='fas fa-line-chart' style={{ paddingLeft: '6px' }} />
            </Link>
          ),
          value: <span title={value}>{Utils.formatMetric(value)}</span>,
        };
      })
  );
};

const shellEscape = (str: any) => {
  if (/["\r\n\t ]/.test(str)) {
    return '"' + str.replace(/"/g, '\\"') + '"';
  }
  return str;
};

const styles = {
  runCommandArea: (theme: any) => ({
    fontFamily: 'Menlo, Consolas, monospace',
    width: '100%',
  }),
};
