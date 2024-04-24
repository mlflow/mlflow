import { Button, type NotificationInstance } from '@databricks/design-system';
import { useCallback, useRef, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { CollapsibleSection } from '../../../common/components/CollapsibleSection';
import { Descriptions } from '../../../common/components/Descriptions';
import { EditableNote } from '../../../common/components/EditableNote';
import { EditableTagsTableView } from '../../../common/components/EditableTagsTableView';
import { shouldEnableDeepLearningUI } from '../../../common/utils/FeatureUtils';
import { Link } from '../../../common/utils/RoutingUtils';
import { capitalizeFirstLetter } from '../../../common/utils/StringUtils';
import Utils from '../../../common/utils/Utils';
import Routes from '../../routes';
import type { ExperimentEntity, KeyValueEntity, MetricEntity, RunDatasetWithTags, RunInfoEntity } from '../../types';
import { NOTE_CONTENT_TAG, NoteInfo } from '../../utils/NoteUtils';
import ArtifactPage from '../ArtifactPage';
import { HtmlTableView } from '../HtmlTableView';
import {
  DatasetWithRunType,
  ExperimentViewDatasetDrawer,
} from '../experiment-page/components/runs/ExperimentViewDatasetDrawer';
import { ExperimentViewDatasetWithContext } from '../experiment-page/components/runs/ExperimentViewDatasetWithContext';

export interface RunViewOverviewProps {
  runUuid: string;
  runName: string;
  run: RunInfoEntity;
  experimentId: string;
  experiment: ExperimentEntity;
  params: Record<string, KeyValueEntity>;
  tags: Record<string, KeyValueEntity>;
  latestMetrics: Record<string, MetricEntity>;
  datasets: DatasetWithRunType['datasetWithTags'][];
  getMetricPagePath: (metric: string) => string;
  notificationContextHolder: React.ReactNode;
  notificationAPI: NotificationInstance;
  handleSetRunTag: (key: string, value: string) => Promise<void>;
  setTagApi: (runUuid: string, name: string, value: string) => Promise<void>;
  deleteTagApi: (runUuid: string, name: string) => Promise<void>;
}

export const RunViewOverview = ({
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
  handleSetRunTag,
  notificationAPI,
  setTagApi,
  deleteTagApi,
}: RunViewOverviewProps) => {
  const [showNoteEditor, setShowNoteEditor] = useState(false);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [selectedDatasetWithRun, setSelectedDatasetWithRun] = useState<DatasetWithRunType | null>(null);
  const [isTagsRequestPending, setIsTagsRequestPending] = useState(false);

  const handleCancelEditNote = () => {
    setShowNoteEditor(false);
  };

  const handleSubmitEditNote = (note: any) => {
    return handleSetRunTag(NOTE_CONTENT_TAG, note).then(() => {
      setShowNoteEditor(false);
    });
  };

  const startEditingDescription = (e: any) => {
    e.stopPropagation();
    setShowNoteEditor(true);
  };

  const intl = useIntl();
  const renderLifecycleLink = () => {
    const lifecycleStage = run.getLifecycleStage();
    return (
      <Link to={Routes.searchRunsByLifecycleStage(experiment.experiment_id, capitalizeFirstLetter(lifecycleStage))}>
        {lifecycleStage}
      </Link>
    );
  };

  const formRef = useRef<any>();

  const handleSaveEdit = ({ name, value }: any) => {
    return setTagApi(runUuid, name, value).catch((ex: any) => {
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

  const handleDeleteTag = ({ name }: any) => {
    return deleteTagApi(runUuid, name).catch((ex: any) => {
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

  const handleAddTag = (values: any) => {
    const form = formRef.current;

    setIsTagsRequestPending(true);

    setTagApi(runUuid, values.name, values.value)
      .then(() => {
        setIsTagsRequestPending(false);
        (form as any).resetFields();
      })
      .catch((ex: any) => {
        setIsTagsRequestPending(false);
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

  const handleCollapseChange = (_sectionName: string) => {
    return undefined;
  };

  const renderArtifactSection = () => {
    // After enabling deep learning flag, the artifact browser
    // resides in a dedicated tab
    if (shouldEnableDeepLearningUI()) {
      return null;
    }
    return (
      <CollapsibleSection
        title={intl.formatMessage({
          defaultMessage: 'Artifacts',
          description:
            // eslint-disable-next-line max-len
            'Label for the collapsible area to display the artifacts page',
        })}
        onChange={handleCollapseChange('artifacts')}
        data-test-id="run-artifacts-section"
      >
        <ArtifactPage runUuid={runUuid} runTags={tags} />
      </CollapsibleSection>
    );
  };

  const renderSectionTitle = (title: any, count = 0) => {
    if (count === 0) {
      return title;
    }

    return (
      <>
        {title} ({count})
      </>
    );
  };

  const noteInfo = NoteInfo.fromTags(tags);
  const startTime = run.getStartTime() ? Utils.formatTimestamp(run.getStartTime()) : '(unknown)';
  const duration = Utils.getDuration(run.getStartTime(), run.getEndTime());
  const status = getRunStatusDisplayName(run.getStatus());
  const lifecycleStage = run.getLifecycleStage();
  const queryParams = window.location && window.location.search ? window.location.search : '';
  const runCommand = getRunCommand({ tags, params });
  const noteContent = noteInfo && noteInfo.content;
  const plotTitle = intl.formatMessage({
    defaultMessage: 'Plot chart',
    description: 'Link to the view the plot chart for the experiment run',
  });

  return (
    <>
      {/* Metadata List */}
      {/* @ts-expect-error TS(2322): Type '{ children: (Element | null)[]; className: s... Remove this comment to see the full error message */}
      <Descriptions className="metadata-list">
        <Descriptions.Item
          label={intl.formatMessage({
            defaultMessage: 'Run ID',
            description: 'Label for displaying the ID of the experiment run',
          })}
        >
          {runUuid}
        </Descriptions.Item>
        <Descriptions.Item
          label={intl.formatMessage({
            defaultMessage: 'Date',
            description: 'Label for displaying the start time of the experiment ran',
          })}
        >
          {startTime}
        </Descriptions.Item>
        <Descriptions.Item
          label={intl.formatMessage({
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
            label={intl.formatMessage({
              defaultMessage: 'Git Commit',
              description: 'Label for displaying the tag or the commit hash of the git commit',
            })}
          >
            {Utils.renderVersion(tags, false)}
          </Descriptions.Item>
        ) : null}
        {Utils.getSourceType(tags) === 'PROJECT' ? (
          <Descriptions.Item
            label={intl.formatMessage({
              defaultMessage: 'Entry Point',
              description: 'Label for displaying entry point of the project',
            })}
          >
            {Utils.getEntryPointName(tags) || 'main'}
          </Descriptions.Item>
        ) : null}
        <Descriptions.Item
          label={intl.formatMessage({
            defaultMessage: 'User',
            description: 'Label for displaying the user who created the experiment run',
          })}
        >
          {renderUserIdLink(run, tags, experiment)}
        </Descriptions.Item>
        {duration ? (
          <Descriptions.Item
            label={intl.formatMessage({
              defaultMessage: 'Duration',
              description: 'Label for displaying the duration of the experiment run',
            })}
          >
            {duration}
          </Descriptions.Item>
        ) : null}
        <Descriptions.Item
          label={intl.formatMessage({
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
            label={intl.formatMessage({
              defaultMessage: 'Lifecycle Stage',
              description:
                // eslint-disable-next-line max-len
                'Label for displaying lifecycle stage of the experiment run to see if its active or deleted',
            })}
          >
            {renderLifecycleLink()}
          </Descriptions.Item>
        ) : null}
        {tags['mlflow.parentRunId'] !== undefined ? (
          <Descriptions.Item
            label={intl.formatMessage({
              defaultMessage: 'Parent Run',
              description: 'Label for displaying a link to the parent experiment run if any present',
            })}
          >
            <Link to={Routes.getRunPageRoute(experimentId, tags['mlflow.parentRunId'].value)}>
              {tags['mlflow.parentRunId'].value}
            </Link>
          </Descriptions.Item>
        ) : null}
        {tags['mlflow.databricks.runURL'] !== undefined ? (
          <Descriptions.Item
            label={intl.formatMessage({
              defaultMessage: 'Job Output',
              description: 'Label for displaying the output logs for the experiment run job',
            })}
          >
            {/* Reported during ESLint upgrade */}
            {/* eslint-disable-next-line react/jsx-no-target-blank */}
            <a href={Utils.setQueryParams(tags['mlflow.databricks.runURL'].value, queryParams)} target="_blank">
              <FormattedMessage defaultMessage="Logs" description="Link to the logs for the job output" />
            </a>
          </Descriptions.Item>
        ) : null}
      </Descriptions>

      {/* Page Sections */}
      <div className="RunView-info">
        {runCommand ? (
          <CollapsibleSection
            title={intl.formatMessage({
              defaultMessage: 'Run Command',
              description:
                // eslint-disable-next-line max-len
                'Label for the collapsible area to display the run command used for the experiment run',
            })}
            onChange={handleCollapseChange('runCommand')}
            data-test-id="run-command-section"
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
            <span className="RunView-editDescriptionHeader">
              <FormattedMessage
                defaultMessage="Description"
                description="Label for the notes editable content for the experiment run"
              />
              {!showNoteEditor && (
                <>
                  {' '}
                  <Button
                    componentId="codegen_mlflow_app_src_experiment-tracking_components_run-page_runviewoverview.tsx_411"
                    type="link"
                    onClick={startEditingDescription}
                    data-test-id="edit-description-button"
                  >
                    <FormattedMessage
                      defaultMessage="Edit"
                      // eslint-disable-next-line max-len
                      description="Text for the edit button next to the description section title on the run view"
                    />
                  </Button>
                </>
              )}
            </span>
          }
          forceOpen={showNoteEditor}
          defaultCollapsed={!noteContent}
          onChange={handleCollapseChange('notes')}
          data-test-id="run-notes-section"
        >
          <EditableNote
            defaultMarkdown={noteContent}
            onSubmit={handleSubmitEditNote}
            onCancel={handleCancelEditNote}
            showEditor={showNoteEditor}
          />
        </CollapsibleSection>
        <CollapsibleSection
          defaultCollapsed
          title={renderSectionTitle(
            intl.formatMessage({
              defaultMessage: 'Datasets',
              description:
                // eslint-disable-next-line max-len
                'Label for the collapsible area to display the datasets used during the experiment run',
            }),
            datasets ? datasets.length : 0,
          )}
          onChange={handleCollapseChange('parameters')}
          data-test-id="run-parameters-section"
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
                    componentId="codegen_mlflow_app_src_experiment-tracking_components_run-page_runviewoverview.tsx_460"
                    type="link"
                    css={{
                      textAlign: 'left',
                    }}
                    onClick={() => {
                      setSelectedDatasetWithRun({
                        datasetWithTags: dataset,
                        runData: {
                          experimentId: experimentId,
                          runUuid: runUuid,
                          runName: runName,
                          datasets: datasets,
                          tags: tags,
                        },
                      });
                      setIsDrawerOpen(true);
                    }}
                  >
                    <ExperimentViewDatasetWithContext datasetWithTags={dataset} displayTextAsLink />
                  </Button>
                </div>
              ))}
          </div>
        </CollapsibleSection>
        <CollapsibleSection
          defaultCollapsed
          title={renderSectionTitle(
            intl.formatMessage({
              defaultMessage: 'Parameters',
              description:
                // eslint-disable-next-line max-len
                'Label for the collapsible area to display the parameters used during the experiment run',
            }),
            getParamValues(params).length,
          )}
          onChange={handleCollapseChange('parameters')}
          data-test-id="run-parameters-section"
        >
          <HtmlTableView
            testId="params-table"
            columns={[
              {
                title: intl.formatMessage({
                  defaultMessage: 'Name',
                  description:
                    // eslint-disable-next-line max-len
                    'Column title for name column for displaying the params name for the experiment run',
                }),
                dataIndex: 'name',
              },
              {
                title: intl.formatMessage({
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
          title={renderSectionTitle(
            intl.formatMessage({
              defaultMessage: 'Metrics',
              description:
                // eslint-disable-next-line max-len
                'Label for the collapsible area to display the output metrics after the experiment run',
            }),
            getMetricValues(latestMetrics, getMetricPagePath, plotTitle).length,
          )}
          onChange={handleCollapseChange('metrics')}
          data-test-id="run-metrics-section"
        >
          <HtmlTableView
            testId="metrics-table"
            columns={[
              {
                title: intl.formatMessage({
                  defaultMessage: 'Name',
                  description:
                    // eslint-disable-next-line max-len
                    'Column title for name column for displaying the metrics name for the experiment run',
                }),
                dataIndex: 'name',
              },
              {
                title: intl.formatMessage({
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
        <div data-test-id="tags-section">
          <CollapsibleSection
            title={renderSectionTitle(
              intl.formatMessage({
                defaultMessage: 'Tags',
                description: 'Label for the collapsible area to display the tags for the experiment run',
              }),
              Utils.getVisibleTagValues(tags).length,
            )}
            defaultCollapsed
            onChange={handleCollapseChange('tags')}
            data-test-id="run-tags-section"
          >
            <EditableTagsTableView
              // @ts-expect-error TS(2322): Type '{ innerRef: RefObject<unknown>; handleAddTag... Remove this comment to see the full error message
              innerRef={formRef}
              handleAddTag={handleAddTag}
              handleDeleteTag={handleDeleteTag}
              handleSaveEdit={handleSaveEdit}
              tags={tags}
              isRequestPending={isTagsRequestPending}
            />
          </CollapsibleSection>
        </div>
        {renderArtifactSection()}
      </div>
      {notificationContextHolder}
      {selectedDatasetWithRun && (
        <ExperimentViewDatasetDrawer
          isOpen={isDrawerOpen}
          setIsOpen={setIsDrawerOpen}
          selectedDatasetWithRun={selectedDatasetWithRun}
          setSelectedDatasetWithRun={setSelectedDatasetWithRun}
        />
      )}
    </>
  );
};

const styles = {
  runCommandArea: {
    fontFamily: 'Menlo, Consolas, monospace',
    width: '100%',
  },
};

const getRunStatusDisplayName = (status: string) => {
  return status !== 'RUNNING' ? status : 'UNFINISHED';
};

const shellEscape = (str: string) => {
  if (/["\r\n\t ]/.test(str)) {
    return '"' + str.replace(/"/g, '\\"') + '"';
  }
  return str;
};

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
              <i className="fas fa-line-chart" style={{ paddingLeft: '6px' }} />
            </Link>
          ),
          value: <span title={value}>{value}</span>,
        };
      })
  );
};

// Utility functions below
const renderUserIdLink = (run: any, tags: any, experiment: any) => {
  const user = Utils.getUser(run, tags);
  return <Link to={Routes.searchRunsByUser(experiment.experiment_id, user)}>{user}</Link>;
};

const getRunCommand = ({ tags, params }: any) => {
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
};
