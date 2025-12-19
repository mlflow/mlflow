/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import { deleteRunApi, openErrorModal } from '../../actions';
import { connect } from 'react-redux';
import Utils from '../../../common/utils/Utils';
import type { IntlShape } from 'react-intl';
import { injectIntl } from 'react-intl';
import { Button, Modal } from '@databricks/design-system';
import { EXPERIMENT_PARENT_ID_TAG } from '../experiment-page/utils/experimentPage.common-utils';

interface State {
  deletingMode: null | 'selected' | 'withChildren';
}

type Props = {
  isOpen: boolean;
  onClose: (...args: any[]) => any;
  selectedRunIds: string[];
  openErrorModal: (...args: any[]) => any;
  deleteRunApi: (...args: any[]) => any;
  onSuccess?: () => void;
  intl: IntlShape;
  childRunIdsBySelectedParent: Record<string, string[]>;
};

export class DeleteRunModalImpl extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.handleDelete = this.handleDelete.bind(this);
    this.handleDeleteSelected = this.handleDeleteSelected.bind(this);
    this.handleDeleteWithChildren = this.handleDeleteWithChildren.bind(this);
    this.onRequestClose = this.onRequestClose.bind(this);
  }

  state: State = {
    deletingMode: null,
  };

  getChildRunIdsToDelete() {
    const { childRunIdsBySelectedParent, selectedRunIds } = this.props;
    const selectedRunIdsSet = new Set(selectedRunIds);
    const childRunIdsSet = new Set<string>();

    Object.values(childRunIdsBySelectedParent).forEach((childIds = []) => {
      childIds.forEach((childId) => {
        if (!selectedRunIdsSet.has(childId)) {
          childRunIdsSet.add(childId);
        }
      });
    });

    return Array.from(childRunIdsSet);
  }

  getRunIdsToDelete(includeDescendants: boolean) {
    if (!includeDescendants) {
      return this.props.selectedRunIds;
    }
    const childRunIds = this.getChildRunIdsToDelete();
    return [...this.props.selectedRunIds, ...childRunIds];
  }

  handleDelete(includeDescendants: boolean) {
    if (this.state.deletingMode) {
      return Promise.resolve();
    }
    this.setState({ deletingMode: includeDescendants ? 'withChildren' : 'selected' });
    return this.deleteRuns(includeDescendants);
  }

  handleDeleteSelected() {
    return this.handleDelete(false);
  }

  handleDeleteWithChildren() {
    return this.handleDelete(true);
  }

  deleteRuns(includeDescendants: boolean) {
    const deletePromises: any = [];
    const runIdsToDelete = this.getRunIdsToDelete(includeDescendants);
    runIdsToDelete.forEach((runId) => {
      deletePromises.push(this.props.deleteRunApi(runId));
    });
    return Promise.all(deletePromises)
      .catch(() => {
        const errorModalContent = `${this.props.intl.formatMessage({
          defaultMessage: 'While deleting an experiment run, an error occurred.',
          description: 'Experiment tracking > delete run modal > error message',
        })}`;
        this.props.openErrorModal(errorModalContent);
      })
      .then(() => {
        this.props.onSuccess?.();
      })
      .finally(() => {
        this.setState({ deletingMode: null });
        this.props.onClose();
      });
  }

  onRequestClose() {
    if (!this.state.deletingMode) {
      this.props.onClose();
    }
  }

  render() {
    const number = this.props.selectedRunIds.length;
    const childRunIds = this.getChildRunIdsToDelete();
    const childRunCount = childRunIds.length;
    const hasChildRuns = childRunCount > 0;
    const totalRunsWithChildren = number + childRunCount;
    const { deletingMode } = this.state;
    const isDeletingSelected = deletingMode === 'selected';
    const isDeletingWithChildren = deletingMode === 'withChildren';

    const deleteSelectedButton = (
      <Button
        componentId="delete-selected"
        key="delete-selected"
        onClick={this.handleDeleteSelected}
        disabled={isDeletingWithChildren}
        loading={isDeletingSelected}
        type="primary"
      >
        Delete selected
      </Button>
    );

    const deleteSelectedAndChildrenButton = hasChildRuns ? (
      <Button
        componentId="delete-selected-children"
        key="delete-selected-children"
        type="primary"
        onClick={this.handleDeleteWithChildren}
        disabled={isDeletingSelected}
        loading={isDeletingWithChildren}
      >
        Delete selected and children
      </Button>
    ) : null;

    const footerButtons = [
      <Button componentId="cancel" key="cancel" onClick={this.onRequestClose} disabled={!!deletingMode}>
        Cancel
      </Button>,
      deleteSelectedButton,
      ...(deleteSelectedAndChildrenButton ? [deleteSelectedAndChildrenButton] : []),
    ];

    return (
      <Modal
        componentId="delete-run-modal"
        data-testid="delete-run-modal"
        title={`Delete Experiment ${Utils.pluralize('Run', number)}`}
        visible={this.props.isOpen}
        onCancel={this.onRequestClose}
        footer={footerButtons}
      >
        <div className="modal-explanatory-text">
          <p>
            <b>
              Selected {number} experiment {Utils.pluralize('run', number)}.
            </b>
          </p>
          {hasChildRuns ? (
            <p>
              The selected run has {childRunCount} child {Utils.pluralize('run', childRunCount)}. Delete this run alone
              or all {totalRunsWithChildren}?
            </p>
          ) : null}
          {/* @ts-expect-error TS(4111): Property 'MLFLOW_SHOW_GDPR_PURGING_MESSAGES' comes from a... Remove this comment to see the full error message */}
          {process.env.MLFLOW_SHOW_GDPR_PURGING_MESSAGES === 'true' ? (
            <p>
              Deleted runs are restorable for 30 days, after which they are purged along with associated metrics,
              params, tags, and artifacts.
            </p>
          ) : (
            ''
          )}
        </div>
      </Modal>
    );
  }
}

const mapStateToProps = (state: any, ownProps: { selectedRunIds: string[] }) => {
  const tagsByRunUuid = state.entities?.tagsByRunUuid || {};
  const parentToChildrenMap: Record<string, string[]> = {};

  Object.entries(tagsByRunUuid).forEach(([runId, tags]) => {
    const parentTag = (tags as any)?.[EXPERIMENT_PARENT_ID_TAG];
    const parentRunId = parentTag?.value;
    if (parentRunId) {
      if (!parentToChildrenMap[parentRunId]) {
        parentToChildrenMap[parentRunId] = [];
      }
      parentToChildrenMap[parentRunId].push(runId);
    }
  });

  const resolveDescendants = (runId: string) => {
    const result: string[] = [];
    const stack = [...(parentToChildrenMap[runId] || [])];
    const visited = new Set<string>();

    while (stack.length) {
      const current = stack.pop();
      if (current && !visited.has(current)) {
        visited.add(current);
        result.push(current);
        const children = parentToChildrenMap[current];
        if (children) {
          stack.push(...children);
        }
      }
    }

    return result;
  };

  const childRunIdsBySelectedParent: Record<string, string[]> = {};
  ownProps.selectedRunIds.forEach((runId) => {
    const descendants = resolveDescendants(runId);
    if (descendants.length) {
      childRunIdsBySelectedParent[runId] = descendants;
    }
  });

  return { childRunIdsBySelectedParent };
};

const mapDispatchToProps = {
  deleteRunApi,
  openErrorModal,
};

export default connect(mapStateToProps, mapDispatchToProps)(injectIntl(DeleteRunModalImpl));
