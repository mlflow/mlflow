import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { FormattedMessage } from 'react-intl';
// eslint-disable-next-line no-unused-vars
import { Descriptions, Icon, Menu, Popover } from 'antd';
import './ExperimentView.css';
import { CollapsibleSection } from '../../common/components/CollapsibleSection';
import { EditableNote } from '../../common/components/EditableNote';
import { IconButton } from '../../common/components/IconButton';
import { Experiment } from '../sdk/MlflowMessages';

export function ExperimentNoteSection(props) {
  const {
    handleCancelEditNote,
    handleSubmitEditNote,
    startEditingDescription,
    noteInfo,
    showNotesEditor,
  } = props;

  const editIcon = <IconButton icon={<Icon type='form' />} onClick={startEditingDescription} />;

  const content = noteInfo && noteInfo.content;

  return (
    <CollapsibleSection
      title={
        <span>
          <FormattedMessage
            defaultMessage='Notes'
            description='Header for displaying notes for the experiment table'
          />
          {showNotesEditor ? null : editIcon}
        </span>
      }
      forceOpen={showNotesEditor}
      defaultCollapsed={!content}
      data-test-id='experiment-notes-section'
    >
      <EditableNote
        defaultMarkdown={content}
        onSubmit={handleSubmitEditNote}
        onCancel={handleCancelEditNote}
        showEditor={showNotesEditor}
      />
    </CollapsibleSection>
  );
}

ExperimentNoteSection.propTypes = {
  startEditingDescription: PropTypes.func.isRequired,
  handleSubmitEditNote: PropTypes.func.isRequired,
  handleCancelEditNote: PropTypes.func.isRequired,
  showNotesEditor: PropTypes.bool,
  noteInfo: PropTypes.object,
};

export class ArtifactLocation extends Component {
  static propTypes = {
    experiment: PropTypes.instanceOf(Experiment).isRequired,
    intl: PropTypes.shape({ formatMessage: PropTypes.func.isRequired }).isRequired,
    permissionsLearnMoreLinkUrl: PropTypes.string,
  };
  render() {
    const { artifact_location } = this.props.experiment;
    const label = this.props.intl.formatMessage({
      defaultMessage: 'Artifact Location',
      description: 'Label for displaying the experiment artifact location',
    });
    return <Descriptions.Item label={label}>{artifact_location}</Descriptions.Item>;
  }
}
