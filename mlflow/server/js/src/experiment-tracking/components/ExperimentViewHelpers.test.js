import React from 'react';
import { ExperimentNoteSection } from './ExperimentViewHelpers';
import { shallow } from 'enzyme';

const getDefaultExperimentNoteProps = () => {
  return {
    showNotesEditor: true,
    noteInfo: {
      content: 'mock-content',
    },
    handleCancelEditNote: jest.fn(),
    handleSubmitEditNote: jest.fn(),
    startEditingDescription: jest.fn(),
  };
};

const getExperimentNotebookPanelMock = (componentProps = {}) => {
  const mergedProps = {
    ...getDefaultExperimentNoteProps(),
    ...componentProps,
  };
  return shallow(<ExperimentNoteSection {...mergedProps} />);
};

describe('Experiment Notes', () => {
  test('Notes panel appears', () => {
    const wrapper = getExperimentNotebookPanelMock();
    expect(wrapper.exists('[data-test-id="experiment-notes-section"]')).toBe(true);
  });
});
