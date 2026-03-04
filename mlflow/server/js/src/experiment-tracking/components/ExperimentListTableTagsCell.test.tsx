import { describe, expect, it, jest, beforeEach } from '@jest/globals';
import React from 'react';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';
import { ExperimentListTableTagsCell } from './ExperimentListTableTagsCell';
import type { ExperimentEntity } from '../types';

// Mock KeyValueTag component
jest.mock('../../common/components/KeyValueTag', () => ({
  KeyValueTag: ({ tag }: { tag: { key: string; value?: string } }) => (
    <span data-testid="key-value-tag">{`${tag.key}: ${tag.value || ''}`}</span>
  ),
}));

// Mock isUserFacingTag to pass through all tags
jest.mock('../../common/utils/TagUtils', () => ({
  isUserFacingTag: () => true,
}));

const createMockTableMeta = (onEditTags = jest.fn()) => ({
  onEditTags,
});

const createMockRow = (tags: Array<{ key: string; value?: string }> = []) => ({
  original: {
    experimentId: '1',
    name: 'Test Experiment',
    tags,
  } as ExperimentEntity,
});

const createMockCellContext = (tags: Array<{ key: string; value?: string }> = [], onEditTags = jest.fn()) => ({
  row: createMockRow(tags),
  table: {
    options: {
      meta: createMockTableMeta(onEditTags),
    },
  },
});

describe('ExperimentListTableTagsCell', () => {
  let onEditTags: ReturnType<typeof jest.fn>;

  beforeEach(() => {
    onEditTags = jest.fn();
  });

  it('renders tags correctly when there are 3 or fewer tags', () => {
    const tags = [
      { key: 'tag1', value: 'value1' },
      { key: 'tag2', value: 'value2' },
      { key: 'tag3', value: 'value3' },
    ];

    renderWithDesignSystem(<ExperimentListTableTagsCell {...createMockCellContext(tags, onEditTags)} />);

    const renderedTags = screen.getAllByTestId('key-value-tag');
    expect(renderedTags).toHaveLength(3);
    expect(screen.queryByText(/\+.*more/)).not.toBeInTheDocument();
  });

  it('shows "+N more" button when there are more than 3 tags', () => {
    const tags = [
      { key: 'tag1', value: 'value1' },
      { key: 'tag2', value: 'value2' },
      { key: 'tag3', value: 'value3' },
      { key: 'tag4', value: 'value4' },
      { key: 'tag5', value: 'value5' },
    ];

    renderWithDesignSystem(<ExperimentListTableTagsCell {...createMockCellContext(tags, onEditTags)} />);

    // Should only show 3 tags initially
    const renderedTags = screen.getAllByTestId('key-value-tag');
    expect(renderedTags).toHaveLength(3);

    // Should show "+2 more" button
    expect(screen.getByText('+2 more')).toBeInTheDocument();
  });

  it('expands to show all tags when "+N more" is clicked', async () => {
    const tags = [
      { key: 'tag1', value: 'value1' },
      { key: 'tag2', value: 'value2' },
      { key: 'tag3', value: 'value3' },
      { key: 'tag4', value: 'value4' },
      { key: 'tag5', value: 'value5' },
    ];

    renderWithDesignSystem(<ExperimentListTableTagsCell {...createMockCellContext(tags, onEditTags)} />);

    // Click "+2 more" button
    await userEvent.click(screen.getByText('+2 more'));

    // Should now show all 5 tags
    const renderedTags = screen.getAllByTestId('key-value-tag');
    expect(renderedTags).toHaveLength(5);

    // Should show "Show less" button
    expect(screen.getByText('Show less')).toBeInTheDocument();
  });

  it('collapses tags when "Show less" is clicked', async () => {
    const tags = [
      { key: 'tag1', value: 'value1' },
      { key: 'tag2', value: 'value2' },
      { key: 'tag3', value: 'value3' },
      { key: 'tag4', value: 'value4' },
    ];

    renderWithDesignSystem(<ExperimentListTableTagsCell {...createMockCellContext(tags, onEditTags)} />);

    // Expand
    await userEvent.click(screen.getByText('+1 more'));
    expect(screen.getAllByTestId('key-value-tag')).toHaveLength(4);

    // Collapse
    await userEvent.click(screen.getByText('Show less'));
    expect(screen.getAllByTestId('key-value-tag')).toHaveLength(3);
    expect(screen.getByText('+1 more')).toBeInTheDocument();
  });

  it('shows "Add tags" button when there are no tags', () => {
    renderWithDesignSystem(<ExperimentListTableTagsCell {...createMockCellContext([], onEditTags)} />);

    expect(screen.getByText('Add tags')).toBeInTheDocument();
    expect(screen.queryByTestId('key-value-tag')).not.toBeInTheDocument();
  });

  it('calls onEditTags when edit button is clicked', async () => {
    const tags = [{ key: 'tag1', value: 'value1' }];
    const mockOnEditTags = jest.fn();

    renderWithDesignSystem(<ExperimentListTableTagsCell {...createMockCellContext(tags, mockOnEditTags)} />);

    const editButton = screen.getByRole('button', { name: 'Edit tags' });
    await userEvent.click(editButton);

    expect(mockOnEditTags).toHaveBeenCalledWith(createMockRow(tags).original);
  });
});
