import { describe, it, expect, jest } from '@jest/globals';
import { screen, renderWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import { ExperimentListTableTagsCell } from './ExperimentListTableTagsCell';
import type { ExperimentEntity } from '../types';

const createMockExperiment = (tags: Array<{ key: string; value: string }>): ExperimentEntity =>
  ({
    experimentId: '1',
    name: 'Test Experiment',
    tags,
  }) as ExperimentEntity;

const createMockCellContext = (tags: Array<{ key: string; value: string }>, onEditTags = jest.fn()) => ({
  row: { original: createMockExperiment(tags) },
  table: { options: { meta: { onEditTags } } },
});

const TagsCell = ExperimentListTableTagsCell as React.FC<any>;

const renderTagsCell = (tags: Array<{ key: string; value: string }>, onEditTags = jest.fn()) =>
  renderWithIntl(
    <DesignSystemProvider>
      <TagsCell {...createMockCellContext(tags, onEditTags)} />
    </DesignSystemProvider>,
  );

describe('ExperimentListTableTagsCell', () => {
  it('should render "Add tags" button when there are no tags', () => {
    renderTagsCell([]);

    expect(screen.getByText('Add tags')).toBeInTheDocument();
  });

  it('should render a single tag without overflow indicator', () => {
    renderTagsCell([{ key: 'env', value: 'production' }]);

    expect(screen.getByText('env')).toBeInTheDocument();
    expect(screen.queryByText(/\+\d+/)).not.toBeInTheDocument();
  });

  it('should show overflow indicator when tags exceed visible count', () => {
    renderTagsCell([
      { key: 'tag1', value: 'value1' },
      { key: 'tag2', value: 'value2' },
      { key: 'tag3', value: 'value3' },
    ]);

    expect(screen.getByText('tag1')).toBeInTheDocument();
    expect(screen.getByText('+2')).toBeInTheDocument();
  });
});
