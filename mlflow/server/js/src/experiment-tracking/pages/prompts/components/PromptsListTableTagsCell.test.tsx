import { describe, it, expect, jest } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { screen, renderWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import type { RegisteredPrompt } from '../types';
import { PromptsListTableTagsCell } from './PromptsListTableTagsCell';

const createMockPrompt = (tags: Array<{ key: string; value: string }>): RegisteredPrompt =>
  ({
    name: 'Test Prompt',
    tags,
  }) as RegisteredPrompt;

const createMockCellContext = (tags: Array<{ key: string; value: string }>, onEditTags = jest.fn()) => ({
  row: { original: createMockPrompt(tags) },
  table: { options: { meta: { onEditTags } } },
});

const TagsCell = PromptsListTableTagsCell as React.FC<any>;

const renderTagsCell = (tags: Array<{ key: string; value: string }>, onEditTags = jest.fn()) =>
  renderWithIntl(
    <DesignSystemProvider>
      <TagsCell {...createMockCellContext(tags, onEditTags)} />
    </DesignSystemProvider>,
  );

describe('PromptsListTableTagsCell', () => {
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
