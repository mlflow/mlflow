import { describe, test, expect } from '@jest/globals';
import { IntlProvider } from 'react-intl';
import { render, screen } from '../../../../../common/utils/TestUtils.react18';
import { ExperimentViewDatasetSchemaTable } from './ExperimentViewDatasetSchemaTable';
import { DesignSystemProvider } from '@databricks/design-system';

describe('ExperimentViewDatasetSchemaTable', () => {
  const renderTestComponent = ({ schema, filter }: { schema: any[]; filter: string }) => {
    return render(<ExperimentViewDatasetSchemaTable schema={schema} filter={filter} />, {
      wrapper: ({ children }) => (
        <IntlProvider locale="en">
          <DesignSystemProvider>{children}</DesignSystemProvider>
        </IntlProvider>
      ),
    });
  };

  test('it renders regular column names', () => {
    const schema = [
      { name: 'feature_a', type: 'long' },
      { name: 'feature_b', type: 'string' },
      { name: 'target', type: 'double' },
    ];
    renderTestComponent({ schema, filter: '' });

    expect(screen.getByText('feature_a')).toBeInTheDocument();
    expect(screen.getByText('feature_b')).toBeInTheDocument();
    expect(screen.getByText('target')).toBeInTheDocument();
  });

  test('it renders MultiIndex column names as dot-separated strings', () => {
    const schema = [
      { name: ['foo', 'a'], type: 'long' },
      { name: ['foo', 'b'], type: 'long' },
      { name: ['bar', 'c'], type: 'long' },
      { name: ['bar', 'd'], type: 'long' },
    ];
    renderTestComponent({ schema, filter: '' });

    expect(screen.getByText('foo.a')).toBeInTheDocument();
    expect(screen.getByText('foo.b')).toBeInTheDocument();
    expect(screen.getByText('bar.c')).toBeInTheDocument();
    expect(screen.getByText('bar.d')).toBeInTheDocument();
  });

  test('it filters regular columns by name', () => {
    const schema = [
      { name: 'feature_a', type: 'long' },
      { name: 'feature_b', type: 'string' },
      { name: 'target', type: 'double' },
    ];
    renderTestComponent({ schema, filter: 'feature' });

    expect(screen.getByText('feature_a')).toBeInTheDocument();
    expect(screen.getByText('feature_b')).toBeInTheDocument();
    expect(screen.queryByText('target')).not.toBeInTheDocument();
  });

  test('it filters MultiIndex columns by first level', () => {
    const schema = [
      { name: ['foo', 'a'], type: 'long' },
      { name: ['foo', 'b'], type: 'long' },
      { name: ['bar', 'c'], type: 'long' },
      { name: ['bar', 'd'], type: 'long' },
    ];
    renderTestComponent({ schema, filter: 'foo' });

    expect(screen.getByText('foo.a')).toBeInTheDocument();
    expect(screen.getByText('foo.b')).toBeInTheDocument();
    expect(screen.queryByText('bar.c')).not.toBeInTheDocument();
    expect(screen.queryByText('bar.d')).not.toBeInTheDocument();
  });

  test('it filters MultiIndex columns by second level', () => {
    const schema = [
      { name: ['foo', 'a'], type: 'long' },
      { name: ['foo', 'b'], type: 'long' },
      { name: ['bar', 'c'], type: 'long' },
      { name: ['bar', 'd'], type: 'long' },
    ];
    renderTestComponent({ schema, filter: '.d' });

    expect(screen.queryByText('foo.a')).not.toBeInTheDocument();
    expect(screen.queryByText('foo.b')).not.toBeInTheDocument();
    expect(screen.queryByText('bar.c')).not.toBeInTheDocument();
    expect(screen.getByText('bar.d')).toBeInTheDocument();
  });

  test('it filters columns by type', () => {
    const schema = [
      { name: 'feature_a', type: 'long' },
      { name: 'feature_b', type: 'string' },
      { name: 'target', type: 'double' },
    ];
    renderTestComponent({ schema, filter: 'string' });

    expect(screen.queryByText('feature_a')).not.toBeInTheDocument();
    expect(screen.getByText('feature_b')).toBeInTheDocument();
    expect(screen.queryByText('target')).not.toBeInTheDocument();
  });

  test('it shows "No results" message when no columns match filter', () => {
    const schema = [
      { name: 'feature_a', type: 'long' },
      { name: 'feature_b', type: 'string' },
    ];
    renderTestComponent({ schema, filter: 'nonexistent' });

    expect(screen.getByText('No results match this search.')).toBeInTheDocument();
    expect(screen.queryByText('feature_a')).not.toBeInTheDocument();
    expect(screen.queryByText('feature_b')).not.toBeInTheDocument();
  });

  test('it handles case-insensitive filtering', () => {
    const schema = [
      { name: 'Feature_A', type: 'long' },
      { name: 'feature_b', type: 'string' },
    ];
    renderTestComponent({ schema, filter: 'FEATURE' });

    expect(screen.getByText('Feature_A')).toBeInTheDocument();
    expect(screen.getByText('feature_b')).toBeInTheDocument();
  });
});
