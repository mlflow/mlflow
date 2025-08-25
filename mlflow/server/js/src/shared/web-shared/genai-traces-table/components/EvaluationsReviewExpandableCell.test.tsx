import { render, screen } from '@testing-library/react';

import { EvaluationsReviewExpandedJSONValueCell } from './EvaluationsReviewExpandableCell';

describe('EvaluationsReviewExpandedJSONValueCell', () => {
  const renderTestComponent = (value: string | Record<string, unknown>) =>
    render(<EvaluationsReviewExpandedJSONValueCell value={value} />);

  // Helper function to normalize JSON string for comparison
  const normalizeJSON = (json: string) => JSON.stringify(JSON.parse(json));

  test('renders string value as is when not valid JSON', () => {
    const value = 'This is a plain text string';
    renderTestComponent(value);
    expect(screen.getByText(value)).toBeInTheDocument();
  });

  test('renders formatted JSON when given a valid JSON string', () => {
    const jsonString = '{"name":"test","value":123}';
    renderTestComponent(jsonString);
    const element = screen.getByText((content) => {
      try {
        // Try to parse the content as JSON and compare normalized versions
        return normalizeJSON(content) === normalizeJSON(jsonString);
      } catch {
        return false;
      }
    });
    expect(element).toBeInTheDocument();
  });

  test('renders formatted JSON when given an object', () => {
    const objectValue = { name: 'test', value: 123, nested: { key: 'value' } };
    renderTestComponent(objectValue);
    const element = screen.getByText((content) => {
      try {
        // Try to parse the content as JSON and compare with the original object
        return normalizeJSON(content) === JSON.stringify(objectValue);
      } catch {
        return false;
      }
    });
    expect(element).toBeInTheDocument();
  });

  test('renders string value when given invalid JSON string', () => {
    const invalidJson = '{invalid json}';
    renderTestComponent(invalidJson);
    expect(screen.getByText(invalidJson)).toBeInTheDocument();
  });

  test('handles complex nested objects', () => {
    const complexObject = {
      name: 'test',
      array: [1, 2, 3],
      nested: {
        key: 'value',
        another: {
          deep: true,
        },
      },
    };
    renderTestComponent(complexObject);
    const element = screen.getByText((content) => {
      try {
        // Try to parse the content as JSON and compare with the original object
        return normalizeJSON(content) === JSON.stringify(complexObject);
      } catch {
        return false;
      }
    });
    expect(element).toBeInTheDocument();
  });
});
