import { render, screen, fireEvent } from '@testing-library/react';
import { DesignSystemProvider } from '@databricks/design-system';
import { HighlightedTextArea } from './HighlightedTextArea';
import { describe, it, expect, jest, beforeEach } from '@jest/globals';

const renderWithTheme = (ui: React.ReactElement) => {
  return render(<DesignSystemProvider>{ui}</DesignSystemProvider>);
};

describe('HighlightedTextArea', () => {
  const defaultProps = {
    value: '',
    onChange: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should render a textarea with provided value', () => {
    renderWithTheme(<HighlightedTextArea {...defaultProps} value="Hello world" />);

    expect(screen.getByRole('textbox')).toHaveValue('Hello world');
  });

  it('should highlight template variables with MARK tag', () => {
    renderWithTheme(
      <HighlightedTextArea
        {...defaultProps}
        value="Compare {{ inputs }} with {{ outputs }} and check {{ expectations }}"
      />,
    );

    expect(screen.getByText('{{ inputs }}').tagName).toBe('MARK');
    expect(screen.getByText('{{ outputs }}').tagName).toBe('MARK');
    expect(screen.getByText('{{ expectations }}').tagName).toBe('MARK');
  });

  it('should not highlight text that is not a template variable', () => {
    renderWithTheme(<HighlightedTextArea {...defaultProps} value="Regular text without variables" />);

    expect(screen.queryAllByRole('mark')).toHaveLength(0);
  });

  it('should call onChange when user types', () => {
    const onChange = jest.fn();
    renderWithTheme(<HighlightedTextArea {...defaultProps} onChange={onChange} />);

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'New value' } });

    expect(onChange).toHaveBeenCalledWith('New value');
  });
});
