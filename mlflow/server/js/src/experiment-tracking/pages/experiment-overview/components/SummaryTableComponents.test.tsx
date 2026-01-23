import { describe, it, expect, jest } from '@jest/globals';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderHook, act } from '@testing-library/react';
import { renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import { DesignSystemProvider } from '@databricks/design-system';
import { useSortState, SortableHeader, LinkableNameCell, useSummaryTableStyles } from './SummaryTableComponents';

// Wrapper for hooks that need DesignSystemProvider
const wrapper = ({ children }: { children: React.ReactNode }) => (
  <DesignSystemProvider>{children}</DesignSystemProvider>
);

describe('SummaryTableComponents', () => {
  describe('useSortState', () => {
    it('should initialize with default column and direction', () => {
      const { result } = renderHook(() => useSortState<'a' | 'b'>('a'));

      expect(result.current.sortColumn).toBe('a');
      expect(result.current.sortDirection).toBe('desc');
    });

    it('should allow custom default direction', () => {
      const { result } = renderHook(() => useSortState<'a' | 'b'>('a', 'asc'));

      expect(result.current.sortColumn).toBe('a');
      expect(result.current.sortDirection).toBe('asc');
    });

    it('should toggle direction when sorting same column', () => {
      const { result } = renderHook(() => useSortState<'a' | 'b'>('a'));

      expect(result.current.sortDirection).toBe('desc');

      act(() => {
        result.current.handleSort('a');
      });

      expect(result.current.sortColumn).toBe('a');
      expect(result.current.sortDirection).toBe('asc');

      act(() => {
        result.current.handleSort('a');
      });

      expect(result.current.sortDirection).toBe('desc');
    });

    it('should switch to new column with desc direction', () => {
      const { result } = renderHook(() => useSortState<'a' | 'b'>('a', 'asc'));

      expect(result.current.sortColumn).toBe('a');
      expect(result.current.sortDirection).toBe('asc');

      act(() => {
        result.current.handleSort('b');
      });

      expect(result.current.sortColumn).toBe('b');
      expect(result.current.sortDirection).toBe('desc');
    });
  });

  describe('SortableHeader', () => {
    const defaultProps = {
      column: 'test' as const,
      sortColumn: 'test' as const,
      sortDirection: 'desc' as const,
      onSort: jest.fn(),
    };

    it('should render children', () => {
      renderWithIntl(
        <DesignSystemProvider>
          <SortableHeader {...defaultProps}>Test Header</SortableHeader>
        </DesignSystemProvider>,
      );

      expect(screen.getByText('Test Header')).toBeInTheDocument();
    });

    it('should call onSort when clicked', async () => {
      const onSort = jest.fn();
      renderWithIntl(
        <DesignSystemProvider>
          <SortableHeader {...defaultProps} onSort={onSort}>
            Test Header
          </SortableHeader>
        </DesignSystemProvider>,
      );

      await userEvent.click(screen.getByRole('button'));

      expect(onSort).toHaveBeenCalledWith('test');
    });

    it('should call onSort when Enter key is pressed', async () => {
      const onSort = jest.fn();
      renderWithIntl(
        <DesignSystemProvider>
          <SortableHeader {...defaultProps} onSort={onSort}>
            Test Header
          </SortableHeader>
        </DesignSystemProvider>,
      );

      const button = screen.getByRole('button');
      button.focus();
      await userEvent.keyboard('{Enter}');

      expect(onSort).toHaveBeenCalledWith('test');
    });

    it('should show sort icon when column is active', () => {
      renderWithIntl(
        <DesignSystemProvider>
          <SortableHeader {...defaultProps}>Test Header</SortableHeader>
        </DesignSystemProvider>,
      );

      // Should have sort icon
      expect(screen.getByRole('img', { hidden: true })).toBeInTheDocument();
    });

    it('should not show sort icon when column is not active', () => {
      renderWithIntl(
        <DesignSystemProvider>
          <SortableHeader {...defaultProps} sortColumn={'other' as any}>
            Test Header
          </SortableHeader>
        </DesignSystemProvider>,
      );

      // Should not have sort icon
      expect(screen.queryByRole('img', { hidden: true })).not.toBeInTheDocument();
    });
  });

  describe('LinkableNameCell', () => {
    it('should render name with color indicator as clickable', () => {
      renderWithIntl(
        <DesignSystemProvider>
          <LinkableNameCell name="test_item" color="#00ff00" scrollToElementId="test-element" />
        </DesignSystemProvider>,
      );

      expect(screen.getByText('test_item')).toBeInTheDocument();
      expect(screen.getByRole('button')).toBeInTheDocument();
    });

    it('should scroll to element when clicked', async () => {
      const mockElement = document.createElement('div');
      mockElement.id = 'scroll-target';
      mockElement.scrollIntoView = jest.fn();
      document.body.appendChild(mockElement);

      renderWithIntl(
        <DesignSystemProvider>
          <LinkableNameCell name="clickable_item" color="#0000ff" scrollToElementId="scroll-target" />
        </DesignSystemProvider>,
      );

      await userEvent.click(screen.getByRole('button'));

      expect(mockElement.scrollIntoView).toHaveBeenCalledWith({ behavior: 'smooth', block: 'start' });

      document.body.removeChild(mockElement);
    });

    it('should scroll to element on Enter key', async () => {
      const mockElement = document.createElement('div');
      mockElement.id = 'keyboard-target';
      mockElement.scrollIntoView = jest.fn();
      document.body.appendChild(mockElement);

      renderWithIntl(
        <DesignSystemProvider>
          <LinkableNameCell name="keyboard_item" color="#ff00ff" scrollToElementId="keyboard-target" />
        </DesignSystemProvider>,
      );

      const button = screen.getByRole('button');
      button.focus();
      await userEvent.keyboard('{Enter}');

      expect(mockElement.scrollIntoView).toHaveBeenCalledWith({ behavior: 'smooth', block: 'start' });

      document.body.removeChild(mockElement);
    });
  });

  describe('useSummaryTableStyles', () => {
    it('should return style objects', () => {
      const { result } = renderHook(() => useSummaryTableStyles('1fr 1fr'), { wrapper });

      expect(result.current.rowStyle).toBeDefined();
      expect(result.current.headerRowStyle).toBeDefined();
      expect(result.current.bodyRowStyle).toBeDefined();
      expect(result.current.cellStyle).toBeDefined();
    });

    it('should include grid columns in row style', () => {
      const { result } = renderHook(() => useSummaryTableStyles('minmax(80px, 2fr) 1fr 1fr'), { wrapper });

      expect(result.current.rowStyle.gridTemplateColumns).toBe('minmax(80px, 2fr) 1fr 1fr');
    });
  });
});
