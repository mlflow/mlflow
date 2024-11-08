import React from 'react';
import { SimplePagination } from './SimplePagination';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

const minimalProps = {
  currentPage: 3,
  isLastPage: false,
  onClickNext: jest.fn(),
  onClickPrev: jest.fn(),
  getSelectedPerPageSelection: () => 25,
};

const paginationSectionDataTestId = 'pagination-section';
const previousPageButtonTitle = 'Previous Page';
const nextPageButtonTitle = 'Next Page';

describe('SimplePagination', () => {
  test('prev and next buttons are rendered and not disabled when the current page is in the middle', () => {
    renderWithIntl(<SimplePagination {...minimalProps} />);
    expect(screen.getByTestId(paginationSectionDataTestId)).toBeInTheDocument();

    expect(screen.getByTitle(previousPageButtonTitle)).toHaveAttribute('aria-disabled', 'false');
    expect(screen.getByTitle(nextPageButtonTitle)).toHaveAttribute('aria-disabled', 'false');
  });

  test('prev button is disabled when the current page is first page', () => {
    renderWithIntl(<SimplePagination {...minimalProps} currentPage={1} />);
    expect(screen.getByTestId(paginationSectionDataTestId)).toBeInTheDocument();

    expect(screen.getByTitle(previousPageButtonTitle)).toHaveAttribute('aria-disabled', 'true');
    expect(screen.getByTitle(nextPageButtonTitle)).toHaveAttribute('aria-disabled', 'false');
  });

  test('next button is disabled when the current page is the last page', () => {
    renderWithIntl(<SimplePagination {...minimalProps} currentPage={2} isLastPage />);
    expect(screen.getByTestId(paginationSectionDataTestId)).toBeInTheDocument();

    expect(screen.getByTitle(previousPageButtonTitle)).toHaveAttribute('aria-disabled', 'false');
    expect(screen.getByTitle(nextPageButtonTitle)).toHaveAttribute('aria-disabled', 'true');
  });

  test('both buttons are disabled when there is only one page', () => {
    renderWithIntl(<SimplePagination {...minimalProps} currentPage={1} isLastPage />);
    expect(screen.getByTestId(paginationSectionDataTestId)).toBeInTheDocument();

    expect(screen.getByTitle(previousPageButtonTitle)).toHaveAttribute('aria-disabled', 'true');
    expect(screen.getByTitle(nextPageButtonTitle)).toHaveAttribute('aria-disabled', 'true');
  });
});
