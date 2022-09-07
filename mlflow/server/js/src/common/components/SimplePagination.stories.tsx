import { useState, useEffect } from 'react';
import { SimplePagination } from './SimplePagination';

export default {
  title: 'Common/SimplePagination',
  component: SimplePagination,
  argTypes: {},
};

const Wrapper = () => {
  const [currentPage, setCurrentPage] = useState(1);
  const [perPageSelection, setPerPageSelection] = useState(10);

  // There's a weird thing where ant sets this to be 0 when perPage changes.
  useEffect(() => setCurrentPage(1), [perPageSelection]);
  return (
    <SimplePagination
      currentPage={currentPage}
      isLastPage={currentPage === 10}
      onClickNext={() => setCurrentPage((prev) => prev + 1)}
      onClickPrev={() => setCurrentPage((prev) => prev - 1)}
      maxResultOptions={['10', '25', '50', '100']}
      handleSetMaxResult={({ key }: { key: number }) => setPerPageSelection(key)}
      getSelectedPerPageSelection={() => perPageSelection}
    />
  );
};

export const Pagination = () => <Wrapper />;
