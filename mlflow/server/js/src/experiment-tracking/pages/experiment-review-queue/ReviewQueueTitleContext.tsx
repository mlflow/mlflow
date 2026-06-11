import { createContext, useContext, useMemo, useState } from 'react';

type ReviewQueueTitleContextValue = {
  title: string | null;
  setTitle: (title: string | null) => void;
};

const ReviewQueueTitleContext = createContext<ReviewQueueTitleContextValue>({
  title: null,
  setTitle: () => {},
});

export const ReviewQueueTitleProvider = ({ children }: { children: React.ReactNode }) => {
  const [title, setTitle] = useState<string | null>(null);
  const value = useMemo(() => ({ title, setTitle }), [title]);
  return <ReviewQueueTitleContext.Provider value={value}>{children}</ReviewQueueTitleContext.Provider>;
};

export const useReviewQueueTitle = () => useContext(ReviewQueueTitleContext);
