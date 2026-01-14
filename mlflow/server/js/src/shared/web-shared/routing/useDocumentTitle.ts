import { useEffect } from 'react';

interface UseDocumentTitleProps {
  /** title to be displayed as browser title */
  title?: string;
  /** skip updating the title */
  skip?: boolean;
}

export const useDocumentTitle = ({ skip, title }: UseDocumentTitleProps) => {
  const productName = 'MLflow';

  useEffect(() => {
    if (skip) {
      return;
    }
    if (document) {
      document.title = title ? `${title} - ${productName}` : productName;
    }

    return () => {
      if (document) {
        document.title = productName;
      }
    };
  }, [productName, skip, title]);
};
