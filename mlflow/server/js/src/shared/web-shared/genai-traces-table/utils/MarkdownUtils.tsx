import React from 'react';

import sanitize from '../../html-content/sanitize';

// Default to DOMPurify sanitization so that components rendered without a
// MarkdownConverterProvider still produce safe HTML (defense-in-depth for XSS).
const MarkdownConverterProviderContext = React.createContext({
  makeHTML: (markdown?: string) => (markdown ? sanitize(markdown) : markdown),
});

export const MarkdownConverterProvider = ({
  children,
  makeHtml,
}: {
  children: React.ReactNode;
  makeHtml: (markdown?: string) => string;
}) => {
  return (
    <MarkdownConverterProviderContext.Provider value={{ makeHTML: makeHtml }}>
      {children}
    </MarkdownConverterProviderContext.Provider>
  );
};

export const useMarkdownConverter = () => React.useContext(MarkdownConverterProviderContext);
