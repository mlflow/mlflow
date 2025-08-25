import React from 'react';

const MarkdownConverterProviderContext = React.createContext({
  makeHTML: (markdown?: string) => markdown,
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
