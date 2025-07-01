import { MouseEvent, ReactNode, useCallback } from 'react';

interface NotebookDownloadButtonProps {
  children: ReactNode;
  href: string;
}

export function NotebookDownloadButton({ children, href }: NotebookDownloadButtonProps) {
  const handleClick = useCallback(
    async (e: MouseEvent) => {
      e.preventDefault();

      if ((window as any).gtag) {
        try {
          (window as any).gtag('event', 'notebook-download', {
            href,
          });
        } catch {
          // do nothing if the gtag call fails
        }
      }

      const response = await fetch(href);
      const blob = await response.blob();

      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.style.display = 'none';
      link.href = url;
      const filename = href.split('/').pop();
      link.download = filename;

      document.body.appendChild(link);
      link.click();

      window.URL.revokeObjectURL(url);
      document.body.removeChild(link);
    },
    [href],
  );

  return (
    <a
      className="button button--primary"
      style={{ marginBottom: '1rem', display: 'block', width: 'min-content' }}
      href={href}
      download
      onClick={handleClick}
    >
      {children}
    </a>
  );
}
