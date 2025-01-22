import { MouseEvent, ReactNode, useCallback } from "react";

interface NotebookDownloadButtonProps {
  children: ReactNode;
  href: string;
}

export function NotebookDownloadButton({
  children,
  href,
}: NotebookDownloadButtonProps) {
  const handleClick = useCallback(
    async (e: MouseEvent) => {
      e.preventDefault();
      const response = await fetch(href);
      const blob = await response.blob();

      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.style.display = "none";
      link.href = url;
      const filename = href.split("/").pop();
      link.download = filename;

      document.body.appendChild(link);
      link.click();

      window.URL.revokeObjectURL(url);
      document.body.removeChild(link);
    },
    [href]
  );

  return (
    <a
      className="button button--primary"
      style={{ marginBottom: "1rem", display: "block", width: "min-content" }}
      href={href}
      download
      onClick={handleClick}
    >
      {children}
    </a>
  );
}
