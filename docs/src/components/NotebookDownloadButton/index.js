import { useCallback } from 'react';
import { Version } from '@site/src/constants';
export function NotebookDownloadButton({ children, href }) {
    const handleClick = useCallback(async (e) => {
        e.preventDefault();
        if (window.gtag) {
            try {
                window.gtag('event', 'notebook-download', {
                    href,
                });
            }
            catch {
                // do nothing if the gtag call fails
            }
        }
        if (!Version.includes('dev')) {
            // Replace 'master' with the current version to pin the download to the released version
            // and avoid 404 errors
            href = href.replace(/\/master\//, `/v${Version}/`);
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
    }, [href]);
    return (<a className="button button--primary" style={{ marginBottom: '1rem', display: 'block', width: 'min-content' }} href={href} download onClick={handleClick}>
      {children}
    </a>);
}
