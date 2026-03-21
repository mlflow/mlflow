// eslint-disable-next-line import/no-namespace
import type * as pdfjs from 'pdfjs-dist';

export function setupReactPDFWorker(pdfjsInstance: typeof pdfjs) {
  pdfjsInstance.GlobalWorkerOptions.workerSrc = new URL(
    'pdfjs-dist/build/pdf.worker.min.mjs',
    import.meta.url,
  ).toString();
}
