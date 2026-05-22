import type { JsonRecordEditorProps } from './JsonRecordEditor';
import { JsonRecordEditor } from './JsonRecordEditor';

/**
 * OSS stub for the lazy editor wrapper. Universe lazy-loads Monaco off the critical path;
 * the OSS stub is a textarea, so lazy-loading buys nothing and we re-export directly. Kept
 * as a separate file so callers' imports don't change.
 */
export const LazyJsonRecordEditor = (props: JsonRecordEditorProps) => <JsonRecordEditor {...props} />;
