import { MLFLOW_INTERNAL_PREFIX } from '../../common/utils/TagUtils';

export const NOTE_CONTENT_TAG = MLFLOW_INTERNAL_PREFIX + 'note.content';

export class NoteInfo {
  constructor(content: any) {
    this.content = content;
  }

  static fromTags = (tags: any) => {
    const contentTag = Object.values(tags).find((t) => (t as any).getKey() === NOTE_CONTENT_TAG);
    if (contentTag === undefined) {
      return undefined;
    }
    return new NoteInfo((contentTag as any).getValue());
  };
  content: any;
}
