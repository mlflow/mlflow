export async function recordEvent(
  eventName: string,
  additionalTags?: Record<string, string>,
  eventData?: string,
  // eslint-disable-next-line @typescript-eslint/no-empty-function
) {}

export * from './UserActionErrorHandler';
