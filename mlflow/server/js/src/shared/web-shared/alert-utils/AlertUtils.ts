/**
 * OSS-equivalent of the internal AlertUtils logging utility.
 *
 * In the Databricks codebase this fires structured alerts to observability
 * backends. In OSS we simply centralise the call so that the `no-console`
 * lint rule only needs to be suppressed in one place.
 */

type DebugBlob = string | Record<string, unknown> | Error | unknown;

// eslint-disable-next-line @typescript-eslint/no-extraneous-class
export class AlertUtils {
  static log(eventId: string, debugBlob: DebugBlob = '') {
    // eslint-disable-next-line no-console
    console.error(`[${eventId}]`, debugBlob);
  }
}
