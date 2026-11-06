/**
 * Entry point that older OpenCode V1 releases load.
 *
 * OpenCode V2 and newer V1 releases load the package's `./server` export
 * instead. This version only supports OpenCode V2, so on V1 the plugin tells
 * the user which version to pin.
 */

const V1_UNSUPPORTED_MESSAGE =
  '[mlflow] @mlflow/opencode 1.x requires OpenCode V2 and does not trace on OpenCode V1. ' +
  'To keep tracing on OpenCode V1, pin "@mlflow/opencode@0.4" in your opencode.json plugin list.';

interface V1PluginInput {
  client?: {
    tui?: {
      showToast?: (options: {
        body: { title?: string; message: string; variant: 'warning' };
      }) => Promise<unknown>;
    };
  };
}

// Older OpenCode V1 releases call every exported function as a plugin, so do not
// export other functions or values from this module.
export const MLflowTracingPlugin = (input?: V1PluginInput): Promise<Record<string, never>> => {
  console.error(V1_UNSUPPORTED_MESSAGE);
  // The TUI may not be connected yet, and console output is not always visible.
  void input?.client?.tui
    ?.showToast?.({
      body: { title: 'MLflow tracing', message: V1_UNSUPPORTED_MESSAGE, variant: 'warning' },
    })
    .catch(() => {});
  return Promise.resolve({});
};

export default MLflowTracingPlugin;
