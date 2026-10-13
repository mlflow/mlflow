// Custom View pulls in @a2ui (ESM-only), which transitively pulls date-fns v4.
// These symbols live behind the dedicated `web-shared/model-trace-explorer/custom-view`
// entrypoint (not the main barrel) so that consumers of the standard trace explorer do not drag
// @a2ui onto their static module graph. Import from here only when you actually need custom views.
export {
  CustomViewAssistantConnectorProvider,
  useCustomViewAssistantConnector,
  type CustomViewAssistantConnector,
  type OpenCustomViewAssistantOptions,
} from './assistant/CustomViewAssistantConnector';
export {
  CustomViewDefinitionProvider,
  useCustomViewDefinition,
  useOptionalCustomViewDefinition,
  type CustomViewDefinitionContextValue,
} from './CustomViewDefinitionContext';
export {
  type CustomView,
  type CustomViewApplyTarget,
  toCustomViewApplyTarget,
  parseCustomView,
  serializeCustomView,
  CUSTOM_VIEW_TAG_PREFIX,
  CUSTOM_VIEW_PREFIX_V1,
  CUSTOM_VIEW_TAG_VALUE_SAFE_MAX_BYTES,
  MAX_CUSTOM_VIEWS_PER_EXPERIMENT,
  getUtf8ByteLength,
  viewTagKey,
} from './customViewDefinition';
export {
  getCustomViewAuthoringContext,
  registerCustomViewAuthoringContext,
  latchDispatchedCustomViewApplyTarget,
  getDispatchedCustomViewApplyTarget,
  type CustomViewAuthoringContext,
} from './assistant/customViewAuthoringContext';
export {
  getCustomViewSpecApplier,
  registerCustomViewSpecApplier,
  getCurrentApplierSessionId,
  waitForCustomViewSpecApplier,
  CustomViewValidationError,
  type CustomViewSpecApplier,
  type RenderCustomViewSpec,
  type CustomViewApplyResult,
} from './assistant/customViewSpecApplier';
export { RENDER_CUSTOM_VIEW_TOOL_NAME, buildCustomViewAuthoringGuide } from './agent/buildAgentPrompt';
