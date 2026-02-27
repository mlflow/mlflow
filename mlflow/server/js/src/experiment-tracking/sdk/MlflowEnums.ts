/**
 * DO NOT EDIT!!!
 *
 * @NOTE(dli) 01-12-2017
 *   This file is generated. For now, it is a snapshot of the proto enums as of
 *   Sep 17, 2018 6:47:39 PM. We will update the generation pipeline to actually
 *   place these generated enums in the correct location shortly.
 */

export enum SourceType {
  NOTEBOOK = 'NOTEBOOK',
  JOB = 'JOB',
  PROJECT = 'PROJECT',
  LOCAL = 'LOCAL',
  UNKNOWN = 'UNKNOWN',
}

export const ViewType = {
  ACTIVE_ONLY: 'ACTIVE_ONLY',
  DELETED_ONLY: 'DELETED_ONLY',
  ALL: 'ALL',
};
export enum ModelGatewayRouteTask {
  LLM_V1_COMPLETIONS = 'llm/v1/completions',
  LLM_V1_CHAT = 'llm/v1/chat',
  LLM_V1_EMBEDDINGS = 'llm/v1/embeddings',
}
