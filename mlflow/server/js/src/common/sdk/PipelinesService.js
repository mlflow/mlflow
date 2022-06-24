import { getJson } from '../utils/FetchUtils';

export class PipelinesService {
  /**
   * Get a pipeline's details
   */
  static getPipeline = (pipelineId) =>
    getJson({
      relativeUrl: `/ajax-api/2.0/pipelines/${pipelineId}`,
    });
}
