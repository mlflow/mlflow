/**
 * Get a unique key for a model version object.
 * @param modelName
 * @param version
 * @returns {string}
 */
export const getModelVersionKey = (modelName, version) => `${modelName}_${version}`;

export const getProtoField = (fieldName) => `${fieldName}`;
