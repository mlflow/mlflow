/**
 * Checks if a given model name is a valid UC entity name.
 * A valid UC entity name follows the pattern: "catalog.schema.model".
 * This is used to distinguish from other registries model names which should not contain dots.
 */
export const isUCModelName = (name: string) => Boolean(name.match(/^[^. /]+\.[^. /]+\.[^. /]+$/));
