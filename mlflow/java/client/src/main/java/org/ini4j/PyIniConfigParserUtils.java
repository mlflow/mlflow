package org.ini4j;

/**
 * Helper to expose package-private fields of ini4j.
 */
public class PyIniConfigParserUtils {
  /**
   * Returns the [DEFAULT] section of the ini file. Unclear why this is package-private.
   */
  public static Profile.Section getDefaultSection(ConfigParser config) {
    return ((ConfigParser.PyIni) config.getIni()).getDefaultSection();
  }

  /**
   * Returns the Section with the given name from the ini file. Unclear why package-private.
   */
  public static Profile.Section getSection(ConfigParser config, String sectionName) {
    return config.getIni().get(sectionName);
  }
}