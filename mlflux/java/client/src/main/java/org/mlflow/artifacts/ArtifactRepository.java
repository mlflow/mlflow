package org.mlflow.artifacts;

import java.io.File;
import java.util.List;

import org.mlflow.api.proto.Service.FileInfo;

/**
 * Allows logging, listing, and downloading artifacts against a remote Artifact Repository.
 * This is used for storing potentially-large objects associated with MLflow runs.
 */
public interface ArtifactRepository {

  /**
   * Uploads the given local file to the run's root artifact directory. For example,
   *
   *   <pre>
   *   logArtifact("/my/localModel")
   *   listArtifacts() // returns "localModel"
   *   </pre>
   *
   * @param localFile File to upload. Must exist, and must be a simple file (not a directory).
   */
  void logArtifact(File localFile);

  /**
   * Uploads the given local file to an artifactPath within the run's root directory. For example,
   *
   *   <pre>
   *   logArtifact("/my/localModel", "model")
   *   listArtifacts("model") // returns "model/localModel"
   *   </pre>
   *
   * (i.e., the localModel file is now available in model/localModel).
   *
   * @param localFile File to upload. Must exist, and must be a simple file (not a directory).
   * @param artifactPath Artifact path relative to the run's root directory. Should NOT
   *                     start with a /.
   */
  void logArtifact(File localFile, String artifactPath);

  /**
   * Uploads all files within the given local director the run's root artifact directory.
   * For example, if /my/local/dir/ contains two files "file1" and "file2", then
   *
   *   <pre>
   *   logArtifacts("/my/local/dir")
   *   listArtifacts() // returns "file1" and "file2"
   *   </pre>
   *
   * @param localDir Directory to upload. Must exist, and must be a directory (not a simple file).
   */
  void logArtifacts(File localDir);


  /**
   * Uploads all files within the given local director an artifactPath within the run's root
   * artifact directory. For example, if /my/local/dir/ contains two files "file1" and "file2", then
   *
   *   <pre>
   *   logArtifacts("/my/local/dir", "model")
   *   listArtifacts("model") // returns "model/file1" and "model/file2"
   *   </pre>
   *
   * (i.e., the contents of the local directory are now available in model/).
   *
   * @param localDir Directory to upload. Must exist, and must be a directory (not a simple file).
   * @param artifactPath Artifact path relative to the run's root directory. Should NOT
   *                     start with a /.
   */
  void logArtifacts(File localDir, String artifactPath);

  /**
   * Lists the artifacts immediately under the run's root artifact directory. This does not
   * recursively list; instead, it will return FileInfos with isDir=true where further
   * listing may be done.
   */
  List<FileInfo> listArtifacts();

  /**
   * Lists the artifacts immediately under the given artifactPath within the run's root artifact
   * irectory. This does not recursively list; instead, it will return FileInfos with isDir=true
   * where further listing may be done.
   * @param artifactPath Artifact path relative to the run's root directory. Should NOT
   *                     start with a /.
   */
  List<FileInfo> listArtifacts(String artifactPath);

  /**
   * Returns a local directory containing *all* artifacts within the run's artifact directory.
   * Note that this will download the entire directory path, and so may be expensive if
   * the directory a lot of data.
   */
  File downloadArtifacts();

  /**
   * Returns a local file or directory containing all artifacts within the given artifactPath
   * within the run's root artifactDirectory. For example, if "model/file1" and "model/file2"
   * exist within the artifact directory, then
   *
   *   <pre>
   *   downloadArtifacts("model") // returns a local directory containing "file1" and "file2"
   *   downloadArtifacts("model/file1") // returns a local *file* with the contents of file1.
   *   </pre>
   *
   * Note that this will download the entire subdirectory path, and so may be expensive if
   * the subdirectory a lot of data.
   *
   * @param artifactPath Artifact path relative to the run's root directory. Should NOT
   *                     start with a /.
   */
  File downloadArtifacts(String artifactPath);
}
