from abc import ABCMeta, abstractmethod


class UnityCatalogFileSystem:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _list(self, path):
        """
        List files under the specified path
        :param path:
        :return:
        """
        pass

    @abstractmethod
    def _copy_file(self, src_path, dst_path):
        """
        Copy file from the specified source (local) to destination (remote) path.
        :param src_path:
        :param dst_path:
        :return:
        """
        pass


    def _download_file(self, src_path, dst_path):
        """
        Download the specified file from the source path to the destination path
        :param src_path: Source path (remote) of the file to download
        :param dst_path: Destination directory on the local filesystem
        :return:
        """

    def download(self, src_path, dst_path):
        """
        Download files from the source path (remote) to the destination path
        on the local filesystem
        :param src_path:
        :param dst_path:
        :return:
        """
        # TODO: recursively list and download files
        pass

    def upload(self, src_path, dst_path):
        """

        :param src_path:
        :param dst_path:
        :return:
        """
