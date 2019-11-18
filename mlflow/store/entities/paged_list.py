class PagedList(list):

    def __init__(self, items, token, total_run_count=10):
        super(PagedList, self).__init__(items)
        self.token = token
        self.total_run_count = total_run_count
