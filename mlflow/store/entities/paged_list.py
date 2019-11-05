class PagedList(list):

    def __init__(self, items, token):
        super(PagedList, self).__init__(items)
        self.token = token
