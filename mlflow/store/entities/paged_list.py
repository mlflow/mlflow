class PagedList(list):
    def __init__(self, items, token):
        super().__init__(items)
        self.token = token
