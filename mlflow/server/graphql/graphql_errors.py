import graphene


class ErrorDetail(graphene.ObjectType):
    # NOTE: This is not an exhaustive list, might need to add more things in the future if needed.
    field = graphene.String()
    message = graphene.String()


class ApiError(graphene.ObjectType):
    code = graphene.String()
    message = graphene.String()
    help_url = graphene.String()
    trace_id = graphene.String()
    error_details = graphene.List(ErrorDetail)
