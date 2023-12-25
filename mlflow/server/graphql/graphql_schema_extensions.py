import graphene


class Test(graphene.ObjectType):
    output = graphene.String(description="Echoes the input string")


class Query(graphene.ObjectType):
    test = graphene.Field(Test, input_string=graphene.String(), description="Simple echoing field")

    def resolve_test(self, info, input_string):
        return {"output": input_string}


schema = graphene.Schema(query=Query)
