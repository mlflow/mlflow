package main

import (
	"fmt"

	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
)

func main() {
	s := protos.File_service_proto.Services().ByName("MlflowService")
	m := s.Methods()
	serviceMethod := m.Get(0)
	options := serviceMethod.Options()
	fmt.Printf("%#v\n%s\n", options, options)
	// rpc := protos.File_databricks_proto.Extensions().ByName("rpc")
	// var rpc protoreflect.FieldDescriptor
	options.ProtoReflect().Range(func(fd protoreflect.FieldDescriptor, v protoreflect.Value) bool {
		// rpc = fd
		fmt.Printf("field: %#v\n", fd)
		fmt.Printf("value: %#v\n", v)
		return true
	})
	// fmt.Printf("%#v\n", options.ProtoReflect().ProtoMethods().)
	// d := options.ProtoReflect().Descriptor()
	// fmt.Printf("%#v", d.Fields().ByNumber(35).)
	extension := proto.GetExtension(options, protos.E_Rpc)
	for _, endpoint := range extension.(*protos.DatabricksRpcOptions).Endpoints {
		fmt.Printf("endpoint: %#v\n", endpoint)
		fmt.Printf("%s %s\n", *endpoint.Method, *endpoint.Path)
	}
}
