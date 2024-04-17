package main

import (
	"fmt"

	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
	"github.com/mlflow/mlflow/mlflow/go/pkg/protos/artifacts"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
)

type Endpoint struct {
	Method string
	Path   string
}

func main() {
	servicEndpoints := make([]Endpoint, 0)

	services := []struct {
		Name       string
		Descriptor protoreflect.FileDescriptor
	}{
		{"MlflowService", protos.File_service_proto},
		{"ModelRegistryService", protos.File_model_registry_proto},
		{"MlflowArtifactsService", artifacts.File_mlflow_artifacts_proto},
	}

	for _, service := range services {
		serviceInfo := service.Descriptor.Services().ByName(protoreflect.Name(service.Name))

		if serviceInfo == nil {
			panic(fmt.Sprintf("Service %s not found", service.Name))
		}

		methods := serviceInfo.Methods()
		for mIdx := range methods.Len() {
			method := methods.Get(mIdx)
			options := method.Options()

			extension := proto.GetExtension(options, protos.E_Rpc)
			for _, endpoint := range extension.(*protos.DatabricksRpcOptions).Endpoints {
				servicEndpoints = append(servicEndpoints, Endpoint{Method: *endpoint.Method, Path: *endpoint.Path})
			}
		}
	}

	for e := range servicEndpoints {
		println(servicEndpoints[e].Method, servicEndpoints[e].Path)
	}
}
