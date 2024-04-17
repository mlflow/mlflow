package server

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
	"github.com/mlflow/mlflow/mlflow/go/pkg/protos/artifacts"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
)

type Endpoint struct {
	Method string
	Path   string
}

// Get the pattern used in http.HandleFunc
// Notice that method is only Go 1.22, see https://tip.golang.org/doc/go1.22#enhanced_routing_patterns
func (e Endpoint) GetPattern() string {
	// TODO: e.Path cannot be trusted, it could be something like /mlflow-artifacts/artifacts/<path:artifact_path>
	// Which would need to converted to /mlflow-artifacts/artifacts/{path}
	re := regexp.MustCompile(`<[^>]+:([^>]+)>`)
	path := re.ReplaceAllStringFunc(e.Path, func(s string) string {
		parts := strings.Split(s, ":")
		return fmt.Sprintf("{%s}", strings.Trim(parts[0], "< "))
	})

	return fmt.Sprintf("%s %s", strings.ToUpper(e.Method), path)
}

// Discover all the service endpoints from the protobuf files
func GetServiceEndpoints() map[string]Endpoint {
	serviceEndpoints := make(map[string]Endpoint, 0)

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
				key := fmt.Sprintf("%s_%s", *endpoint.Method, string(method.FullName()))
				serviceEndpoints[key] = Endpoint{Method: *endpoint.Method, Path: *endpoint.Path}
			}
		}
	}

	return serviceEndpoints
}
