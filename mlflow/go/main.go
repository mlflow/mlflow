package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"

	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
	"github.com/mlflow/mlflow/mlflow/go/pkg/protos/artifacts"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
)

type Endpoint struct {
	Method string
	Path   string
}

func logBatchHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	defer r.Body.Close()
	if err != nil {
		http.Error(w, "Error reading request body", http.StatusInternalServerError)
		return
	}

	var logBatchRequest protos.LogBatch
	err = json.Unmarshal(body, &logBatchRequest)
	if err != nil {
		http.Error(w, "Error parsing JSON body", http.StatusBadRequest)
		return
	}

	w.WriteHeader(http.StatusOK)
}

var proxy *httputil.ReverseProxy

func initProxy(randomPort int) {
	targetURL, _ := url.Parse(fmt.Sprintf("http://localhost:%d", randomPort))
	proxy = httputil.NewSingleHostReverseProxy(targetURL)

	proxy.Director = func(req *http.Request) {
		req.URL.Scheme = targetURL.Scheme
		req.URL.Host = targetURL.Host
		req.Host = targetURL.Host // Optionally set the Host header to the target host
		// Ensure headers are copied over
		req.Header = make(http.Header)
		for h, val := range req.Header {
			req.Header[h] = val
		}
	}
}

func proxyHandler(w http.ResponseWriter, r *http.Request) {
	proxy.ServeHTTP(w, r)
}

func main() {
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
			fmt.Println(method.FullName())
			options := method.Options()

			extension := proto.GetExtension(options, protos.E_Rpc)
			for _, endpoint := range extension.(*protos.DatabricksRpcOptions).Endpoints {
				serviceEndpoints[string(method.FullName())] = Endpoint{Method: *endpoint.Method, Path: *endpoint.Path}
				// servicEndpoints = append(servicEndpoints, Endpoint{Method: *endpoint.Method, Path: *endpoint.Path})
			}
		}
	}

	fmt.Printf("%#v\n", serviceEndpoints)

	handlers := map[string]func(http.ResponseWriter, *http.Request){
		"mlflow.MlflowService.logBatch": logBatchHandler,
	}

	for service, handler := range handlers {
		endpoint := serviceEndpoints[service]
		http.HandleFunc(endpoint.Path, handler)
	}

	// var p *protos.CreateRegisteredModel
	// fn := p.ProtoReflect().Descriptor().Name()
	// fmt.Println(fn)

	// for e := range servicEndpoints {
	// 	println(servicEndpoints[e].Method, servicEndpoints[e].Path)
	// }

	// // TODO: parse port from arguments
	initProxy(9001)

	// http.HandleFunc("/mlflow/runs/log-batch", logBatchHandler)
	http.HandleFunc("/", proxyHandler)
	log.Fatal(http.ListenAndServe(":8080", nil))
}
