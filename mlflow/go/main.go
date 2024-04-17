package main

import (
	"fmt"
	"log"
	"net/http"

	"github.com/mlflow/mlflow/mlflow/go/pkg/server"
)

func main() {
	serviceEndpoints := server.GetServiceEndpoints()

	for service, handler := range server.Handlers {
		endpoint, ok := serviceEndpoints[service]
		if ok {
			http.HandleFunc(endpoint.GetPattern(), handler)
		} else {
			// TODO: maybe panic here?
			log.Printf("No handler found for %s", service)
		}
	}

	for _, s := range serviceEndpoints {
		fmt.Println(s.GetPattern())
	}

	// TODO: parse port from arguments
	server.InitProxy(9001)
	log.Fatal(http.ListenAndServe(":8080", nil))
}
