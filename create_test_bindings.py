#!/usr/bin/env python3
"""Create test bindings for routes."""
import requests
import uuid

BASE_URL = "http://localhost:5000"

def list_routes():
    """List all routes."""
    resp = requests.get(f"{BASE_URL}/ajax-api/3.0/mlflow/secrets/list-routes")
    resp.raise_for_status()
    return resp.json().get("routes", [])

def create_binding(route_id, resource_type, resource_id, field_name):
    """Create a binding."""
    resp = requests.post(
        f"{BASE_URL}/ajax-api/3.0/mlflow/secrets/bind-route",
        json={
            "route_id": route_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "field_name": field_name,
        }
    )
    resp.raise_for_status()
    return resp.json()

def main():
    # Get routes
    routes = list_routes()
    print(f"Found {len(routes)} routes")

    # Take first 2 routes
    for route in routes[:2]:
        route_id = route["route_id"]
        route_name = route.get("name") or route.get("model_name", "unknown")
        print(f"\nCreating bindings for route: {route_name} ({route_id})")

        # Create 5 bindings per route
        for i in range(5):
            resource_id = f"test_scorer_{uuid.uuid4().hex[:8]}"
            try:
                binding = create_binding(
                    route_id=route_id,
                    resource_type="scorer",
                    resource_id=resource_id,
                    field_name="llm_judge_config"
                )
                print(f"  ✓ Created binding for {resource_id}")
            except Exception as e:
                print(f"  ✗ Failed to create binding for {resource_id}: {e}")

if __name__ == "__main__":
    main()
