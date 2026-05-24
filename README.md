# Niti Engine 🚀

`niti-engine` is a high-performance, containerized JSON REST API for [PolicyEngine US](https://github.com/PolicyEngine/policyengine-us). It provides an out-of-process API for tax and benefit calculations, enabling any web application or service (written in Go, Node.js, Rust, Ruby, etc.) to run deep policy simulations without being restricted to a Python environment. Additionally, it incorporates custom formulas and logic missing from the base PolicyEngine library, such as Affordable Care Act (ACA) Premium Tax Credit (PTC) repayment limitations and associated penalties.

---

## Key Benefits

* **High Performance & Optimization:** PolicyEngine has hundreds of deep parameter structures and geographic county FIPS datasets. `niti-engine` applies pre-compilation, lazy memory caching, and optimized startup lifespan warmups to serve complex calculations with extremely low latency out of the box.
* **Service Decoupling & Independent Scaling:** Offloads CPU-intensive and memory-heavy simulation runs to a dedicated microservice that can be containerized, scaled, and managed separately from client-facing application layers.
* **Strict Egress Security & Offline Mode:** Pre-configured to support high-security, zero-egress production environments. All geographic and third-party dependencies are fully patched to run 100% offline, making it perfect for air-gapped, zero-gateway, or private VPC deployments.

---

## Performance Enhancements Included Out-of-the-Box

Standard PolicyEngine setups face cold-start parameter parsing and CSV dataset loading latencies. `niti-engine` resolves these automatically:
* **Pre-Initialized TaxBenefitSystem:** Reuses pre-compiled, cached global parameters instead of re-instantiating the US tax ruleset on every single request.
* **Lazy County FIPS Caching:** Speeds up geographic and county-level lookups by lazy-loading and globally caching the FIPS dataset in-memory on the first query, completely eliminating redundant CSV file I/O overhead.
* **Offline HuggingFace Patching:** Intercepts `hf_hub_download` metadata checks to bypass slow synchronous internet validation checks, utilizing local cache files immediately.
* **FastAPI Lifespan Warmup:** Runs a baseline calculation warmup at server boot so that the first API request computes instantly.

---

## API Reference

### `POST /calculate`
Executes federal and state simulations for a given household situation dictionary.

**Request Payload:**
```json
{
  "situation": {
    "people": {
      "you": {
        "age": { "2025": 45 },
        "employment_income": { "2025": 80000 }
      }
    },
    "tax_units": {
      "tax_unit": {
        "members": ["you"]
      }
    },
    "families": {
      "family": { "members": ["you"] }
    },
    "households": {
      "household": {
        "members": ["you"],
        "state_code": { "2025": "CA" }
      }
    }
  },
  "variables": ["income_tax", "ca_income_tax"],
  "period": 2025
}
```

**Response Payload:**
```json
{
  "arrays": {
    "income_tax": [6890.50],
    "ca_income_tax": [2450.00]
  }
}
```

---
## Run the API Server
To build and run the FastAPI web server locally on port `8000`:
```bash
# Build the Docker image
docker build -t niti-engine .

# Start the API server container
docker run -p 8000:8000 niti-engine
```
---

## CLI & Task Runner (`run.sh`)

We provide a local developer task runner [`run.sh`](run.sh) that allows you to easily build, test, and run scripts inside the Docker container without needing to configure a local Python environment or dependencies on your host.

### Run a Script
Run python scripts directly in the container:
```bash
./run.sh run benchmark.py
```

### Force a Rebuild
Add the `--build` flag to force build the Python container before running:
```bash
./run.sh --build run benchmark.py
```

### Drop into an Interactive Shell
Inspect packages, trace math calculations, or run interactive Python REPLs inside the environment:
```bash
./run.sh shell
```

---

## Docker Integration

Integrating `niti-engine` into your service stack is extremely simple:

```yaml
services:
  niti-engine:
    build:
      context: ./niti-engine
      dockerfile: Dockerfile
    expose:
      - "8000"
    restart: always

  web-app:
    build: ./my-web-app
    ports:
      - "8080:8080"
    environment:
      - NITI_ENGINE_URL=http://niti-engine:8000
```

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See the [LICENSE](LICENSE) file for details.

---

## Git Hook (Auto-sync .dockerignore)

We provide a helper script to automatically synchronize `.gitignore` rules into `.dockerignore` while keeping your Docker-specific files excluded. 

To enable this pre-commit hook in your local Git repository, run this command from the root of the repository:

```bash
ln -sf ../../scripts/sync-dockerignore.sh .git/hooks/pre-commit
```

Whenever you run `git commit`, this hook runs automatically, updates `.dockerignore`, and stages it before committing.
