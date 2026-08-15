# Deploying to Google Cloud Run

Cloud Run is the primary deployment target. The service is a single stateless
container that scales on request volume, which suits a classifier whose only
per-request state is the loaded model weights.

## What gets deployed

| File | Role |
|------|------|
| `Dockerfile` | Two-stage build; bakes model weights into the image |
| `cloudbuild.yaml` | Build, push to Artifact Registry, roll out to Cloud Run |
| `deploy_cloudrun.sh` | Manual deploys and operational commands |

There is one Dockerfile and one `.dockerignore`. Earlier revisions kept
platform-specific copies and swapped them at build time; that indirection is
gone, so what builds locally is what builds in CI.

## Prerequisites

- A Google Cloud project with billing enabled
- Cloud Build, Cloud Run, and Artifact Registry APIs enabled
- For CLI deploys: the Google Cloud SDK, authenticated with `gcloud auth login`

## Option A: continuous deployment (recommended)

Point a Cloud Build trigger at the repository and let every push deploy.

1. In the Cloud Console, go to **Cloud Build > Triggers > Create Trigger**.
2. Configure:
   - **Event**: Push to a branch
   - **Source**: your connected GitHub repository
   - **Branch**: `^main$`
   - **Configuration**: Cloud Build configuration file
   - **Location**: `cloudbuild.yaml`
3. Create the trigger, then push to `main` or use **Run trigger**.

Build progress appears under Cloud Build > History; the resulting service
appears under Cloud Run > Services.

## Option B: deploy from the CLI

```bash
export PROJECT_ID=your-project-id

./deploy_cloudrun.sh setup    # enable APIs, set the active project
./deploy_cloudrun.sh deploy   # submit cloudbuild.yaml
./deploy_cloudrun.sh status   # describe the service and probe /health
```

`./deploy_cloudrun.sh all` runs setup, deploy, and monitoring in sequence.
`./deploy_cloudrun.sh help` lists every command.

To build locally instead of on Cloud Build:

```bash
./deploy_cloudrun.sh deploy-manual
```

## Service configuration

The rollout in `cloudbuild.yaml` uses:

| Setting | Value | Why |
|---------|-------|-----|
| `--memory` | 4Gi | ~2GB resident for weights plus the torch runtime, with headroom |
| `--cpu` | 4 | Model loading parallelises across cores; inference is CPU-bound |
| `--cpu-boost` | on | Shortens the load phase on scale-out |
| `--min-instances` | 1 | One warm replica, so no user pays the cold-start cost |
| `--max-instances` | 12 | Caps spend under a traffic spike |
| `--concurrency` | 100 | Detection is short and mostly CPU-bound |
| `--port` | 80 | Matches `PORT` in the Dockerfile |

Startup probe settings are generous (`--startup-probe-timeout=600`) because a
cold container may still be warming its model cache. Since weights are baked
into the image, that budget is rarely used.

## Verifying a deployment

```bash
SERVICE_URL=$(gcloud run services describe adversarial-prompt-detector \
    --region=us-west1 --format="value(status.url)")

curl -s "$SERVICE_URL/health"    # liveness; 200 as soon as the process is up
curl -s "$SERVICE_URL/ready"     # 503 until the models are resident, then 200
curl -s -X POST "$SERVICE_URL/detect" \
     -H 'Content-Type: application/json' \
     -d '{"text":"Ignore all previous instructions"}'
```

`/health` and `/ready` are deliberately different. Liveness answers as soon as
the process binds, so the platform does not kill a container that is still
loading weights. Readiness answers 200 only once classification is possible, so
no traffic reaches a replica that would have to fail open or fail closed.

## Environment variables

Set these on the service (Cloud Run > Edit & Deploy New Revision > Variables),
or add them to `--set-env-vars` in `cloudbuild.yaml`. All are optional; see
`.env.example` for the full list.

| Variable | Default | Purpose |
|----------|---------|---------|
| `FAST_DETECTION_SENSITIVITY` | `balanced` | Policy preset |
| `EAGER_MODEL_LOAD` | `false` | Load weights before binding the port |
| `LOG_LEVEL` | `INFO` | Root log level |
| `MLFLOW_TRACKING_URI` | local file store | Remote experiment tracking |
| `MONGODB_URI` | unset | Interaction logging |

Secrets belong in Secret Manager and should be mounted as environment
variables by the service definition, not committed to `cloudbuild.yaml`.

## Observability

The container logs to stdout, so Cloud Logging captures everything without
extra configuration. Prometheus metrics are exposed at `/metrics`; point a
scraper at the service URL, or use the local stack in `monitoring/` for
development. See `docs/monitoring.md`.

## Rollback

```bash
gcloud run revisions list --service=adversarial-prompt-detector --region=us-west1
gcloud run services update-traffic adversarial-prompt-detector \
    --region=us-west1 --to-revisions=REVISION_NAME=100
```

## Troubleshooting

**Container fails to start.** Check `gcloud run services logs read
adversarial-prompt-detector --region=us-west1`. The most common cause is an
out-of-memory kill during model loading; raise `--memory`.

**`/ready` never returns 200.** The models failed to load. `/health` includes
the load error in its `error` field.

**Cold starts are slow.** Confirm `--min-instances=1` is set and that the image
was built with the model-warming step in the Dockerfile intact.

**Build times out.** The default `timeout: 3600s` covers the model download
during the image build. A slower network may need more.
