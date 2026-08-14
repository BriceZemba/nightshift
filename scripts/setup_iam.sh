#!/usr/bin/env bash
# Scoped identities for Nightshift.
#
# Three service accounts, each holding only what its job requires. The split is not
# ceremony: this system executes code a model wrote in response to text a stranger
# authored, so the blast radius of each component is bounded by what its identity can do,
# not by what its code intends to do.
#
#   nightshift-runtime   reasons, reads Firestore, calls Model Armor. Never executes
#                        untrusted code.
#   nightshift-verifier  runs model-written patches against a repository's test suite.
#                        Holds NO roles at all. If a patch escapes the sandbox it inherits
#                        an identity that cannot read, write or call anything.
#   nightshift-scheduler triggers the nightly run. Can invoke one Cloud Run service and
#                        nothing else.
#
# Usage:  ./scripts/setup_iam.sh YOUR_PROJECT_ID [REGION]

set -euo pipefail

PROJECT="${1:?usage: setup_iam.sh PROJECT_ID [REGION]}"
REGION="${2:-us-central1}"
SERVICE="nightshift"

RUNTIME="nightshift-runtime@${PROJECT}.iam.gserviceaccount.com"
VERIFIER="nightshift-verifier@${PROJECT}.iam.gserviceaccount.com"
SCHEDULER="nightshift-scheduler@${PROJECT}.iam.gserviceaccount.com"

echo "Project: ${PROJECT}  Region: ${REGION}"

# --- create the identities ---------------------------------------------------
for pair in \
  "nightshift-runtime:Nightshift agent runtime" \
  "nightshift-verifier:Nightshift patch verifier (deliberately unprivileged)" \
  "nightshift-scheduler:Nightshift nightly trigger"
do
  name="${pair%%:*}"
  desc="${pair#*:}"
  gcloud iam service-accounts create "${name}" \
    --project="${PROJECT}" \
    --display-name="${desc}" 2>/dev/null || echo "  ${name} already exists"
done

# --- runtime: the minimum needed to reason and remember ----------------------
# aiplatform.user  : call Gemini and Gemma through Vertex
# datastore.user   : read and write agent state in Firestore
# modelarmor.user  : screen untrusted advisory text
#
# Deliberately absent: no storage admin, no project editor, no secret admin, and no
# permission to modify its own IAM policy.
for role in \
  "roles/aiplatform.user" \
  "roles/datastore.user" \
  "roles/modelarmor.user"
do
  gcloud projects add-iam-policy-binding "${PROJECT}" \
    --member="serviceAccount:${RUNTIME}" \
    --role="${role}" \
    --condition=None \
    --quiet >/dev/null
  echo "  runtime + ${role}"
done

# --- verifier: nothing, and that is the point --------------------------------
# No roles are granted. This identity exists so that the component executing
# model-written code has something to be, and that something can do nothing.
echo "  verifier + (no roles, intentionally)"

# --- scheduler: invoke exactly one service -----------------------------------
gcloud run services add-iam-policy-binding "${SERVICE}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --member="serviceAccount:${SCHEDULER}" \
  --role="roles/run.invoker" \
  --quiet >/dev/null 2>&1 || \
  echo "  (deploy the ${SERVICE} service first, then re-run to bind the scheduler)"

# --- secrets: read one secret, not all of them -------------------------------
# Grant per-secret rather than project-wide, so the runtime cannot read secrets belonging
# to anything else in the project.
for secret in "nightshift-github-token" "nightshift-run-token"; do
  gcloud secrets add-iam-policy-binding "${secret}" \
    --project="${PROJECT}" \
    --member="serviceAccount:${RUNTIME}" \
    --role="roles/secretmanager.secretAccessor" \
    --quiet >/dev/null 2>&1 || echo "  (secret ${secret} not created yet)"
done

cat <<EOF

Done. Deploy with the runtime identity attached:

  gcloud run deploy ${SERVICE} \\
    --source . \\
    --region=${REGION} \\
    --service-account=${RUNTIME} \\
    --allow-unauthenticated \\
    --min-instances=0 --max-instances=3

Verify what each identity actually holds:

  gcloud projects get-iam-policy ${PROJECT} \\
    --flatten="bindings[].members" \\
    --filter="bindings.members:nightshift-" \\
    --format="table(bindings.role, bindings.members)"

That last command is worth putting on screen in the demo: it shows the verifier holding
no roles at all.
EOF
