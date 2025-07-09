import logging
from huggingface_hub import snapshot_download

# Enable detailed logging
logging.basicConfig(level=logging.INFO)

snapshot_download(
    repo_id="google/paligemma-3b-pt-224",
    repo_type="model",
    local_dir="paligemma-3b-pt-224",
    resume_download=True,
    use_auth_token=True,
)
