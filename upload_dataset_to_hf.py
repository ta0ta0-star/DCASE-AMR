from huggingface_hub import HfApi

api = HfApi()

# Organization名と新しく作りたいデータセット名を指定
org_name = "ta0ta00h" # ★実際のOrganization名に書き換える
dataset_name = "m2d_text_paraphrase"      # ★好きなデータセット名に書き換える
repo_id = f"{org_name}/{dataset_name}"

print(f"{repo_id} にアクセスします...")

# リポジトリを新規作成（すでにある場合はそのまま続行）
api.create_repo(
    repo_id=repo_id,
    repo_type="dataset",
    private=True,
    exist_ok=True
)

local_file_path = "/data/y255618g/dcase2026_task6/m2d_text_paraphrase.tar.gz"  # ★アップロードしたいローカルのtar.gzファイルのパス
repo_file_name = "m2d_text_paraphrase.tar.gz "     # ★Hugging Face上で保存するファイル名（通常はローカルと同じでOK）

print(f"{local_file_path} をアップロード中...")

api.upload_file(
    path_or_fileobj=local_file_path,
    path_in_repo=repo_file_name,
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Add tar.gz archive"
)

print("🎉 tar.gzファイルのアップロードが完了しました！")
