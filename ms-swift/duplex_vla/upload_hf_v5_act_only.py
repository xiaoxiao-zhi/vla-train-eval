#!/usr/bin/env python3
"""上传 v5_act_only 到 HuggingFace 公有仓库 dazhiga/v5_act_only."""
import os
import sys
import time

os.environ.setdefault('HF_TOKEN', os.environ.get('HUGGING_FACE_HUB_TOKEN', ''))
os.environ['http_proxy'] = 'http://star-proxy.oa.com:3128'
os.environ['https_proxy'] = 'http://star-proxy.oa.com:3128'
os.environ['no_proxy'] = 'localhost,127.0.0.1'

from huggingface_hub import HfApi

REPO = 'dazhiga/v5_act_only'
FOLDER = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
          'chengzhixu/checkpoint/v5_split_action_fa/hf_release/v5_act_only')


def main():
    if not os.environ.get('HF_TOKEN'):
        raise SystemExit('HF_TOKEN not set; export it before uploading')
    api = HfApi()
    who = api.whoami()
    print('whoami:', who['name'], who['type'], flush=True)
    api.create_repo(REPO, repo_type='model', private=False, exist_ok=True)
    print('REPO_READY', REPO, 'public', flush=True)
    t0 = time.time()
    api.upload_folder(
        repo_id=REPO, repo_type='model', folder_path=FOLDER,
        commit_message='Upload v5_act_only (JarvisVLA action-only ablation, flash-attn run, checkpoint-1162)')
    print('UPLOAD_DONE %.1f min' % ((time.time() - t0) / 60), flush=True)
    files = api.list_repo_files(REPO, repo_type='model')
    print('REPO_FILES', len(files), flush=True)
    for name in sorted(files):
        print(' -', name, flush=True)
    info = api.model_info(REPO)
    assert not info.private, 'repo must be public'
    print('PUBLIC_OK', REPO, flush=True)


if __name__ == '__main__':
    main()
