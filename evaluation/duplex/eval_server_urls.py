import sys as _sys
for _p in ('/apdcephfs/private_chengzhixu/evaluation', '/apdcephfs/private_chengzhixu/JarvisVLA'):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from urllib.parse import urlsplit


def parse_base_urls(value):
    urls = [url.strip().rstrip('/') for url in value.split(',')] if value else []
    if len(set(urls)) != len(urls):
        raise ValueError('Server URLs must be distinct')
    for url in urls:
        parsed = urlsplit(url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.path != '/v1':
            raise ValueError('Expected an HTTP(S) server URL ending in /v1: %s' % url)
    return urls


class URLSlots:
    def __init__(self, urls, workers):
        if len(urls) != len(set(urls)) or (urls and workers > len(urls)):
            raise ValueError('Each concurrent task needs a distinct server URL')
        self.urls, self.owners = urls, {}

    def acquire(self, owner):
        if not self.urls:
            return None
        url = next((url for url in self.urls if url not in self.owners.values()), None)
        if url is None or owner in self.owners:
            raise RuntimeError('No free server slot or owner already assigned')
        self.owners[owner] = url
        return url

    def release(self, owner):
        self.owners.pop(owner, None)
