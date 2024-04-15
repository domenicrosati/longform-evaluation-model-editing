import hashlib
import json


# construct sample id from sample
# use md5 hash of requested rewrite
def get_sample_id(sample):
    if 'requested_rewrite' in sample:
        return hashlib.md5(
            json.dumps(sample["requested_rewrite"]).encode()
        ).hexdigest()
    else:
        return hashlib.md5(
            json.dumps(sample).encode()
        ).hexdigest()
