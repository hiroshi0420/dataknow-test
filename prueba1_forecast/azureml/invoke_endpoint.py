import argparse
import json
import os
import urllib.request


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=36)
    p.add_argument("--scoring-uri", type=str, default=os.getenv("AML_SCORING_URI", ""))
    p.add_argument("--key", type=str, default=os.getenv("AML_ENDPOINT_KEY", ""))
    return p.parse_args()


def main():
    args = parse_args()
    if not args.scoring_uri or not args.key:
        raise RuntimeError("Define AML_SCORING_URI y AML_ENDPOINT_KEY (o pásalos por argumentos)")

    payload = json.dumps({"horizon": args.horizon}).encode("utf-8")

    req = urllib.request.Request(args.scoring_uri, data=payload)
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {args.key}")

    with urllib.request.urlopen(req) as resp:
        out = resp.read().decode("utf-8")
        print(out)


if __name__ == "__main__":
    main()
