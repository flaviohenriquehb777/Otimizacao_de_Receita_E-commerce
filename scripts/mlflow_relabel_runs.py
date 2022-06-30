import os
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient
import dagshub


def _to_epoch_ms(iso: str) -> int:
    dt = datetime.fromisoformat(iso.replace('Z', '+00:00'))
    return int(dt.timestamp() * 1000)


def main():
    owner = os.environ.get('DAGSHUB_OWNER')
    repo = os.environ.get('DAGSHUB_REPO')
    tracking_uri_env = os.environ.get('MLFLOW_TRACKING_URI')
    if owner and repo:
        dagshub.init(repo, owner, mlflow=True)
    if tracking_uri_env:
        mlflow.set_tracking_uri(tracking_uri_env)

    start_iso = os.environ.get('PERIOD_START_ISO', '2022-01-01T00:00:00Z')
    end_iso = os.environ.get('PERIOD_END_ISO', '2022-06-30T23:59:59Z')
    start_ms = _to_epoch_ms(start_iso)
    end_ms = _to_epoch_ms(end_iso)
    exp_filter = os.environ.get('MLFLOW_EXPERIMENT_FILTER')

    client = MlflowClient()
    try:
        exps = client.search_experiments()
    except Exception:
        exps = []

    for e in exps:
        if exp_filter and exp_filter not in (e.name or ''):
            continue
        runs = client.search_runs(e.experiment_id, order_by=["attributes.start_time DESC"], max_results=10000)
        for r in runs:
            st = r.info.start_time
            if st is None or st < start_ms or st > end_ms:
                continue
            tags = r.data.tags or {}
            updates = {
                'period_start': start_iso[:10],
                'period_end': end_iso[:10],
            }
            for k, v in updates.items():
                try:
                    client.set_tag(r.info.run_id, k, v)
                except Exception:
                    pass

    print('[MLflow Relabel] Done')


if __name__ == '__main__':
    main()