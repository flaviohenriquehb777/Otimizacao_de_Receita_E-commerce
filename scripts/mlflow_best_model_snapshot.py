import os
import json
from pathlib import Path
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient
import dagshub


def _to_epoch_ms(iso: str) -> int:
    dt = datetime.fromisoformat(iso.replace('Z', '+00:00'))
    return int(dt.timestamp() * 1000)


def main():
    dataset_path = Path('dados/dataset_cafeterias_rj.xlsx')
    print('[CHECK] Dataset existe?', dataset_path.exists(), '->', dataset_path)

    repo = os.environ.get('DAGSHUB_REPO')
    owner = os.environ.get('DAGSHUB_OWNER')
    tracking_uri_env = os.environ.get('MLFLOW_TRACKING_URI')
    if repo and owner:
        dagshub.init(repo, owner, mlflow=True)
    if tracking_uri_env:
        mlflow.set_tracking_uri(tracking_uri_env)

    mlflow.set_experiment('best_model')

    snap_path = Path('models/metrics_snapshot.json')
    if not snap_path.exists():
        raise FileNotFoundError(f'Arquivo não encontrado: {snap_path}')

    with snap_path.open('r', encoding='utf-8') as f:
        snap = json.load(f)

    best_name = snap.get('best_model_name') or snap.get('best_model') or 'unknown'
    metrics = snap.get('metrics') or {}
    features = snap.get('feature_order') or []

    params = {
        'best_model': best_name,
        'feature_count': len(features),
        'period_start': '2022-01-01',
        'period_end': '2022-06-30',
    }

    backdate_iso = os.environ.get('BACKDATE_RUN_ISO') or '2022-06-30T12:00:00Z'
    backdate = os.environ.get('BACKDATE_RUN', '1') not in ('0', 'false', 'False')

    if backdate:
        client = MlflowClient()
        exp = client.get_experiment_by_name('best_model')
        if not exp:
            exp_id = mlflow.create_experiment('best_model')
        else:
            exp_id = exp.experiment_id
        start_ms = _to_epoch_ms(backdate_iso)
        run = client.create_run(exp_id, start_time=start_ms, tags={'mlflow.runName': 'best_model', 'is_best': 'true', 'best_model': best_name})
        mlflow.start_run(run_id=run.info.run_id)
        mlflow.log_params(params)
        if isinstance(metrics, dict):
            flat_metrics = {}
            for k, v in metrics.items():
                try:
                    if isinstance(v, (int, float)):
                        flat_metrics[k] = float(v)
                except Exception:
                    pass
            if flat_metrics:
                mlflow.log_metrics(flat_metrics)
        mlflow.log_artifact(str(snap_path), artifact_path='snapshot')
        mlflow.end_run()
        end_ms = _to_epoch_ms(backdate_iso)
        client.set_terminated(run.info.run_id, status='FINISHED', end_time=end_ms)
        print('[MLflow] Run ID:', run.info.run_id)
    else:
        with mlflow.start_run(run_name='best_model') as run:
            mlflow.log_params(params)
            if isinstance(metrics, dict):
                flat_metrics = {}
                for k, v in metrics.items():
                    try:
                        if isinstance(v, (int, float)):
                            flat_metrics[k] = float(v)
                    except Exception:
                        pass
                if flat_metrics:
                    mlflow.log_metrics(flat_metrics)
            mlflow.set_tag('is_best', 'true')
            mlflow.set_tag('best_model', best_name)
            mlflow.set_tag('mlflow.runName', 'best_model')
            mlflow.log_artifact(str(snap_path), artifact_path='snapshot')
            print('[MLflow] Run ID:', run.info.run_id)

    print('[MLflow] Tracking URI:', mlflow.get_tracking_uri())
    print('[MLflow] Experimento criado/selecionado: best_model')


if __name__ == '__main__':
    main()