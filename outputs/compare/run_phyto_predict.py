"""Run PhytoRNP_PSLM inference on test set and save predictions."""
import sys, os
sys.path.insert(0, '/home/xiaojian/LLPS_model_project/PhytoRNP_PSLM')
from src.predict import predict_csv

result_df = predict_csv(
    input_path='/home/xiaojian/LLPS_model_project/PhytoRNP_PSLM/data/processed/test.csv',
    output_dir='/home/xiaojian/LLPS_model_project/PhytoRNP_PSLM/outputs/compare',
    seq_col='sequence',
    label=None,
    batch_size=32,
)
print(f'Done. {len(result_df)} predictions saved.')
