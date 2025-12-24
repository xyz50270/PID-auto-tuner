import pytest
import pandas as pd
from io import StringIO
from src.ingestion import load_and_validate_csv, IngestionError

def test_load_valid_csv():
    csv_data = """timestamp,setpoint,process_variable,output
2023-01-01 10:00:00,50,45,10
2023-01-01 10:00:01,50,46,12
"""
    df = load_and_validate_csv(
        StringIO(csv_data),
        column_map={
            'timestamp': 'Time',
            'setpoint': 'SP',
            'process_variable': 'PV',
            'output': 'OP'
        }
    )
    assert list(df.columns) == ['Time', 'SP', 'PV', 'OP']
    assert len(df) == 2
    assert pd.api.types.is_datetime64_any_dtype(df['Time'])

def test_missing_columns():
    csv_data = """timestamp,setpoint,process_variable
2023-01-01 10:00:00,50,45
"""
    with pytest.raises(IngestionError, match="CSV 中缺少指定的列"):
        load_and_validate_csv(
            StringIO(csv_data),
            column_map={
                'timestamp': 'Time',
                'setpoint': 'SP',
                'process_variable': 'PV',
                'output': 'OP'
            }
        )

def test_empty_file():
    with pytest.raises(IngestionError, match="提供的文件为空"):
        load_and_validate_csv(StringIO(""))