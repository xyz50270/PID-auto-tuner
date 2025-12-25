import pandas as pd
from typing import Dict, IO, Union
import os

class IngestionError(Exception):
    """Custom exception for data ingestion errors."""
    pass

def load_and_validate_data(
    file_buffer: Union[str, IO],
    filename: str = "",
    column_map: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Load CSV or Excel data and validate required columns.
    """
    try:
        # Determine file type
        ext = os.path.splitext(filename)[1].lower() if filename else ""
        
        if ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_buffer)
        else:
            # Default to CSV
            df = pd.read_csv(file_buffer)
            
    except Exception as e:
        raise IngestionError(f"文件读取失败: {str(e)}")

    if df.empty:
        raise IngestionError("提供的文件为空。")

    if column_map:
        # Check if mapped columns exist
        missing_source_cols = [col for col in column_map.keys() if col not in df.columns]
        if missing_source_cols:
             raise IngestionError(f"文件中缺少指定的列: {missing_source_cols}")
        
        df = df.rename(columns=column_map)
    
    required_cols = ['Time', 'SP', 'PV', 'OP']
    missing_required = [col for col in required_cols if col not in df.columns]
    
    if missing_required:
        raise IngestionError(f"映射后缺少必要的列: {missing_required}。需要: {required_cols}")

    # Ensure Time is datetime with high precision
    try:
        df['Time'] = pd.to_datetime(df['Time'])
    except Exception:
         raise IngestionError("无法将 'Time' 列转换为日期时间格式。请确保时间列包含完整的日期和时间（包含秒）。")
         
    # Ensure numeric types
    for col in ['SP', 'PV', 'OP']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaNs in critical columns
    original_len = len(df)
    df = df.dropna(subset=['Time', 'SP', 'PV', 'OP'])
    if len(df) < original_len:
        print(f"警告: 由于缺失数据删除了 {original_len - len(df)} 行。")
        
    if df.empty:
        raise IngestionError("清洗后没有剩余的有效数据行。")

    return df.sort_values('Time').reset_index(drop=True)
