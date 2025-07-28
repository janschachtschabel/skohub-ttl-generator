"""
Data combination module for SkoHub TTL Generator.
Handles combining multiple datasets based on different scenarios and strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
from core.multi_file_handler import ScenarioType, FileRole, JoinStrategy


class DataCombiner:
    """Combines multiple datasets based on scenario configuration."""
    
    def __init__(self):
        self.join_strategy = JoinStrategy()
        self.max_memory_usage = 500_000_000  # 500MB limit for large datasets
    
    def combine_datasets(self, datasets: List[Dict], scenario_type: ScenarioType, 
                        join_config: Dict) -> pd.DataFrame:
        """Main method to combine datasets based on scenario type"""
        if not datasets:
            return pd.DataFrame()
        
        if len(datasets) == 1:
            return datasets[0]['data'].copy()
        
        # Check memory usage and use appropriate strategy
        total_memory = sum(self._estimate_memory_usage(d['data']) for d in datasets)
        
        if total_memory > self.max_memory_usage:
            st.warning("⚠️ Large dataset detected. Using memory-efficient processing.")
            return self._combine_large_datasets(datasets, scenario_type, join_config)
        else:
            return self._combine_standard_datasets(datasets, scenario_type, join_config)
    
    def _combine_standard_datasets(self, datasets: List[Dict], scenario_type: ScenarioType,
                                  join_config: Dict) -> pd.DataFrame:
        """Standard dataset combination for normal-sized data"""
        if scenario_type == ScenarioType.DISTRIBUTED_DATA:
            return self.join_strategy.distributed_join(datasets, join_config)
        else:  # COMPLETE_RECORDS
            return self.join_strategy.complete_records_concat(datasets, join_config)
    
    def _combine_large_datasets(self, datasets: List[Dict], scenario_type: ScenarioType,
                               join_config: Dict) -> pd.DataFrame:
        """Memory-efficient combination for large datasets"""
        if scenario_type == ScenarioType.DISTRIBUTED_DATA:
            return self._chunked_distributed_join(datasets, join_config)
        else:  # COMPLETE_RECORDS
            return self._chunked_concat(datasets, join_config)
    
    def _chunked_distributed_join(self, datasets: List[Dict], join_config: Dict) -> pd.DataFrame:
        """Memory-efficient distributed join using chunking"""
        if not datasets:
            return pd.DataFrame()
        
        join_field = join_config.get('join_field', 'id')
        how = join_config.get('how', 'outer')
        chunk_size = 10000
        
        # Start with the primary dataset (usually the first one)
        primary_dataset = datasets[0]
        result_chunks = []
        
        # Process primary dataset in chunks
        primary_df = primary_dataset['data']
        
        for i in range(0, len(primary_df), chunk_size):
            chunk = primary_df.iloc[i:i+chunk_size].copy()
            
            # Join with secondary datasets
            for secondary_dataset in datasets[1:]:
                secondary_df = secondary_dataset['data']
                
                if join_field in chunk.columns and join_field in secondary_df.columns:
                    # Get relevant secondary data for this chunk
                    chunk_join_values = chunk[join_field].unique()
                    relevant_secondary = secondary_df[
                        secondary_df[join_field].isin(chunk_join_values)
                    ]
                    
                    if not relevant_secondary.empty:
                        chunk = pd.merge(
                            chunk,
                            relevant_secondary,
                            on=join_field,
                            how=how,
                            suffixes=('', f'_{secondary_dataset["filename"].split(".")[0]}')
                        )
            
            result_chunks.append(chunk)
            
            # Show progress for large datasets
            if i % (chunk_size * 10) == 0:
                progress = min(100, int((i / len(primary_df)) * 100))
                st.progress(progress / 100)
        
        return pd.concat(result_chunks, ignore_index=True)
    
    def _chunked_concat(self, datasets: List[Dict], join_config: Dict) -> pd.DataFrame:
        """Memory-efficient concatenation using chunking"""
        chunk_size = 10000
        result_chunks = []
        
        for dataset in datasets:
            df = dataset['data']
            filename = dataset['filename']
            
            # Add source information if requested
            if join_config.get('add_source_info', True):
                df = df.copy()
                df['_source_file'] = filename
            
            # Process in chunks
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                result_chunks.append(chunk)
        
        # Concatenate all chunks
        combined_df = pd.concat(result_chunks, ignore_index=True, sort=False)
        
        # Remove duplicates if requested
        if join_config.get('remove_duplicates', True):
            combined_df = self._remove_duplicates_efficiently(combined_df)
        
        return combined_df
    
    def _estimate_memory_usage(self, df: pd.DataFrame) -> int:
        """Estimate memory usage of a DataFrame"""
        return df.memory_usage(deep=True).sum()
    
    def _remove_duplicates_efficiently(self, df: pd.DataFrame) -> pd.DataFrame:
        """Efficiently remove duplicates from large datasets"""
        # Try different duplicate removal strategies based on available columns
        dedupe_fields = ['uri', 'id', 'conceptUri', 'identifier', 'uuid']
        dedupe_field = None
        
        for field in dedupe_fields:
            if field in df.columns:
                dedupe_field = field
                break
        
        if dedupe_field:
            return df.drop_duplicates(subset=[dedupe_field], keep='first')
        else:
            # For large datasets without clear ID fields, use hash-based deduplication
            if len(df) > 50000:
                return self._hash_based_deduplication(df)
            else:
                return df.drop_duplicates()
    
    def _hash_based_deduplication(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hash-based deduplication for very large datasets"""
        chunk_size = 10000
        unique_chunks = []
        seen_hashes = set()
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            
            # Create hash of each row for duplicate detection
            chunk_hashes = chunk.apply(
                lambda x: hash(tuple(x.astype(str).fillna(''))), axis=1
            )
            
            # Keep only rows with unseen hashes
            mask = ~chunk_hashes.isin(seen_hashes)
            unique_chunk = chunk[mask]
            
            if not unique_chunk.empty:
                unique_chunks.append(unique_chunk)
                seen_hashes.update(chunk_hashes[mask])
        
        return pd.concat(unique_chunks, ignore_index=True) if unique_chunks else df
    
    def validate_join_compatibility(self, datasets: List[Dict], join_config: Dict) -> Dict[str, Any]:
        """Validate that datasets can be joined with the given configuration"""
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        join_field = join_config.get('join_field')
        
        if not join_field:
            validation_result['errors'].append("No join field specified")
            validation_result['is_valid'] = False
            return validation_result
        
        # Check if join field exists in all datasets
        missing_join_field = []
        join_field_stats = {}
        
        for dataset in datasets:
            df = dataset['data']
            filename = dataset['filename']
            
            if join_field not in df.columns:
                missing_join_field.append(filename)
            else:
                # Collect statistics about the join field
                join_values = df[join_field].dropna()
                join_field_stats[filename] = {
                    'total_values': len(df),
                    'non_null_values': len(join_values),
                    'unique_values': len(join_values.unique()),
                    'null_percentage': ((len(df) - len(join_values)) / len(df)) * 100
                }
        
        if missing_join_field:
            validation_result['errors'].append(
                f"Join field '{join_field}' missing in files: {', '.join(missing_join_field)}"
            )
            validation_result['is_valid'] = False
        
        validation_result['statistics']['join_field_stats'] = join_field_stats
        
        # Check for potential join issues
        if len(datasets) >= 2 and join_field in datasets[0]['data'].columns:
            primary_values = set(datasets[0]['data'][join_field].dropna().astype(str))
            
            for i, dataset in enumerate(datasets[1:], 1):
                if join_field in dataset['data'].columns:
                    secondary_values = set(dataset['data'][join_field].dropna().astype(str))
                    
                    # Calculate overlap
                    overlap = len(primary_values.intersection(secondary_values))
                    overlap_percentage = (overlap / len(primary_values)) * 100 if primary_values else 0
                    
                    if overlap_percentage < 10:
                        validation_result['warnings'].append(
                            f"Low overlap ({overlap_percentage:.1f}%) between primary dataset and {dataset['filename']}"
                        )
                    
                    validation_result['statistics'][f'overlap_with_{dataset["filename"]}'] = {
                        'overlap_count': overlap,
                        'overlap_percentage': overlap_percentage
                    }
        
        return validation_result
    
    def preview_join_result(self, datasets: List[Dict], join_config: Dict, 
                           sample_size: int = 100) -> pd.DataFrame:
        """Generate a preview of the join result using a sample of the data"""
        if not datasets:
            return pd.DataFrame()
        
        # Create sample datasets
        sample_datasets = []
        for dataset in datasets:
            df = dataset['data']
            sample_df = df.head(sample_size) if len(df) > sample_size else df.copy()
            sample_datasets.append({
                **dataset,
                'data': sample_df
            })
        
        # Perform the join on sample data
        scenario_type = ScenarioType.DISTRIBUTED_DATA if len(datasets) > 1 else ScenarioType.COMPLETE_RECORDS
        return self._combine_standard_datasets(sample_datasets, scenario_type, join_config)
    
    def get_combination_statistics(self, original_datasets: List[Dict], 
                                  combined_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistics about the data combination process"""
        stats = {
            'input_files': len(original_datasets),
            'input_total_rows': sum(len(d['data']) for d in original_datasets),
            'output_rows': len(combined_df),
            'columns_added': 0,
            'data_loss_percentage': 0,
            'file_contributions': {}
        }
        
        # Calculate data loss/gain
        if stats['input_total_rows'] > 0:
            stats['data_loss_percentage'] = (
                (stats['input_total_rows'] - stats['output_rows']) / stats['input_total_rows']
            ) * 100
        
        # Count columns from each file
        original_columns = set()
        for dataset in original_datasets:
            file_columns = set(dataset['data'].columns)
            original_columns.update(file_columns)
            stats['file_contributions'][dataset['filename']] = {
                'rows': len(dataset['data']),
                'columns': len(file_columns),
                'column_names': list(file_columns)
            }
        
        stats['columns_added'] = len(combined_df.columns) - len(original_columns)
        
        return stats
