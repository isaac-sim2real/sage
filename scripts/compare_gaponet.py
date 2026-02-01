#!/usr/bin/env python3
"""
Advanced GapONet vs Baseline Comparison Script

This script compares simulation results between GapONet and baseline (non-GapONet) modes
using comprehensive metrics including RMSE, MAPE, correlation, cosine similarity, 
time lag analysis, and smoothness metrics.

Usage:
    python compare_gaponet_advanced.py \
        --baseline <baseline_output_folder> \
        --gaponet <gaponet_output_folder> \
        --output <comparison_output_folder>
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.spatial import distance


class SimulationComparator:
    """
    Compare two simulation results (e.g., GapONet vs Baseline) with comprehensive metrics.
    
    This class is adapted from RobotDataComparator but designed for comparing
    two simulation runs rather than sim vs real robot data.
    """
    
    def __init__(self, baseline_folder, gaponet_folder, sample_dt=0.005):
        """
        Initialize the SimulationComparator.
        
        Args:
            baseline_folder: Path to baseline simulation output folder
            gaponet_folder: Path to GapONet simulation output folder
            sample_dt: Time step for resampling data (default: 0.005s = 200Hz)
        """
        self.baseline_folder = Path(baseline_folder)
        self.gaponet_folder = Path(gaponet_folder)
        self.sample_dt = sample_dt
        
        # Load data
        self.baseline_data = self._load_simulation_data(self.baseline_folder)
        self.gaponet_data = self._load_simulation_data(self.gaponet_folder)
        
        # Validate data
        self._validate_data()
        
    def _load_simulation_data(self, folder):
        """Load simulation data from folder"""
        data = {}
        
        # Load joint list
        joint_list_file = folder / "joint_list.txt"
        if joint_list_file.exists():
            with open(joint_list_file) as f:
                data['joints'] = [line.strip().split("/")[-1] for line in f.readlines()]
        else:
            raise FileNotFoundError(f"Joint list not found: {joint_list_file}")
        
        # Load CSV files
        for data_type in ['control', 'state_motor']:
            csv_file = folder / f"{data_type}.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                # Process time
                df['time_since_zero'] = df['timestamp'] - df['timestamp'].iloc[0]
                data[data_type] = df
            else:
                raise FileNotFoundError(f"Data file not found: {csv_file}")
        
        # Process DOF data
        data['dof_state'] = self._process_dof_data(
            data['state_motor'], 
            data['joints'], 
            ['positions', 'velocities', 'torques']
        )
        
        return data
    
    def _process_dof_data(self, df, joint_list, keys):
        """Process DOF data from string format to columns"""
        df_copy = df.copy()
        
        for key in keys:
            # Parse string representation of lists
            df_copy[key] = df_copy[key].apply(eval)
            
            # Create column for each joint
            for i, joint in enumerate(joint_list):
                df_copy[f"{key}_{joint}"] = df_copy[key].apply(lambda x: x[i])
            
            # Drop original list column
            df_copy = df_copy.drop(key, axis=1)
        
        return df_copy
    
    def _validate_data(self):
        """Validate that baseline and gaponet data have matching joints"""
        baseline_joints = set(self.baseline_data['joints'])
        gaponet_joints = set(self.gaponet_data['joints'])
        
        if baseline_joints != gaponet_joints:
            raise ValueError(
                f"Joint mismatch!\n"
                f"Baseline: {baseline_joints}\n"
                f"GapONet: {gaponet_joints}"
            )
        
        print(f"✓ Validated {len(self.baseline_data['joints'])} joints")
    
    def _resample_waveform(self, df, key, timestamp):
        """Resample a waveform to a new timestamp array using linear interpolation"""
        return pd.DataFrame({
            'timestamp': timestamp,
            'value': np.interp(timestamp, df['time_since_zero'], df[key])
        })
    
    def align_data(self, joint_name, data_type='positions'):
        """
        Align baseline and gaponet data for comparison.
        
        Args:
            joint_name: Name of the joint
            data_type: Type of data ('positions', 'velocities', 'torques')
        
        Returns:
            tuple: (aligned_baseline, aligned_gaponet) DataFrames
        """
        baseline_df = self.baseline_data['dof_state']
        gaponet_df = self.gaponet_data['dof_state']
        
        key = f"{data_type}_{joint_name}"
        
        # Create uniform timestamp array for resampling
        max_time = min(
            baseline_df['time_since_zero'].max(),
            gaponet_df['time_since_zero'].max()
        )
        timestamp = np.arange(self.sample_dt, max_time, self.sample_dt)
        
        # Resample both waveforms
        aligned_baseline = self._resample_waveform(baseline_df, key, timestamp)
        aligned_gaponet = self._resample_waveform(gaponet_df, key, timestamp)
        
        return aligned_baseline, aligned_gaponet
    
    def calculate_metrics(self, joint_name, data_type='positions'):
        """
        Calculate comprehensive comparison metrics for a single joint.
        
        Args:
            joint_name: Name of the joint
            data_type: Type of data ('positions', 'velocities', 'torques')
        
        Returns:
            dict: Dictionary of calculated metrics
        """
        # Align data
        aligned_baseline, aligned_gaponet = self.align_data(joint_name, data_type)
        
        metrics = {}
        
        # 1. RMSE (Root Mean Squared Error)
        mse = np.mean((aligned_baseline['value'] - aligned_gaponet['value']) ** 2)
        metrics['rmse'] = np.sqrt(mse)
        
        # 2. MAPE (Mean Absolute Percentage Error)
        with np.errstate(divide='ignore', invalid='ignore'):
            # Filter out baseline values with absolute value less than 0.1
            valid_indices = np.abs(aligned_baseline['value']) >= 0.1
            
            if valid_indices.sum() == 0:
                metrics['mape'] = np.nan
            else:
                filtered_baseline = aligned_baseline[valid_indices]
                filtered_gaponet = aligned_gaponet[valid_indices]
                
                abs_percentage_errors = np.abs(
                    (filtered_gaponet['value'] - filtered_baseline['value']) / 
                    filtered_baseline['value']
                )
                valid_errors = abs_percentage_errors[
                    ~np.isinf(abs_percentage_errors) & ~np.isnan(abs_percentage_errors)
                ]
                
                if len(valid_errors) > 0:
                    metrics['mape'] = np.mean(valid_errors) * 100  # as percentage
                else:
                    metrics['mape'] = np.nan
        
        # 3. Correlation coefficient
        correlation = np.corrcoef(
            aligned_baseline['value'], 
            aligned_gaponet['value']
        )[0, 1]
        metrics['correlation'] = correlation
        
        # 4. Cosine similarity
        similarity = 1 - distance.cosine(
            aligned_baseline['value'], 
            aligned_gaponet['value']
        )
        metrics['cosine_similarity'] = similarity
        
        # 5. Cross-correlation for lag analysis
        corr = signal.correlate(
            aligned_baseline['value'], 
            aligned_gaponet['value'], 
            method='fft'
        )
        corr /= np.max(corr)
        
        max_corr_idx = np.argmax(np.abs(corr))
        max_lag = (max_corr_idx - (len(aligned_baseline) - 1)) * self.sample_dt
        metrics['max_lag_seconds'] = max_lag
        
        # 6. Smoothness metrics (for velocities and torques)
        if data_type in ['velocities', 'torques']:
            baseline_diff = np.diff(aligned_baseline['value']) / self.sample_dt
            gaponet_diff = np.diff(aligned_gaponet['value']) / self.sample_dt
            
            metrics['baseline_smoothness_mean'] = np.mean(np.abs(baseline_diff))
            metrics['baseline_smoothness_std'] = np.std(baseline_diff)
            metrics['gaponet_smoothness_mean'] = np.mean(np.abs(gaponet_diff))
            metrics['gaponet_smoothness_std'] = np.std(gaponet_diff)
        
        # 7. Max absolute error
        metrics['max_abs_error'] = np.max(np.abs(
            aligned_baseline['value'] - aligned_gaponet['value']
        ))
        
        # 8. Mean absolute error
        metrics['mae'] = np.mean(np.abs(
            aligned_baseline['value'] - aligned_gaponet['value']
        ))
        
        # 9. RMS values (for torque comparison)
        if data_type == 'torques':
            metrics['baseline_rms'] = np.sqrt(np.mean(aligned_baseline['value'] ** 2))
            metrics['gaponet_rms'] = np.sqrt(np.mean(aligned_gaponet['value'] ** 2))
            if metrics['baseline_rms'] > 0:
                metrics['rms_change_percent'] = (
                    (metrics['gaponet_rms'] - metrics['baseline_rms']) / 
                    metrics['baseline_rms'] * 100
                )
        
        return metrics
    
    def analyze_all_joints(self, output_folder):
        """
        Analyze all joints and generate comprehensive comparison report.
        
        Args:
            output_folder: Folder to save analysis results
        """
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        joints = self.baseline_data['joints']
        data_types = ['positions', 'velocities', 'torques']
        
        # Storage for all metrics
        all_metrics = {dt: {} for dt in data_types}
        
        print(f"\n📊 Analyzing {len(joints)} joints × {len(data_types)} data types...")
        
        for joint in joints:
            print(f"  Processing: {joint}")
            for data_type in data_types:
                try:
                    metrics = self.calculate_metrics(joint, data_type)
                    all_metrics[data_type][joint] = metrics
                except Exception as e:
                    print(f"    ⚠️  Error with {joint} ({data_type}): {e}")
        
        # Save metrics to JSON
        metrics_file = output_path / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\n✓ Saved metrics to: {metrics_file}")
        
        # Create summary statistics
        self._create_summary_report(all_metrics, output_path)
        
        # Create visualizations
        self._create_visualizations(all_metrics, output_path)
        
        return all_metrics
    
    def _create_summary_report(self, all_metrics, output_path):
        """Create a summary report with key statistics"""
        summary = {}
        
        for data_type in ['positions', 'velocities', 'torques']:
            metrics_list = all_metrics[data_type]
            
            if not metrics_list:
                continue
            
            # Aggregate metrics across all joints
            rmse_values = [m['rmse'] for m in metrics_list.values()]
            mae_values = [m['mae'] for m in metrics_list.values()]
            corr_values = [m['correlation'] for m in metrics_list.values() if not np.isnan(m['correlation'])]
            
            summary[data_type] = {
                'rmse': {
                    'mean': float(np.mean(rmse_values)),
                    'std': float(np.std(rmse_values)),
                    'min': float(np.min(rmse_values)),
                    'max': float(np.max(rmse_values))
                },
                'mae': {
                    'mean': float(np.mean(mae_values)),
                    'std': float(np.std(mae_values)),
                    'min': float(np.min(mae_values)),
                    'max': float(np.max(mae_values))
                },
                'correlation': {
                    'mean': float(np.mean(corr_values)) if corr_values else None,
                    'min': float(np.min(corr_values)) if corr_values else None
                }
            }
            
            # Torque-specific metrics
            if data_type == 'torques':
                baseline_rms = [m['baseline_rms'] for m in metrics_list.values()]
                gaponet_rms = [m['gaponet_rms'] for m in metrics_list.values()]
                
                summary[data_type]['torque_rms'] = {
                    'baseline_mean': float(np.mean(baseline_rms)),
                    'gaponet_mean': float(np.mean(gaponet_rms)),
                    'change_percent': float(
                        (np.mean(gaponet_rms) - np.mean(baseline_rms)) / 
                        np.mean(baseline_rms) * 100
                    )
                }
        
        # Save summary
        summary_file = output_path / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary to console
        print("\n" + "="*60)
        print("📈 SUMMARY REPORT: GapONet vs Baseline")
        print("="*60)
        
        for data_type, stats in summary.items():
            print(f"\n{data_type.upper()}:")
            print(f"  RMSE: {stats['rmse']['mean']:.6f} ± {stats['rmse']['std']:.6f}")
            print(f"  MAE:  {stats['mae']['mean']:.6f} ± {stats['mae']['std']:.6f}")
            if stats['correlation']['mean'] is not None:
                print(f"  Correlation: {stats['correlation']['mean']:.4f}")
            
            if 'torque_rms' in stats:
                print(f"  Torque RMS Change: {stats['torque_rms']['change_percent']:+.2f}%")
        
        print("="*60 + "\n")
        
        return summary
    
    def _create_visualizations(self, all_metrics, output_path):
        """Create comparison visualizations"""
        joints = list(self.baseline_data['joints'])
        
        # 1. Create per-joint RMSE comparison
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        data_types = ['positions', 'velocities', 'torques']
        
        for i, data_type in enumerate(data_types):
            rmse_values = [
                all_metrics[data_type][joint]['rmse'] 
                for joint in joints
            ]
            
            axes[i].bar(range(len(joints)), rmse_values, color='steelblue', alpha=0.7)
            axes[i].set_ylabel(f'RMSE ({data_type})')
            axes[i].set_title(f'{data_type.capitalize()} RMSE per Joint')
            axes[i].grid(True, alpha=0.3)
            
            if i == 2:  # Last subplot
                axes[i].set_xticks(range(len(joints)))
                axes[i].set_xticklabels(joints, rotation=45, ha='right')
            else:
                axes[i].set_xticks([])
        
        plt.tight_layout()
        plt.savefig(output_path / 'rmse_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path / 'rmse_comparison.png'}")
        
        # 2. Create time series comparison for first joint (example)
        example_joint = joints[0]
        self._plot_time_series_comparison(example_joint, output_path)
    
    def _plot_time_series_comparison(self, joint_name, output_path):
        """Plot time series comparison for a specific joint"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        data_types = ['positions', 'velocities', 'torques']
        labels = ['Position (rad)', 'Velocity (rad/s)', 'Torque (Nm)']
        
        for i, (data_type, label) in enumerate(zip(data_types, labels)):
            aligned_baseline, aligned_gaponet = self.align_data(joint_name, data_type)
            
            axes[i].plot(
                aligned_baseline['timestamp'], 
                aligned_baseline['value'], 
                'b-', 
                label='Baseline', 
                linewidth=1.5, 
                alpha=0.7
            )
            axes[i].plot(
                aligned_gaponet['timestamp'], 
                aligned_gaponet['value'], 
                'r--', 
                label='GapONet', 
                linewidth=1.5, 
                alpha=0.7
            )
            
            axes[i].set_ylabel(label)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            if i == 2:
                axes[i].set_xlabel('Time (s)')
        
        fig.suptitle(f'Time Series Comparison: {joint_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = output_path / f'timeseries_{joint_name}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Advanced comparison of GapONet vs Baseline simulation results'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        required=True,
        help='Path to baseline simulation output folder'
    )
    parser.add_argument(
        '--gaponet',
        type=str,
        required=True,
        help='Path to GapONet simulation output folder'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='comparison_results',
        help='Output folder for comparison results (default: comparison_results)'
    )
    parser.add_argument(
        '--sample-freq',
        type=int,
        default=200,
        help='Sampling frequency in Hz for data alignment (default: 200)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 Advanced GapONet vs Baseline Comparison")
    print("="*60)
    print(f"Baseline: {args.baseline}")
    print(f"GapONet:  {args.gaponet}")
    print(f"Output:   {args.output}")
    print(f"Sample frequency: {args.sample_freq} Hz")
    print("="*60 + "\n")
    
    # Create comparator
    comparator = SimulationComparator(
        baseline_folder=args.baseline,
        gaponet_folder=args.gaponet,
        sample_dt=1.0 / args.sample_freq
    )
    
    # Run analysis
    metrics = comparator.analyze_all_joints(args.output)
    
    print("\n✅ Comparison complete!")
    print(f"📁 Results saved to: {args.output}/")


if __name__ == '__main__':
    main()

