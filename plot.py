"""
Veros Output Analysis Script
Based on: https://veros.readthedocs.io/en/latest/tutorial/analysis.html

This script analyzes Veros ocean model output files using xarray and matplotlib.
Install dependencies: pip install xarray matplotlib netcdf4 cmocean
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

try:
    import cmocean
    HAS_CMOCEAN = True
except ImportError:
    print("Warning: cmocean not installed. Using default colormaps.")
    print("Install with: pip install cmocean")
    HAS_CMOCEAN = False


def find_output_files():
    """Find all NetCDF output files from Veros simulation."""
    patterns = {
        'snapshot': '**/*snapshot*.nc',
        'averages': '**/*average*.nc',
        'overturning': '**/*overturning*.nc',
        'energy': '**/*energy*.nc',
        'other': '**/*.nc'
    }
    
    files = {}
    for key, pattern in patterns.items():
        found = glob.glob(pattern, recursive=True)
        if found:
            files[key] = found
    
    return files


def analyze_snapshot(filepath):
    """Analyze snapshot output file."""
    print(f"\n{'='*70}")
    print(f"ANALYZING SNAPSHOT: {filepath}")
    print('='*70)
    
    ds = xr.open_dataset(filepath)
    print("\nDataset Overview:")
    print(ds)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Snapshot Analysis: {os.path.basename(filepath)}', 
                 fontsize=16, fontweight='bold')
    
    plot_idx = 1
    
    # Plot sea surface temperature (SST) if available
    if 'temp' in ds.data_vars:
        ax = plt.subplot(2, 3, plot_idx)
        plot_idx += 1
        
        # Get surface temperature (last time, surface level)
        sst = ds['temp'].isel(Time=-1, zt=0)
        cmap = 'cmo.thermal' if HAS_CMOCEAN else 'RdYlBu_r'
        sst.plot(ax=ax, cmap=cmap)
        ax.set_title('Sea Surface Temperature (SST)', fontweight='bold')
    
    # Plot sea surface salinity if available
    if 'salt' in ds.data_vars:
        ax = plt.subplot(2, 3, plot_idx)
        plot_idx += 1
        
        sss = ds['salt'].isel(Time=-1, zt=0)
        cmap = 'cmo.haline' if HAS_CMOCEAN else 'viridis'
        sss.plot(ax=ax, cmap=cmap)
        ax.set_title('Sea Surface Salinity (SSS)', fontweight='bold')
    
    # Plot surface velocity if available
    if 'u' in ds.data_vars and 'v' in ds.data_vars:
        ax = plt.subplot(2, 3, plot_idx)
        plot_idx += 1
        
        u_surf = ds['u'].isel(Time=-1, zt=0)
        v_surf = ds['v'].isel(Time=-1, zt=0)
        speed = np.sqrt(u_surf**2 + v_surf**2)
        cmap = 'cmo.speed' if HAS_CMOCEAN else 'plasma'
        speed.plot(ax=ax, cmap=cmap)
        ax.set_title('Surface Current Speed', fontweight='bold')
    
    plt.tight_layout()
    output_file = filepath.replace('.nc', '_snapshot_analysis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    plt.show()
    
    ds.close()


def analyze_averages(filepath):
    """Analyze time-averaged output file."""
    print(f"\n{'='*70}")
    print(f"ANALYZING AVERAGES: {filepath}")
    print('='*70)
    
    ds = xr.open_dataset(filepath)
    print("\nDataset Overview:")
    print(ds)
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Time-Averaged Analysis: {os.path.basename(filepath)}', 
                 fontsize=16, fontweight='bold')
    
    plot_idx = 1
    
    # Plot barotropic stream function (psi)
    if 'psi' in ds.data_vars:
        ax = plt.subplot(2, 3, plot_idx)
        plot_idx += 1
        
        # Convert to Sverdrups (Sv) if in m³/s
        psi = ds['psi'].isel(Time=-1) / 1e6
        cmap = 'cmo.balance' if HAS_CMOCEAN else 'RdBu_r'
        psi.plot.contourf(ax=ax, levels=50, cmap=cmap)
        ax.set_title('Barotropic Stream Function (Sv)', fontweight='bold')
    
    # Plot zonal-mean temperature
    if 'temp' in ds.data_vars:
        ax = plt.subplot(2, 3, plot_idx)
        plot_idx += 1
        
        # Compute zonal mean
        temp_zonal = ds['temp'].isel(Time=-1).mean(dim='xt')
        cmap = 'cmo.thermal' if HAS_CMOCEAN else 'RdYlBu_r'
        temp_zonal.plot.contourf(ax=ax, levels=50, cmap=cmap)
        ax.set_title('Zonal-Mean Temperature', fontweight='bold')
        ax.invert_yaxis()
    
    # Plot zonal-mean salinity
    if 'salt' in ds.data_vars:
        ax = plt.subplot(2, 3, plot_idx)
        plot_idx += 1
        
        # Compute decadal mean of zonal-mean salinity
        # (last 10 time steps if available)
        if len(ds['Time']) >= 10:
            salt_zm = ds['salt'].isel(Time=slice(-10, None)).mean(dim=('Time', 'xt'))
        else:
            salt_zm = ds['salt'].isel(Time=-1).mean(dim='xt')
        
        cmap = 'cmo.haline' if HAS_CMOCEAN else 'viridis'
        salt_zm.plot.contourf(ax=ax, levels=50, cmap=cmap)
        ax.set_title('Zonal-Mean Salinity', fontweight='bold')
        ax.invert_yaxis()
    
    # Plot mixed layer depth if available
    if 'mld' in ds.data_vars:
        ax = plt.subplot(2, 3, plot_idx)
        plot_idx += 1
        
        mld = ds['mld'].isel(Time=-1)
        cmap = 'cmo.deep' if HAS_CMOCEAN else 'viridis_r'
        mld.plot(ax=ax, cmap=cmap)
        ax.set_title('Mixed Layer Depth', fontweight='bold')
    
    plt.tight_layout()
    output_file = filepath.replace('.nc', '_averages_analysis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    plt.show()
    
    ds.close()


def analyze_overturning(filepath):
    """Analyze meridional overturning circulation."""
    print(f"\n{'='*70}")
    print(f"ANALYZING OVERTURNING: {filepath}")
    print('='*70)
    
    ds = xr.open_dataset(filepath)
    print("\nDataset Overview:")
    print(ds)
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Overturning Circulation: {os.path.basename(filepath)}', 
                 fontsize=16, fontweight='bold')
    
    plot_idx = 1
    
    # Plot overturning stream functions
    overturning_vars = [v for v in ds.data_vars if 'vsf' in v or 'bolus' in v or 'trans' in v]
    
    for var_name in overturning_vars[:6]:  # Plot up to 6
        ax = plt.subplot(2, 3, plot_idx)
        plot_idx += 1
        
        var_data = ds[var_name].isel(Time=-1)
        
        # Convert to Sverdrups if needed
        if var_data.max() > 100:
            var_data = var_data / 1e6
            unit = 'Sv'
        else:
            unit = var_data.attrs.get('units', '')
        
        cmap = 'cmo.balance' if HAS_CMOCEAN else 'RdBu_r'
        var_data.plot.contourf(ax=ax, levels=30, cmap=cmap)
        ax.set_title(f'{var_name} ({unit})', fontweight='bold')
        ax.invert_yaxis()
    
    plt.tight_layout()
    output_file = filepath.replace('.nc', '_overturning_analysis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    plt.show()
    
    ds.close()


def analyze_energy(filepath):
    """Analyze energy diagnostics."""
    print(f"\n{'='*70}")
    print(f"ANALYZING ENERGY: {filepath}")
    print('='*70)
    
    ds = xr.open_dataset(filepath)
    print("\nDataset Overview:")
    print(ds)
    
    # Plot time series of energy components
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Energy Diagnostics: {os.path.basename(filepath)}', 
                 fontsize=16, fontweight='bold')
    
    # Find energy-related variables
    energy_vars = [v for v in ds.data_vars if any(
        key in v.lower() for key in ['energy', 'ke', 'pe', 'eke', 'tke']
    )]
    
    for idx, var_name in enumerate(energy_vars[:4]):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        ds[var_name].plot(ax=ax)
        ax.set_title(f'{var_name}', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = filepath.replace('.nc', '_energy_analysis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    plt.show()
    
    ds.close()


def main():
    """Main analysis function."""
    print("="*70)
    print("VEROS OUTPUT ANALYSIS")
    print("="*70)
    
    # Find all output files
    files = find_output_files()
    
    if not files:
        print("\nNo NetCDF files found!")
        print("Make sure you're running this script in the directory")
        print("containing your Veros output files.")
        return
    
    print(f"\nFound output files:")
    for category, file_list in files.items():
        print(f"\n{category.upper()}:")
        for f in file_list:
            print(f"  - {f}")
    
    # Analyze each type of output
    for category, file_list in files.items():
        for filepath in file_list:
            try:
                if 'snapshot' in filepath.lower():
                    analyze_snapshot(filepath)
                elif 'average' in filepath.lower():
                    analyze_averages(filepath)
                elif 'overturning' in filepath.lower():
                    analyze_overturning(filepath)
                elif 'energy' in filepath.lower():
                    analyze_energy(filepath)
                else:
                    # Generic analysis for other files
                    print(f"\n{'='*70}")
                    print(f"GENERIC ANALYSIS: {filepath}")
                    print('='*70)
                    ds = xr.open_dataset(filepath)
                    print(ds)
                    ds.close()
            
            except Exception as e:
                print(f"\n✗ Error analyzing {filepath}: {e}")
                continue
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
