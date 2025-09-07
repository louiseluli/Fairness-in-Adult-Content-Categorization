# -*- coding: utf-8 -*-
"""
theme_manager.py

Purpose:
    Provides a centralized and powerful system for generating dual-theme
    visualizations, driven by a professional-grade configuration file.
    UPGRADED: Now allows for passing custom figure sizes for better control
    over plot layout and readability.
"""

import os
import functools
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- 1. Advanced Configuration Loading ---

def _resolve_paths(config, key, value):
    """Recursively resolves path variables like ${project.root}."""
    if isinstance(value, str) and "${project.root}" in value:
        return value.replace("${project.root}", config['project']['root'])
    if isinstance(value, dict):
        return {k: _resolve_paths(config, k, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(config, i, v) for i, v in enumerate(value)]
    return value

def load_config():
    """
    Loads the master YAML config and resolves all path variables.
    """
    config_path = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return _resolve_paths(config, 'root', config)
    except Exception as e:
        print(f"ERROR: Failed to load or parse configuration file: {e}")
        return None

CONFIG = load_config()

# --- 2. Core Plotting Wrapper ---

def plot_dual_theme(section: str):
    """
    A decorator factory for plotting. Pass the config section name ('eda', 'fairness', etc.).
    """
    def decorator(plot_func):
        @functools.wraps(plot_func)
        def wrapper(*args, **kwargs):
            if CONFIG is None:
                print("Aborting plot generation due to missing configuration.")
                return

            save_path_base = kwargs.get("save_path")
            if not save_path_base:
                raise ValueError("Plotting function must be called with 'save_path'.")

            # Allow custom figure size, with a sensible default
            figsize = kwargs.get("figsize", (10, 8))
            Path(save_path_base).parent.mkdir(parents=True, exist_ok=True)

            for theme in ['light', 'dark']:
                print(f"Generating '{theme}' theme plot for section '{section}'...")

                theme_config = CONFIG['viz']['themes'][theme]
                plt.style.use('seaborn-v0_8-whitegrid' if theme == 'light' else 'seaborn-v0_8-darkgrid')
                plt.rcParams.update({
                    'figure.facecolor': theme_config['facecolor'],
                    'axes.facecolor': theme_config['facecolor'],
                    'axes.labelcolor': theme_config['textcolor'],
                    'axes.edgecolor': theme_config['gridcolor'],
                    'xtick.color': theme_config['textcolor'],
                    'ytick.color': theme_config['textcolor'],
                    'text.color': theme_config['textcolor'],
                    'grid.color': theme_config['gridcolor'],
                    'legend.facecolor': theme_config['facecolor'],
                    'legend.edgecolor': theme_config['gridcolor']
                })

                fig, ax = plt.subplots(figsize=figsize)

                palette = CONFIG['viz']['palettes']['sections'][section][theme]
                
                try:
                    plot_func(ax=ax, palette=palette, *args, **kwargs)
                except Exception as e:
                    print(f"ERROR executing plotting function '{plot_func.__name__}': {e}")
                    plt.close(fig)
                    continue

                ax.title.set_color(theme_config['textcolor'])
                plt.tight_layout()
                
                if CONFIG['viz'].get('save_png', True):
                    final_save_path_png = f"{save_path_base}_{theme}.png"
                    fig.savefig(final_save_path_png, dpi=CONFIG['viz']['dpi'], bbox_inches='tight')
                    print(f"✓ Artefact saved: {Path(final_save_path_png).resolve()}")

                if CONFIG['viz'].get('save_pdf', False):
                    final_save_path_pdf = f"{save_path_base}_{theme}.pdf"
                    fig.savefig(final_save_path_pdf, bbox_inches='tight')
                    print(f"✓ Artefact saved: {Path(final_save_path_pdf).resolve()}")

                plt.close(fig)
        return wrapper
    return decorator