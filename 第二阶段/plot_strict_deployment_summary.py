"""Create reproducible summary figures for the verified 2014-2019 strict deployment result."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / 'result_p1deploy_ft_strict_20260817' / 'analysis_2014_2019'
ANNUAL = RESULT / 'annual_metrics_all_stations_2014_2019.csv'
AUDIT = RESULT / 'p1_match_audit_summary_2014_2019.csv'


def require_columns(frame, columns, source):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f'{source} missing columns: {sorted(missing)}')


def make_annual_rmse(annual):
    models = {
        'clim_surf_p1': ('Climate + P1 surface replacement', '#0072B2'),
        'clim_adj_p1': ('Climate + P1 full-profile adjustment', '#E69F00'),
        'gpt3': ('GPT3 baseline', '#D55E00'),
    }
    fig, ax = plt.subplots(figsize=(8.5, 5.1), constrained_layout=True)
    for model, (label, color) in models.items():
        subset = annual.loc[annual['model'].eq(model)].sort_values('year')
        ax.plot(subset['year'], subset['RMSE'], marker='o', linewidth=2.2,
                markersize=6, label=label, color=color)
    ax.set_title('Strict replay stacking evaluation: annual PWV RMSE')
    ax.set_xlabel('Year')
    ax.set_ylabel('RMSE (mm)')
    ax.set_xticks(sorted(annual['year'].unique()))
    ax.set_ylim(bottom=0)
    ax.grid(True, axis='y', alpha=0.28)
    ax.legend(frameon=False, loc='upper left')
    fig.savefig(RESULT / 'strict_annual_rmse_2014_2019.png', dpi=220)
    plt.close(fig)


def make_coverage(audit):
    columns = ['year', 'summary_station_count_attempted', 'summary_records_written']
    coverage = audit.loc[:, columns].copy().sort_values('year')
    if coverage['year'].duplicated().any():
        raise ValueError('audit summary contains duplicate yearly rows')
    fig, ax_left = plt.subplots(figsize=(8.5, 5.1), constrained_layout=True)
    bars = ax_left.bar(coverage['year'].astype(str), coverage['summary_records_written'],
                       color='#56B4E9', label='Common samples')
    ax_left.set_title('Strict replay stacking evaluation: usable yearly coverage')
    ax_left.set_xlabel('Year')
    ax_left.set_ylabel('Common samples')
    ax_left.grid(True, axis='y', alpha=0.28)
    for bar, value in zip(bars, coverage['summary_records_written']):
        ax_left.annotate(f'{value:,}', (bar.get_x() + bar.get_width() / 2, value),
                         xytext=(0, 4), textcoords='offset points', ha='center',
                         va='bottom', fontsize=8)
    ax_right = ax_left.twinx()
    ax_right.plot(coverage['year'].astype(str), coverage['summary_station_count_attempted'],
                  color='#009E73', marker='o', linewidth=2.2, label='Usable stations')
    ax_right.set_ylabel('Usable stations')
    ax_right.set_ylim(0, 36)
    ax_right.set_yticks(range(0, 37, 6))
    fig.legend(loc='upper right', bbox_to_anchor=(0.89, 0.86), frameon=False)
    fig.savefig(RESULT / 'strict_coverage_2014_2019.png', dpi=220)
    plt.close(fig)


def main():
    annual = pd.read_csv(ANNUAL)
    audit = pd.read_csv(AUDIT)
    require_columns(annual, ['model', 'year', 'RMSE'], ANNUAL)
    require_columns(audit, ['year', 'summary_station_count_attempted', 'summary_records_written'], AUDIT)
    if sorted(annual['year'].unique().tolist()) != list(range(2014, 2020)):
        raise ValueError('annual metrics must contain exactly 2014-2019')
    if sorted(audit['year'].unique().tolist()) != list(range(2014, 2020)):
        raise ValueError('audit summary must contain exactly 2014-2019')
    make_annual_rmse(annual)
    make_coverage(audit)
    print('PLOTS_OK')


if __name__ == '__main__':
    main()

