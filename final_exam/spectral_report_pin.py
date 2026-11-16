"""Portable Excel/CSV tables and bounded matplotlib figures for spectral probes."""
from collections import defaultdict
import csv
import html
import json
from pathlib import Path

from device_jepa.excel import Cell, Sheet, _column, _write_workbook


NOTES = [
    ('任务', '只读诊断：未重新训练、未改 checkpoint、未测压缩模型的 MAE/RMSE/R²。'),
    ('参数统计', '每个可部署模型分别统计，包含冻结 OP 预测器；跨模型相加会重复计算共享预测器。偏置/归一化/嵌入计入总参数，但不全部参与矩阵分解。'),
    ('矩阵方向', '统一 y=W x；nn.Linear 为 [out,in]；MoE 每个 expert 单独转成 [out,in]；MHA 的 Q/K/V 分开分析。'),
    ('秩指标', 'rank_95/99/999：平方奇异值能量达到阈值的最小秩；stable_rank=||W||F²/σmax²；effective_rank_energy=exp(平方奇异值概率熵)。'),
    ('数值秩', 'numerical_rank 使用原权重 dtype 的 eps × max(m,n) × σmax；零矩阵没有定义良好的主子空间，重合度留空。'),
    ('相对误差', 'relative_weight_error=||W-Wr||F/||W||F；relative_layer_output_error=||(W-Wr)X||F/||WX||F。它们不是 MSE 或性能预测误差。'),
    ('激活校准', '只用保存的 ID train 样本、原预处理统计；不读真实性能或真实 OP 作模型输入，不重新拟合归一化。排除 padding、保留 graph/真实 token。'),
    ('校准下界', 'optimal_calibration_output_error 是同一校准子集上 WX 的最优 rank-r 截断误差；未验证泛化，也未模拟多层同时压缩。'),
    ('缺失激活', 'MHA 内部 functional projection 等未捕获位置留空；不能把留空理解为 0 误差。N_SAMPLES=0 时完全不加载数据。'),
    ('子空间比较', 'SubspaceOverlap 使用保存的原始坐标，未对齐隐藏通道/注意力头；低重合不能排除排列对齐后可共享，高重合不证明功能相同。'),
    ('共享基底', '同模型、同 seed、同架构、同矩阵形状才比较。各电路矩阵先按 Frobenius 范数归一化，再用横/纵拼接 SVD 求 U,V，S_c=UᵀW_cV。'),
    ('共享误差', '是联合行/列子空间初始化的误差，未优化 Tucker 分解、未蒸馏/微调、未神经元对齐；不是最优共享压缩能力。'),
    ('预算', '独立低秩 r(m+n)；共享 C 个矩阵 r(m+n)+Cr²。预算仅针对该矩阵权重，不含偏置或模型其他部分；saving<0 表示反而增参。'),
    ('OOD 边界', '共享基底用全部列出的电路权重拟合。它们都是拟合电路；不能把其中某个再称作未见 OOD 电路。'),
    ('图表覆盖', 'CSV/Excel 保存所有矩阵；图只显示按参数量排序的前 MAX_PLOTS 个模型/seed/矩阵组。完整图索引见 plots_pin/manifest_pin.json。'),
]


def require_plotting_pin():
    try:
        import matplotlib
    except ImportError as exc:
        raise ImportError('Install matplotlib in the job environment: python -m pip install matplotlib; or set MAX_PLOTS=0') from exc
    matplotlib.use('Agg')


def sheet_pin(name, rows, note):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    sheet = Sheet(name, [75 if k in ('matrix', 'path', 'reason', 'config', 'model_config') else
                        38 if k in ('model', 'circuit', 'circuit_names', 'fit_circuits') else 25 for k in fields],
                  freeze_rows=3, freeze_columns=min(3, len(fields)-1))
    sheet.note(name, 1, 30)
    sheet.note(note, 2, 45)
    sheet.add([Cell(k, 3) for k in fields])
    for row in rows:
        values = [row.get(k) for k in fields]
        sheet.add([json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v for v in values])
    if not rows:
        sheet.add(['Not applicable / no matching groups'])
    sheet.filter_range = f'A3:{_column(len(fields))}{len(sheet.rows)}'
    return sheet


def plots_pin(output, tables, limit):
    if not limit:
        return []
    require_plotting_pin()
    import matplotlib.pyplot as plt
    directory = output / 'plots_pin'
    directory.mkdir()
    groups = defaultdict(list)
    for row in tables['SingularValues']:
        groups[(row['model'], row['seed'], row['matrix'])].append(row)
    sizes = {(r['model'], r['seed'], r['matrix']): r['parameters'] for r in tables['Matrices']}
    order = sorted(groups, key=lambda k: (-sizes[k], k))[:limit]
    manifest = []
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}):
        for index, key in enumerate(order):
            model, seed, matrix = key
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
            circuits = defaultdict(list)
            for row in groups[key]:
                circuits[row['circuit']].append(row)
            for circuit, rows in sorted(circuits.items()):
                rows.sort(key=lambda r: r['index'])
                maximum = rows[0]['singular_value']
                xs = [r['index'] for r in rows]
                ys = [r['singular_value']/maximum if maximum else 0. for r in rows]
                axes[0].plot(xs, ys, label=circuit)
                axes[1].plot(xs, [r['cumulative_energy'] for r in rows], label=circuit)
            axes[0].set(xlabel='Singular value index', ylabel='Singular value / largest', ylim=(0, 1.05))
            axes[1].set(xlabel='Retained rank', ylabel='Cumulative squared singular-value energy', ylim=(0, 1.02))
            axes[1].axhline(.99, color='#777777', lw=.8, ls='--')
            for ax in axes:
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
            fig.suptitle(f'{model} | seed {seed}\n{matrix}', fontsize=10)
            path = directory / f'spectrum_{index+1:03d}_pin.png'
            fig.savefig(path, dpi=160)
            plt.close(fig)
            manifest.append(dict(model=model, seed=seed, matrix=matrix, path=str(path.relative_to(output))))
    (directory / 'manifest_pin.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def write_report_pin(output, tables, record, max_plots=12):
    output = Path(output)
    names = ('Parameters', 'Matrices', 'IndependentLowRank', 'SubspaceOverlap',
             'SharedBudget', 'SharedErrors', 'SingularValues', 'Notes')
    for name in names:
        rows = tables[name]
        fields = list(dict.fromkeys(k for row in rows for k in row)) or ['status']
        with (output / (name + '_pin.csv')).open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    readme = Sheet('ReadMe', [25, 130], freeze_rows=1, freeze_columns=1)
    readme.add([Cell('项目', 3), Cell('解释与边界', 3)])
    for key, value in NOTES:
        readme.add([key, value], height=48)
    sheets = [readme, sheet_pin('Checkpoints', record['checkpoints'], 'Source identity, configuration, hashes and parameter coverage')]
    for name in names:
        if name == 'SingularValues':
            continue  # unbounded full spectra stay in CSV; ranks are in Matrices
        sheets.append(sheet_pin(name, tables[name],
            'Saved coordinates, no alignment; train-only calibration; no compressed-model accuracy claim'))
    _write_workbook(output / 'spectral_summary_pin.xlsx', sheets)
    images = plots_pin(output, tables, max_plots)
    content = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Low-rank probe</title>',
        '<style>body{font:16px/1.6 system-ui;max-width:1150px;margin:32px auto;padding:0 18px;color:#25354a}'
        'img{width:100%;border:1px solid #ddd}table{border-collapse:collapse}td{padding:10px;border:1px solid #ddd}'
        'code{overflow-wrap:anywhere}</style><h1>低秩与共享基底诊断</h1>',
        '<p><a href="spectral_summary_pin.xlsx">下载 Excel</a> · <a href="probe_pin.json">配置与来源记录</a></p><table>']
    content += [f'<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>' for k, v in NOTES]
    content.append('</table><h2>完整 CSV</h2><ul>')
    content += [f'<li><a href="{name}_pin.csv">{name}</a></li>' for name in names]
    content.append('</ul><h2>谱图</h2>')
    if not images:
        content.append('<p>MAX_PLOTS=0：未生成图；完整数值见 Excel / CSV。</p>')
    for image in images:
        caption = f'{image["model"]} / seed {image["seed"]} / {image["matrix"]}'
        content.append(f'<p><code>{html.escape(caption)}</code></p><img src="{image["path"]}" alt="Singular-value spectrum">')
    content.append('</html>')
    (output / 'report_pin.html').write_text('\n'.join(content), encoding='utf-8')
