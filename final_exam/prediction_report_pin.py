"""Offline figures, full prediction CSV, and Excel for checkpoint replay."""
from __future__ import annotations

import csv
import gzip
import html
import json
from pathlib import Path

import numpy as np

from final_exam.prediction_analysis_pin import IDENTITY, stable_seed


NOTES = [
    ('数据来源', '只重放已保存 checkpoint；使用其实际 selected_sample_ids、manifest 和 preprocessing。没有重新采样、拟合归一化、训练或选择 checkpoint。'),
    ('完成判定', '同时有 checkpoint 和完成的 test_pin.json 才纳入。整个 run 未完成不妨碍分析其中已完成的模型/seed；跳过项见 Inventory。'),
    ('完整样本', '推理和统计使用全部保存的 val/test 样本；MAX_PLOT_POINTS 只限制散点图显示，不影响任何统计。原始划分见 sources_pin，实际 checkpoint 子集见 selection_pin.json。'),
    ('缺失标签', '保留全量预测 CSV 的 observed 标志。缺失或训练未支持的指标不参与统计；n<2 或标签恒定时 R² 留空。没有裁剪负 R² 或删除大误差样本。'),
    ('归一化', '物理量 MAE/RMSE/R² 与 normalized_mse 同时保存。Aggregate 的总 MSE 按有效标签计数汇总；各模型沿用自身训练保存的变换。'),
    ('训练分位数', 'Thresholds 仅由 checkpoint 实际训练子集里有监督的真实标签计算；Strata 的五个区间依据这些分位数划分 val/test。不是按测试误差划分训练区间。'),
    ('大误差', 'LargestErrors 按物理绝对残差排序，PairedCases 按归一化平方误差差值排序。Top 1%/5% SSE 比例是残差贡献诊断，不证明这些样本是错误标签。'),
    ('重复实验', 'Summary 使用全部已完成 seed，std 为样本标准差 ddof=1；仅一个 seed 时 std 留空。不同电路或不同 cohort 不混合。同模型不同 seed 的保存训练设置若不同则拒绝合并。'),
    ('配对边界', '要求原始数据指纹、划分、实际 checkpoint 样本子集和性能归一化一致，再按相同 seed、样本 ID 和标签配对；不做不同集合的交集比较。缺少基线见 PairStatus。'),
    ('差值方向', '所有 delta=model-baseline。MAE/RMSE/MSE 的负差值表示模型更好，R² 的正差值表示模型更好。配对表与非配对 Summary 的 seed 数可能不同。'),
    ('置信区间', '95% percentile bootstrap 的单位是配对训练 seed；条件于固定数据/划分，只反映训练随机性，不是新样本总体泛化区间。少 seed 时区间不稳定。'),
    ('显著性检验', '双侧配对 seed sign-flip，假定零假设下差值符号可交换；<=16 seed 精确枚举，否则 Monte Carlo。Holm 对同 cohort/电路/split 的全部模型×指标×统计量校正。'),
    ('小样本限制', '单 seed 不计算 CI/p。三个同向非零配对 seed 的精确双侧 p 最小为 0.25；不能把几千个测试样本当成几千次独立训练重复。'),
    ('结果重放', 'Aggregate 同时列原 test 报告和重放 MSE 的差值，便于发现版本/硬件/批量差异；明显不一致时应先核查，不能直接用于解释原训练结果。'),
    ('后续用途', 'val/test 分析为诊断；本脚本不据此更换 checkpoint 或筛选最好/最坏 seed。所有模型的不同推理 OP 需求记录在 probe_pin.json。'),
]


def csv_value(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value


def write_table_pin(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with Path(path).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: csv_value(v) for k, v in row.items()} for row in rows)


class PredictionReportPin:
    def __init__(self, output, plots=True, max_plot_points=5000, seed=17):
        self.output, self.plots = Path(output), plots
        self.max_plot_points, self.seed = max_plot_points, seed
        self.plot_index = []

    def __enter__(self):
        self.prediction_file = gzip.open(self.output / 'predictions_pin.csv.gz', 'wt', newline='', encoding='utf-8')
        self.prediction_writer = csv.DictWriter(self.prediction_file, fieldnames=[*IDENTITY,
            'sample_id', 'source_sample_id', 'observed', 'target', 'prediction', 'residual',
            'normalized_target', 'normalized_prediction', 'normalized_residual'])
        self.prediction_writer.writeheader()
        self.membership_file = (self.output / 'sample_membership_pin.csv').open('w', newline='', encoding='utf-8-sig')
        self.membership_writer = csv.DictWriter(self.membership_file,
            fieldnames=['source_id', 'circuit', 'split', 'sample_id', 'source_sample_id', 'labeled_train'])
        self.membership_writer.writeheader()
        (self.output / 'plots_pin').mkdir()
        return self

    def __exit__(self, *args):
        self.prediction_file.close()
        self.membership_file.close()

    def prediction(self, row):
        self.prediction_writer.writerow(row)

    def membership(self, row):
        self.membership_writer.writerow(row)

    def plot(self, item, checkpoint_id, thresholds):
        if not self.plots:
            return
        import matplotlib.pyplot as plt
        from device_jepa.excel import METRICS
        y, p = item['target'], item['prediction']
        metric = item['metric']
        title, unit, scale = METRICS.get(metric, (metric, 'native units', 1.))
        identity = {k: item[k] for k in IDENTITY}
        label = f'{item["circuit"]} | {item["model"]} | seed {item["seed"]} | {item["split"]} | {metric}'
        filename = f'{checkpoint_id}_{item["split"]}_{len(self.plot_index):05d}_pin.png'
        path = self.output / 'plots_pin' / filename
        rng = np.random.default_rng(stable_seed(label, self.seed))
        selected = np.sort(rng.choice(len(y), min(len(y), self.max_plot_points), replace=False)) if len(y) else []
        with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}):
            fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
            fig.suptitle(label, fontsize=9)
            if len(y):
                xs, ps = y[selected] / scale, p[selected] / scale
                error, abs_error = (p - y) / scale, np.abs(p - y) / scale
                axes[0, 0].scatter(xs, ps, s=7, alpha=.35, rasterized=True, color='#16769b')
                low, high = min(y.min(), p.min()) / scale, max(y.max(), p.max()) / scale
                if low == high:
                    low, high = low - 1, high + 1
                axes[0, 0].plot([low, high], [low, high], '--', color='#666666', lw=1)
                stats = item['stats']
                r2 = f'{stats["r2"]:.5f}' if stats['r2'] is not None else 'undefined'
                axes[0, 0].set_title(f'n={len(y)}  R²={r2}\nMAE={stats["mae"]/scale:.4g}  RMSE={stats["rmse"]/scale:.4g}')
                axes[0, 1].scatter(xs, error[selected], s=7, alpha=.35, color='#b86b32', rasterized=True)
                axes[0, 1].axhline(0, color='#666666', ls='--', lw=1)
                if thresholds:
                    for q in ('p05', 'p95'):
                        axes[0, 1].axvline(thresholds[q] / scale, color='#999999', ls=':', lw=1)
                axes[0, 1].set_title('Residual vs truth; dotted lines: train P5/P95')
                ordered = np.sort(abs_error)
                axes[1, 0].step(np.r_[0., ordered], np.r_[0., np.arange(1, len(y) + 1) / len(y)], where='post', color='#16769b')
                axes[1, 0].set_ylim(0, 1.02)
                axes[1, 0].set_title('Absolute error empirical CDF (all samples)')
                norm = float(abs_error.max())
                squared = np.sort((abs_error / norm) ** 2)[::-1] if norm else np.zeros(len(y))
                cumulative = np.cumsum(squared) / squared.sum() if norm else squared
                axes[1, 1].plot(np.r_[0., np.arange(1, len(y) + 1) / len(y) * 100], np.r_[0., cumulative * 100], color='#b86b32')
                axes[1, 1].set_title('Largest residuals: cumulative squared error')
                axes[1, 1].set_ylim(0, 102)
            else:
                for ax in axes.flat:
                    ax.text(.5, .5, 'No observed supported targets', ha='center', transform=ax.transAxes)
            axes[0, 0].set(xlabel=f'True {title} ({unit})', ylabel=f'Predicted ({unit})')
            axes[0, 1].set(xlabel=f'True ({unit})', ylabel=f'Prediction − truth ({unit})')
            axes[1, 0].set(xlabel=f'Absolute error ({unit})', ylabel='Fraction of samples')
            axes[1, 1].set(xlabel='Largest-error samples (%)', ylabel='Share of squared error (%)')
            for ax in axes.flat:
                ax.grid(alpha=.15)
            fig.savefig(path, dpi=150)
            plt.close(fig)
        self.plot_index.append(dict(identity, label=label, path=str(path.relative_to(self.output)),
                                   total_samples=len(y), plotted_scatter_samples=len(selected)))

    def finish(self, tables, record):
        from device_jepa.excel import _write_workbook
        from final_exam.spectral_report_pin import sheet_pin
        tables['ReadMe'] = [dict(topic=k, explanation=v) for k, v in NOTES]
        sheets = []
        for name, rows in tables.items():
            write_table_pin(self.output / f'{name}_pin.csv', rows)
            sheets.append(sheet_pin(name, rows, 'Saved-checkpoint diagnostics; all completed seeds. See ReadMe for statistical definitions.'))
        _write_workbook(self.output / 'prediction_summary_pin.xlsx', sheets)
        (self.output / 'plots_pin/index_pin.json').write_text(json.dumps(self.plot_index, ensure_ascii=False, indent=2), encoding='utf-8')
        (self.output / 'statistics_pin.json').write_text(json.dumps(tables, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        payload = json.dumps(self.plot_index, ensure_ascii=False).replace('<', '\\u003c')
        notes = ''.join(f'<li><b>{html.escape(k)}：</b>{html.escape(v)}</li>' for k, v in NOTES)
        page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>模型预测诊断</title><style>body{font:16px system-ui,sans-serif;margin:24px auto;max-width:1250px;padding:0 18px;color:#183043;background:#f7fafc}h1{font-size:28px}section{background:white;border:1px solid #dce5ec;padding:20px;margin:18px 0;border-radius:10px}select,input{max-width:100%;padding:9px;margin:6px 0}img{width:100%;height:auto}li{margin:9px 0;line-height:1.6}a{color:#066d93}.muted{color:#526677}</style>
<h1>保存模型的预测与误差诊断</h1><p>按原始采样和划分重放；每个模型、seed、指标分别展示 val/test。所有指标保留完整有效样本。</p>
<p><a href="prediction_summary_pin.xlsx">Excel 统计</a> · <a href="predictions_pin.csv.gz">完整逐样本预测 CSV.gz</a> · <a href="sample_membership_pin.csv">样本划分名单</a> · <a href="statistics_pin.json">统计 JSON</a></p>
<section><label>筛选电路 / 模型 / 指标 / seed：<input id="filter" placeholder="例如 Slew_Rise 或 seed 29"></label><br><select id="select"></select><p id="info" class="muted"></p><a id="download"><img id="plot" alt="预测诊断图"></a></section>
<section><h2>统计口径</h2><ul>''' + notes + '''</ul></section><script>
const plots=''' + payload + ''';const select=document.getElementById('select'),filter=document.getElementById('filter');
function show(){const item=select.selectedIndex<0?null:plots[Number(select.value)];document.getElementById('plot').hidden=!item;if(!item){document.getElementById('download').removeAttribute('href');document.getElementById('info').textContent='没有匹配图片；可查看 Excel / CSV。';return;}document.getElementById('plot').src=item.path;document.getElementById('download').href=item.path;document.getElementById('info').textContent='有效样本 '+item.total_samples+'；散点显示 '+item.plotted_scatter_samples+'。误差统计和 CDF 使用全部有效样本。';}
function update(){select.replaceChildren();plots.forEach((p,i)=>{if(p.label.toLowerCase().includes(filter.value.toLowerCase())){const o=document.createElement('option');o.value=i;o.textContent=p.label;select.append(o);}});show();}filter.addEventListener('input',update);select.addEventListener('change',show);update();
</script></html>'''
        (self.output / 'prediction_viewer_pin.html').write_text(page, encoding='utf-8')
