"""Small, shareable case report; all values also available as JSON and CSV."""
from __future__ import annotations

import html
import json
from pathlib import Path


NOTES = [
    ('样本身份', 'source_sample_id 是原 SQLite samples.sample_id，不是数组下标、sample_index 或排序后行号。Cases 记录 canonical ID、原始 ID 和实际 split。'),
    ('推理边界', '仅重放保存 checkpoint；性能和 OP 预测均不读取真实 OP/标签。报告中的真实值在模型前向之外读取。没有训练、重新采样或拟合预处理。'),
    ('OP 对齐', '按照现有 PinResidentDataset 的 device_id 排序建立物理器件轴；按字段名称对应保存的 OP 顺序。外部端口和 padding 不冒充器件。'),
    ('OP 单位', 'predicted_op 通过原 checkpoint 的 device_preprocessing.op 反变换，支持线性/asinh；不对 PMOS 电流/电压取绝对值。不支持预测/未观测字段保留空值，不能当作零误差。'),
    ('OP 误差', 'normalized_residual = 预测标准化 OP − 按保存变换计算的真实 OP。不同物理单位不能直接混合比较。OPSummary 按有效器件字段平均，仅用于诊断，不冒充原训练损失。'),
    ('训练范围', 'Parameters/OPComparison 的 train_* 来自实际保存的完整 train 子集，同器件 ID、类型和物理型号分别统计。经验分位数不依赖高斯分布；percentile 使用中位秩，恒定字段相同值为 50%。'),
    ('误差参照', 'OPReference 用固定 reference_seed 从实际 train 子集无放回取至多 reference_samples 个样本，重放冻结 OP 预测器；名单见 probe_pin.json。是训练内诊断，不能当作泛化精度。'),
    ('同一预测器', 'PredictorChecks 比较每个 seed 的 OP 权重和原器件预处理指纹。相同 OP 预测器并不保证下游性能预测相同。'),
    ('原始证据', 'RawChecks 对照原数据库和保存 canonical 样本的设计、标签和器件 OP。mismatch 应先排查来源，不能混用不同数据副本解释同一实验。仅查询指定 ID，不改数据库。'),
    ('波形限制', '训练保存数据通常只含设计参数、性能标量和 DC OP；它们不能恢复瞬态波形。缺失波形明确标注 not_supplied，不生成伪造波形。ArtifactCandidates 仅列元数据中已有路径，不猜测路径归属。'),
    ('导入波形', '填写 waveform_manifest_template_pin.json 的原始 sample ID 与带表头的 CSV/TSV/空白分隔文件路径，再通过 WAVEFORM_MANIFEST 导入。需注明原保存或重仿真来源、正确的时间/电压单位和输出节点。'),
    ('波形导数', '曲线的斜率是正时间间隔上的相邻有限差分，不平滑、不插值；重复时间间隔不求导。未给定原始激励/测量窗口/提取算法，不能把全局最大导数当成原 SR 标签。'),
    ('结论边界', '高误差、训练分布边缘、异常工作点都不直接等价于坏标签。先核对器件/单位、OP 状态、刺激和波形，再判断原因；本脚本不删除数据、不替换测试标签。'),
]


def finish_case_report_pin(output, tables, record, plots, wave_figures):
    import numpy as np
    from device_jepa.excel import _write_workbook
    from final_exam.prediction_report_pin import write_table_pin
    from final_exam.spectral_report_pin import sheet_pin
    output = Path(output)
    tables['ReadMe'] = [dict(topic=k, explanation=v) for k, v in NOTES]
    sheets = []
    for name, rows in tables.items():
        write_table_pin(output / f'{name}_pin.csv', rows)
        sheets.append(sheet_pin(name, rows, 'Selected original sample IDs; train-only references; see ReadMe.'))
    _write_workbook(output / 'case_summary_pin.xlsx', sheets)
    (output / 'case_statistics_pin.json').write_text(json.dumps(tables, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    figures = list(wave_figures)
    if plots:
        import matplotlib.pyplot as plt
        for index, (model, seed, sid) in enumerate(dict.fromkeys(
                (r['model'], r['seed'], r['source_sample_id']) for r in tables['OPComparison'])):
            rows = [r for r in tables['OPComparison'] if (r['model'], r['seed'], r['source_sample_id']) == (model, seed, sid)]
            devices = list(dict.fromkeys(r['device'] for r in rows))
            fields = list(dict.fromkeys(r['field'] for r in rows))
            residual, percentile = np.full((len(devices), len(fields)), np.nan), np.full((len(devices), len(fields)), np.nan)
            for r in rows:
                i, j = devices.index(r['device']), fields.index(r['field'])
                if r['normalized_residual'] is not None:
                    residual[i, j] = np.log1p(abs(r['normalized_residual']))
                if r['train_percentile'] is not None:
                    percentile[i, j] = r['train_percentile']
            fig, axes = plt.subplots(2, 1, figsize=(max(12, len(fields)*.27), max(8, len(devices)*.52)), layout='constrained')
            for ax, values, title, limits in zip(axes, (residual, percentile),
                    ('log1p(|normalized OP error|); gray = missing/unsupported', 'True OP empirical percentile in saved train set'),
                    ((0., None), (0., 100.))):
                cmap = plt.get_cmap('viridis').copy()
                cmap.set_bad('#dddddd')
                im = ax.imshow(np.ma.masked_invalid(values), aspect='auto', cmap=cmap, vmin=limits[0], vmax=limits[1])
                ax.set_xticks(range(len(fields)), fields, rotation=65, ha='right', fontsize=7)
                ax.set_yticks(range(len(devices)), devices, fontsize=8)
                ax.set_title(title, fontsize=10)
                fig.colorbar(im, ax=ax, fraction=.025)
            fig.suptitle(f'{model} | seed {seed} | original sample {sid}', fontsize=11)
            filename = f'plots_pin/op_case_{index:03d}_pin.png'
            fig.savefig(output / filename, dpi=140)
            plt.close(fig)
            figures.append(dict(label=f'{model} | seed {seed} | sample {sid} | OP error / train range', path=filename))
    (output / 'plots_pin/index_pin.json').write_text(json.dumps(figures, ensure_ascii=False, indent=2), encoding='utf-8')
    def html_table(rows, columns):
        headers = ''.join(f'<th>{html.escape(c)}</th>' for c in columns)
        cells = []
        for row in rows:
            cells.append('<tr>' + ''.join('<td>' + html.escape(
                f'{row.get(c):.6g}' if isinstance(row.get(c), float) else str(row.get(c, ''))) + '</td>' for c in columns) + '</tr>')
        return f'<div class="table"><table><thead><tr>{headers}</tr></thead><tbody>{"".join(cells)}</tbody></table></div>'
    performance = [r for r in tables['Performance'] if r['metric'] in ('Slew_Rise_V_us', 'Slew_Fall_V_us')]
    if not performance:
        performance = tables['Performance']
    panels = ''.join(f'<section><h2>{html.escape(f["label"])}</h2><a href="{html.escape(f["path"], quote=True)}"><img src="{html.escape(f["path"], quote=True)}"></a></section>' for f in figures)
    notes = ''.join(f'<li><b>{html.escape(k)}：</b>{html.escape(v)}</li>' for k, v in NOTES)
    wave_rows = [{k: r.get(k) for k in ('source_sample_id', 'status', 'column', 'provenance', 'reason')} for r in tables['Waveforms']]
    raw_mismatch = sum(r['status'] != 'matched' for r in tables['RawChecks'])
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>大误差样本回查</title><style>body{max-width:1400px;margin:24px auto;padding:0 18px;font:15px system-ui;color:#193448;background:#f5f8fa}section{background:white;border:1px solid #d4e0e8;border-radius:10px;padding:18px;margin:18px 0}h1{font-size:27px}h2{font-size:18px}img{max-width:100%}.table{overflow:auto}table{border-collapse:collapse}th,td{padding:8px;border:1px solid #d4e0e8;text-align:left;white-space:nowrap}a{color:#087698}li{margin:9px 0}</style>
<h1>大误差样本：设计、OP 与波形回查</h1><p>''' + html.escape(record['circuit']) + ' | 原始样本 ID：' + html.escape(str(record['original_sample_ids'])) + '''</p>
<p><a href="case_summary_pin.xlsx">Excel 报告</a> · <a href="case_details_pin.json">完整样本与原始证据</a> · <a href="case_statistics_pin.json">完整统计 JSON</a> · <a href="waveform_manifest_template_pin.json">补充波形模板</a></p>
<section><h2>SR 预测对照</h2>''' + html_table(performance, ['model', 'seed', 'source_sample_id', 'metric', 'target', 'prediction', 'residual']) + '''</section>
<section><h2>原始数据核对与波形状态</h2><p>原始数据库：''' + html.escape(record['raw_source']['status']) + f'；不匹配/缺失检查项：{raw_mismatch}。' + '''</p>''' + html_table(wave_rows, ['source_sample_id', 'status', 'column', 'provenance', 'reason']) + '''</section>
<section><h2>OP 预测器一致性</h2>''' + html_table(tables['PredictorChecks'], ['model', 'seed', 'same_predictor_as', 'predictor_and_preprocessing_digest']) + '</section>' + panels + '<section><h2>口径与限制</h2><ul>' + notes + '</ul></section></html>'
    (output / 'case_review_pin.html').write_text(page, encoding='utf-8')
