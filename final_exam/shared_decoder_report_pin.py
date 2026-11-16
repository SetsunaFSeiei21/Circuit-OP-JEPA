"""Reports for within-checkpoint shared-down-basis decoder diagnostics.

Tables may contain several circuits/models/seeds, but this module only displays
those independently computed records. It never fits or averages across them.
"""
from collections import defaultdict
import csv
import html
import json
from pathlib import Path

from device_jepa.excel import Cell, Sheet, _write_workbook
from final_exam.spectral_report_pin import require_plotting_pin, sheet_pin


TABLE_NAMES_PIN = ('Parameters', 'Groups', 'Budgets', 'Errors', 'SubspaceOverlap',
                   'PrincipalAngles', 'CrossBasis', 'VectorCosines',
                   'DecoderFeatures', 'Validation', 'Spectra', 'Notes')
NOTES_PIN = [
    ('分析单位', '每个电路、模型、seed 的 checkpoint 独立分析。共享只发生于该模型内部不同 PE 对应的 decoder；汇总表不跨电路、模型或 seed 拟合基底，也不求混合平均。'),
    ('共享结构', '权重统一按 [out,in] 记为 W_i=B_i A。对应 decoder 共享下投影 A，各自保留独立上投影 B_i。不同 decoder 层号、self/cross attention 和 Q/K/V/O/FFN 位置分别分组。'),
    ('拟合方法', '将对应 W_i 纵向拼接，直接对原权重做 SVD；A 取前 r 个右奇异向量，B_i=W_i Aᵀ。未按分支范数重加权。这最小化该组的权重平方 Frobenius 误差，不保证最小激活或任务误差。'),
    ('独立对照', '独立低秩对每个 W_i 单独截断 SVD；共享下基对纵向拼接矩阵截断。共享组的最大可用秩与单矩阵可能不同，必须同时查看 requested_rank 和实际秩。'),
    ('两两相似度', '只比较同组两分支的右奇异子空间。actual_rank=min(requested_rank, numerical_rank_a, numerical_rank_b)，排除任意零空间向量。numerical_rank 使用来源权重 dtype 的 eps·max(shape)·最大奇异值阈值。overlap=||V_aᵀ V_b||F²/r；0 表示正交，1 表示两个所选子空间相同。'),
    ('随机参照', '两个独立均匀随机 r 维子空间在 d 维输入空间的期望 overlap=r/d；不是显著性检验。chance_adjusted_overlap=(overlap-r/d)/(1-r/d)，负数保留。r=d 时 overlap 必为 1，调整值留空；不能作为低秩共享证据。'),
    ('主角度与向量', 'PrincipalAngles 按角度从小到大记录 acos(svdvals(V_aᵀ V_b))，单位为度。VectorCosines 每行保存一个 branch_a 右奇异向量与 branch_b 前若干向量的绝对余弦数组，列次序为 1..vector_rank_b。向量图只用于辅助观察，重复奇异值内可旋转；截断处接近重复时 cutoff_ambiguous=true，子空间选择本身也可能不稳定。'),
    ('交叉基重建', 'CrossBasis 的 donor_branch 提供下基 A，target_branch 提供被重建的 W；重建 (W Aᵀ)A。与目标自身同实际秩的最优 SVD 基比较。retained_vs_own 为交叉/自身保留权重能量之比，越接近 1 额外损失越小。excess_relative_weight_sse 为交叉减自身的归一化平方重建误差。'),
    ('交叉激活', 'cross_output_error 与 own_output_error 都在目标分支自己的原始 train 输入上测量，排除 bias；不交换不同 PE 的样本或输入。零秩、未捕获激活等留空。两两分析复用原始权重 SVD，不新增训练或验证前向。'),
    ('参数预算', 'm 个 [out,in] 矩阵：完整权重 m·out·in；独立低秩为各分支 r_i(out+in) 之和；共享下基为 r·in+m·out·r。不包含保持不变的 bias；saving<0 表示该候选反而增加参数。'),
    ('已有低秩模型', '如果来源本来就是低秩模型，权重诊断针对其有效线性映射；参数对照须区分来源实际参数与假想完整矩阵参数。不能把假想完整矩阵预算当作来源模型大小。'),
    ('误差定义', 'relative_weight_error=||W-Wr||F/||W||F；relative_layer_output_error=||(W-Wr)X||F/||WX||F。后者使用原模型捕获的输入激活，不是端到端性能误差。分母为零但残差非零时相对误差留空。'),
    ('数据边界', '激活只取保存划分中的 train 样本，使用来源 checkpoint 的预处理统计，不重新拟合归一化。可选端到端诊断只使用 val；test 不用于本分析或 rank 选择。'),
    ('验证诊断', 'Validation 是指定验证子集上的临时权重近似结果，包含原始/独立低秩/共享下基对照。它不包含微调或重新训练，不能直接代表共享下基从头训练的效果。'),
    ('临时替换', '端到端验证将 B_i A 重建为完整矩阵后临时替换权重，测量近似导致的误差；不会保存新模型，也不是实际因子化模型的耗时或显存测试。DecoderFeatures 单独记录各 PE decoder 特征相对原模型的变化。'),
    ('坐标边界', '使用 checkpoint 保存的隐藏坐标，未对齐 FFN 神经元排列或注意力头。旧分支直接共享误差大，不排除从头共享训练后可行；误差小也不证明训练必然保持精度。'),
    ('只读与缺失值', '不改来源 checkpoint、配置或划分。未采集的激活、未执行的验证、未定义的相对误差留空，空白不等于 0。'),
    ('速度边界', '共享权重减少参数，但分支输入不同，仍需各自计算 A x_i。本报告没有测量共享架构训练/推理加速，不能从参数下降直接推出速度提升。'),
    ('图表覆盖', 'CSV 和 Excel 保存完整诊断记录；完整奇异值序列仅放在 Spectra_pin.csv。MAX_PLOTS 限制绘图分组数，每组包含谱误差图、按 rank 的子空间/交叉基热图以及向量余弦图；每页最多四个 rank 或三个分支对。VECTOR_LIMIT 仅限制向量图每轴根数（默认 64，最多 256），不限制 RANKS 的子空间/主角度/交叉基计算；0 不导出向量表或图。索引见 plots_pin/manifest_pin.json。'),
]
TABLE_NOTES_PIN = {
    'Parameters': '来源模型的实际参数账本；不要跨模型相加后作为单个模型大小。',
    'Groups': '每组仅包含同一 checkpoint 内对应位置的 decoder 权重；谱统计采用平方奇异值能量。',
    'Budgets': '每个候选 rank 的组内权重预算；包含实际采用的秩，bias 等不参与替换。',
    'Errors': '各分支的权重/原始输入激活误差；独立低秩与共享下基分别记录，空白不是零。',
    'SubspaceOverlap': '同一 checkpoint 内各 decoder 对的右奇异子空间相似度；同时查看实际秩、随机期望、截断歧义和能量覆盖。',
    'PrincipalAngles': '每个分支对、候选 rank 下的完整主角度（度，升序）及其余弦；零有效秩不生成角度。',
    'CrossBasis': '方向为 donor_branch 提供下基，target_branch 被重建；目标自身同秩 SVD 为对照，激活只用目标 train 输入。',
    'VectorCosines': '每行是一个 branch_a 右奇异向量，absolute_cosines 数组依次对应 branch_b 的第 1..vector_rank_b 根向量；符号不敏感但旋转敏感，不能替代子空间指标。',
    'DecoderFeatures': '指定 val 子集上各 decoder 输出特征相对原模型的误差；按临时替换的 scope/rank/strategy 分别记录。',
    'Validation': '指定 val 子集的临时近似实验；未训练、未微调、未使用 test，按实际 scope 解释替换范围。',
    'Notes': '未支持或未执行的部分及原因；不得将缺失值视为零误差。',
}


def _identity_pin(row):
    """Do not combine equal projection names from distinct checkpoints."""
    return tuple(str(row.get(field, '')) for field in ('circuit', 'model', 'seed', 'group'))


def _json_pin(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _csv_value_pin(value):
    return _json_pin(value) if isinstance(value, (dict, list, tuple)) else value


def _curve_pin(row):
    # The analysis schema uses 'spectrum' for stacked/independent curves. Accept
    # explicit branch records too, so no branch spectra are silently averaged.
    label = row.get('spectrum', row.get('method', 'stacked'))
    if row.get('branch') is not None:
        label = f'{label} / branch {row["branch"]}'
    return str(label)


def _similarity_plots_pin(plt, output, directory, index, key, tables):
    """Bounded pages; all pairs/ranks remain available in CSV and Excel."""
    import numpy as np
    overlap = [r for r in tables['SubspaceOverlap'] if _identity_pin(r) == key]
    cross = [r for r in tables['CrossBasis'] if _identity_pin(r) == key]
    vectors = [r for r in tables['VectorCosines'] if _identity_pin(r) == key]
    circuit, model, seed, group = key
    prefix = f'{circuit} | {model} | seed {seed}\n{group}'
    manifest = []

    def save(fig, suffix, kind, **metadata):
        path = directory / f'shared_decoder_{index+1:03d}_{suffix}_pin.png'
        fig.savefig(path, dpi=160)
        manifest.append(dict(circuit=circuit, model=model, seed=seed, group=group,
                             kind=kind, path=str(path.relative_to(output)), **metadata))

    ranks = sorted({r['requested_rank'] for r in overlap})
    branches = sorted({r[f'branch_{side}'] for r in overlap for side in ('a', 'b')})
    positions = {branch:i for i,branch in enumerate(branches)}
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#dddddd')
    for offset in range(0, len(ranks), 4):
        page = ranks[offset:offset+4]
        fig, axes = plt.subplots(2, len(page), squeeze=False,
                                 figsize=(4*len(page), 7.8), layout='constrained')
        try:
            for col, rank in enumerate(page):
                similarity = np.full((len(branches), len(branches)), np.nan)
                retained = np.full_like(similarity, np.nan)
                for row in overlap:
                    if row['requested_rank'] != rank or row['overlap'] is None:
                        continue
                    a, b = positions[row['branch_a']], positions[row['branch_b']]
                    similarity[a,b] = similarity[b,a] = row['overlap']
                    similarity[a,a] = similarity[b,b] = 1.
                for row in cross:
                    if row['requested_rank'] != rank or row['retained_vs_own'] is None:
                        continue
                    target, donor = positions[row['target_branch']], positions[row['donor_branch']]
                    retained[target,donor] = row['retained_vs_own']
                    retained[target,target] = 1.
                for ax, values, title in ((axes[0,col], similarity, 'Right-subspace overlap'),
                                          (axes[1,col], retained, 'Cross / own retained energy')):
                    im = ax.imshow(values, vmin=0, vmax=1, cmap=cmap, interpolation='nearest')
                    ax.set(title=f'{title}\nrequested rank {rank}',
                           xticks=range(len(branches)), yticks=range(len(branches)),
                           xticklabels=[f'PE {b}' for b in branches],
                           yticklabels=[f'PE {b}' for b in branches])
                    if len(branches) <= 10:
                        for y,x in np.ndindex(values.shape):
                            value = values[y,x]
                            ax.text(x,y, 'n/a' if np.isnan(value) else f'{value:.3f}',
                                    ha='center',va='center',fontsize=8,
                                    color='white' if np.isfinite(value) and value < .5 else 'black')
                    fig.colorbar(im, ax=ax, shrink=.7)
                axes[0,col].set(xlabel='Decoder branch', ylabel='Decoder branch')
                axes[1,col].set(xlabel='Donor basis', ylabel='Target weight')
            fig.suptitle(prefix + '\nOverlap random expectation r/d and actual ranks are in SubspaceOverlap; gray = undefined.', fontsize=10)
            save(fig, f'subspaces_{offset//4+1:02d}', 'subspace_and_cross_basis', requested_ranks=page)
        finally:
            plt.close(fig)
    pairs = defaultdict(list)
    for row in vectors:
        pairs[(row['branch_a'], row['branch_b'])].append(row)
    ordered = sorted(pairs)
    for offset in range(0, len(ordered), 3):
        page = ordered[offset:offset+3]
        fig, axes = plt.subplots(1, len(page), squeeze=False,
                                 figsize=(5*len(page), 4.7), layout='constrained')
        try:
            for ax, pair in zip(axes[0], page):
                rows = sorted(pairs[pair], key=lambda r:r['vector_index_a'])
                values = np.array([r['absolute_cosines'] for r in rows])
                im = ax.imshow(values, vmin=0, vmax=1, cmap=cmap, interpolation='nearest',
                               extent=(.5, values.shape[1]+.5, values.shape[0]+.5, .5), aspect='auto')
                ax.set(title=f'PE {pair[0]} vs PE {pair[1]}',
                       xlabel=f'PE {pair[1]} right singular vector index',
                       ylabel=f'PE {pair[0]} right singular vector index')
                fig.colorbar(im, ax=ax, shrink=.8, label='Absolute cosine')
            fig.suptitle(prefix + '\nLeading right singular vectors: sign invariant, basis-rotation sensitive.', fontsize=10)
            save(fig, f'vectors_{offset//3+1:02d}', 'right_singular_vector_cosines', branch_pairs=page)
        finally:
            plt.close(fig)
    return manifest


def plots_shared_decoder_pin(output, tables, limit):
    if not limit:
        return []
    spectra, errors = defaultdict(list), defaultdict(list)
    for row in tables['Spectra']:
        spectra[_identity_pin(row)].append(row)
    for row in tables['Errors']:
        errors[_identity_pin(row)].append(row)
    keys = set(spectra) | set(errors)
    if not keys:
        return []
    sizes = {_identity_pin(row): row.get('dense_weight_parameters', row.get('dense_parameters', row.get('parameters', 0)))
             for row in tables['Groups']}
    order = sorted(keys, key=lambda key: (-sizes.get(key, 0), key))[:limit]
    require_plotting_pin()
    import matplotlib.pyplot as plt
    directory = Path(output) / 'plots_pin'
    directory.mkdir()
    manifest = []
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}):
        for index, key in enumerate(order):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout='constrained')
            try:
                spectrum_curves = defaultdict(list)
                for row in spectra[key]:
                    spectrum_curves[_curve_pin(row)].append(row)
                for label, rows in sorted(spectrum_curves.items()):
                    rows = sorted(rows, key=lambda row: row['index'])
                    axes[0].plot([row['index'] for row in rows],
                                 [row['cumulative_energy'] for row in rows], label=label)
                axes[0].set(xlabel='Retained rank', ylabel='Cumulative squared singular-value energy', ylim=(0, 1.02))
                axes[0].axhline(.99, color='#777777', lw=.8, ls='--')
                error_curves = defaultdict(list)
                for row in errors[key]:
                    branch = row.get('branch', row.get('branch_index', 'group'))
                    for column, method, kind in (
                            ('shared_weight_error', 'shared', 'weight'),
                            ('independent_weight_error', 'independent', 'weight'),
                            ('shared_output_error', 'shared', 'activation'),
                            ('independent_output_error', 'independent', 'activation')):
                        if row.get(column) is not None:
                            error_curves[(method, str(branch), kind)].append((row['requested_rank'], row[column]))
                for (method, branch, kind), values in sorted(error_curves.items()):
                    values.sort()
                    axes[1].plot([p[0] for p in values], [p[1] for p in values],
                                 linestyle='--' if kind == 'activation' else '-',
                                 label=f'{method} / PE {branch} / {kind}')
                axes[1].set(xlabel='Requested rank', ylabel='Relative error (Frobenius norm)', ylim=(0, None))
                for ax, curves in zip(axes, (spectrum_curves, error_curves)):
                    ax.grid(alpha=.2)
                    if curves:
                        ax.legend(fontsize=7)
                    else:
                        ax.text(.5, .5, 'Not measured', ha='center', va='center', transform=ax.transAxes)
                circuit, model, seed, group = key
                fig.suptitle(f'{circuit} | {model} | seed {seed}\n{group}', fontsize=10)
                path = directory / f'shared_decoder_{index + 1:03d}_pin.png'
                fig.savefig(path, dpi=160)
            finally:
                plt.close(fig)
            manifest.append(dict(circuit=circuit, model=model, seed=seed, group=group,
                                 kind='spectrum_and_errors',
                                 path=str(path.relative_to(output))))
            manifest.extend(_similarity_plots_pin(plt, output, directory, index, key, tables))
    (directory / 'manifest_pin.json').write_text(_json_pin(manifest), encoding='utf-8')
    return manifest


def write_shared_decoder_report_pin(output, tables, record, max_plots=12):
    """Write complete independent records and a portable workbook into a new run.

    The caller reserves a separate output directory. Existing report artifacts
    are rejected, and all JSON values are validated before any report write.
    """
    if type(max_plots) is not int or max_plots < 0:
        raise ValueError('max_plots must be a nonnegative integer')
    if not isinstance(tables, dict) or not isinstance(record, dict):
        raise TypeError('Report tables and record must be objects')
    data = {name: tables.get(name, []) for name in TABLE_NAMES_PIN}
    if any(not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)
           for rows in data.values()):
        raise ValueError('Each report table must be a list of row objects')
    if not isinstance(record.get('checkpoints', []), list):
        raise ValueError('record.checkpoints must be a list')
    _json_pin(data)
    _json_pin(record)
    output = Path(output)
    names = [name + '_pin.csv' for name in TABLE_NAMES_PIN]
    names += ['shared_decoder_summary_pin.xlsx', 'report_pin.html', 'report_metadata_pin.json', 'plots_pin']
    if any((output / name).exists() for name in names):
        raise FileExistsError('Shared decoder report artifacts already exist; use a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    for name in TABLE_NAMES_PIN:
        rows = data[name]
        fields = list(dict.fromkeys(key for row in rows for key in row)) or ['status']
        with (output / (name + '_pin.csv')).open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: _csv_value_pin(value) for key, value in row.items()} for row in rows)
    readme = Sheet('ReadMe', [26, 130], freeze_rows=1, freeze_columns=1)
    readme.add([Cell('项目', 3), Cell('解释与边界', 3)])
    for key, value in NOTES_PIN:
        readme.add([key, value], height=58)
    sheets = [readme, sheet_pin('Checkpoints', record.get('checkpoints', []),
                               '每行是独立 checkpoint；来源路径、配置、哈希和所用 train/val 样本记录见原始元数据。')]
    for name in TABLE_NAMES_PIN:
        if name != 'Spectra':
            sheets.append(sheet_pin(name, data[name], TABLE_NOTES_PIN[name]))
    workbook_path = output / 'shared_decoder_summary_pin.xlsx'
    _write_workbook(workbook_path, sheets)
    images = plots_shared_decoder_pin(output, data, max_plots)
    metadata = dict(record=record, interpretation_notes=dict(NOTES_PIN), plots=images,
                    tables={name: len(rows) for name, rows in data.items()},
                    sharing_scope='corresponding_decoder_branches_within_one_checkpoint',
                    aggregation='concatenate_independent_records_no_cross_checkpoint_basis_fit')
    (output / 'report_metadata_pin.json').write_text(_json_pin(metadata), encoding='utf-8')
    content = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
        '<title>同模型多 decoder 共享下基分析</title>',
        '<style>body{font:16px/1.65 system-ui;max-width:1180px;margin:32px auto;padding:0 18px;color:#25354a}'
        'img{width:100%;border:1px solid #ddd}table{border-collapse:collapse;width:100%}'
        'th,td{padding:9px;border:1px solid #ddd;text-align:left}code{overflow-wrap:anywhere}</style>',
        '<h1>同模型多 decoder 共享下基分析</h1>',
        '<p>每个电路、模型与 seed 独立分析。多个电路在这里仅并列展示，不共同拟合基底。</p>',
        '<p><a href="shared_decoder_summary_pin.xlsx">Excel 汇总</a> · '
        '<a href="report_metadata_pin.json">配置、来源与解释</a></p>',
        '<h2>来源 checkpoint</h2><table><tr><th>电路</th><th>模型</th><th>Seed</th><th>来源</th></tr>']
    for checkpoint in record.get('checkpoints', []):
        content.append('<tr>' + ''.join('<td>' + html.escape(str(checkpoint.get(key, ''))) + '</td>'
                                      for key in ('circuit', 'model', 'seed', 'path')) + '</tr>')
    content.extend(['</table><h2>解释与边界</h2><table>'])
    content += [f'<tr><td>{html.escape(key)}</td><td>{html.escape(value)}</td></tr>'
                for key, value in NOTES_PIN]
    content.append('</table><h2>完整数据</h2><ul>')
    content += [f'<li><a href="{name}_pin.csv">{name}</a>：{len(data[name])} 条记录</li>'
                for name in TABLE_NAMES_PIN]
    content.append('</ul><h2>组内谱、近似误差与 decoder 相似度</h2>')
    if not images:
        content.append('<p>未生成图片；完整数值见 Excel / CSV。</p>')
    for image in images:
        caption = ' / '.join(str(image[key]) for key in ('circuit', 'model', 'seed', 'group'))
        caption += ' / ' + image.get('kind', 'spectrum_and_errors')
        content.append(f'<p><code>{html.escape(caption)}</code></p>'
                       f'<img src="{html.escape(image["path"], quote=True)}" alt="Within-checkpoint shared-down-basis analysis">')
    content.append('</html>')
    (output / 'report_pin.html').write_text('\n'.join(content), encoding='utf-8')
    return workbook_path
