"""Streaming PE-attention tables and exact MOS matrices for offline inspection."""
from collections import defaultdict
import csv
import json
from pathlib import Path

from device_jepa.excel import Cell, Sheet, _column, _write_workbook
from final_exam.gate_probe_report_pin import Moments

ROLES = ('d', 'g', 's', 'b')
BASE_FIELDS = ('model', 'seed', 'sample_id', 'circuit', 'pe_layer', 'encoder_layer',
               'mask_scope', 'device_id', 'device_type', 'head', 'query_role',
               'query_pin_index', 'query_sequence_index', 'query_net')
VALUES = ('qk_scaled', 'additive_bias', 'attention_logit', 'attention_weight')
SCORE_FIELDS = (*BASE_FIELDS, 'key_role', 'key_pin_index', 'key_sequence_index',
                'key_net', 'same_net', 'hard_masked', *VALUES)
MASS_VALUES = ('intra_device_mass', 'other_device_mass', 'external_port_mass',
               'graph_token_mass', 'padding_mass', 'total_mass')
MASS_FIELDS = (*BASE_FIELDS, *MASS_VALUES)
GROUP_FIELDS = ('model', 'pe_layer', 'encoder_layer', 'mask_scope', 'device_id',
                'device_type', 'head', 'query_role', 'key_role')
SUMMARY_FIELDS = (*GROUP_FIELDS, 'samples', 'unmasked_samples',
                  *(f'{name}_{stat}' for name in VALUES for stat in ('mean', 'std', 'min', 'max')))
MASS_GROUP_FIELDS = GROUP_FIELDS[:-1]
MASS_SUMMARY_FIELDS = (*MASS_GROUP_FIELDS, 'samples', *(name + '_mean' for name in MASS_VALUES))


def table_sheet(name, fields, rows, note):
    sheet = Sheet(name, [48 if key == 'model' else 26 for key in fields], freeze_rows=3, freeze_columns=1)
    sheet.note(name + ' — PE self-attention diagnostics', 1, 30)
    sheet.note(note, 2, 46)
    sheet.add([Cell(field, 3) for field in fields], 34)
    for row in rows:
        sheet.add([Cell(value, 11 if type(value) in (float, int) else 5) for value in row])
    sheet.filter_range = f'A3:{_column(len(fields))}{len(sheet.rows)}'
    return sheet


def new_matrix(row):
    return dict(model=row['model'], sample_id=row['sample_id'], pe_layer=row['pe_layer'],
        encoder_layer=row['encoder_layer'], mask_scope=row['mask_scope'], device_id=row['device_id'],
        device_type=row['device_type'], head=row['head'], nets=[''] * 4, masses=[None] * 4,
        **{name: [[None] * 4 for _ in ROLES] for name in VALUES})


def matrix_sheet(name, matrices, note):
    sheet = Sheet(name, [34, 18, 18, 18, 18, 5, 28, 18, 18, 18, 18], freeze_rows=3)
    sheet.note(name + ' — D / G / S / B', 1, 32)
    sheet.note(note, 2, 42)
    sheet.note('Rows = query, columns = key. Left: full-sequence softmax weight. Right: pre-softmax logit. Blank = hard masked.', height=42)
    for matrix in matrices:
        sheet.note(f'{matrix["model"]} | {matrix["sample_id"]} | PE {matrix["pe_layer"]} '
                   f'(encoder {matrix["encoder_layer"]}, {matrix["mask_scope"]}) | '
                   f'{matrix["device_id"]} ({matrix["device_type"]}) | head {matrix["head"]}', 3, 34)
        sheet.add(['Attention weight', *(role.upper() for role in ROLES), '',
                   'Attention logit', *(role.upper() for role in ROLES)])
        for index, role in enumerate(ROLES):
            sheet.add([role.upper(), *(Cell(v, 11) for v in matrix['attention_weight'][index]), '',
                       role.upper(), *(Cell(v, 11) for v in matrix['attention_logit'][index])])
        sheet.add([])
    return sheet


class AttentionReport:
    preview_limit = 4000

    def __init__(self, output, viewer_ids):
        self.output = Path(output)
        self.viewer_ids = set(viewer_ids)
        self.models, self.viewer = {}, {}
        self.groups = defaultdict(lambda: {name: Moments() for name in VALUES})
        self.mass_groups = defaultdict(lambda: {name: Moments() for name in MASS_VALUES})
        self.preview, self.handles = [], []
        self.score_count = self.mass_count = 0

    def _writer(self, name, fields):
        stream = (self.output / name).open('x', newline='', encoding='utf-8-sig')
        self.handles.append(stream)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        return writer

    def __enter__(self):
        try:
            self.scores = self._writer('attention_scores_pin.csv', SCORE_FIELDS)
            self.masses = self._writer('attention_mass_pin.csv', MASS_FIELDS)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        for stream in self.handles:
            stream.close()

    def add_model(self, mode, info):
        self.models[mode] = info

    def add_rows(self, rows):
        for kind, row in rows:
            viewer = None
            if row['sample_id'] in self.viewer_ids:
                identity = tuple(row[key] for key in ('model', 'sample_id', 'pe_layer', 'device_id', 'head'))
                if identity not in self.viewer:
                    self.viewer[identity] = new_matrix(row)
                viewer = self.viewer[identity]
            query = ROLES.index(row['query_role'])
            if kind == 'score':
                self.scores.writerow(row)
                self.score_count += 1
                if len(self.preview) < self.preview_limit:
                    self.preview.append([row[field] for field in SCORE_FIELDS])
                stats = self.groups[tuple(row[field] for field in GROUP_FIELDS)]
                for name in VALUES:
                    if row[name] is not None:
                        stats[name].add(row[name])
                if viewer is not None:
                    key = ROLES.index(row['key_role'])
                    for name in VALUES:
                        viewer[name][query][key] = row[name]
                    viewer['nets'][query] = row['query_net']
            elif kind == 'mass':
                self.masses.writerow(row)
                self.mass_count += 1
                stats = self.mass_groups[tuple(row[field] for field in MASS_GROUP_FIELDS)]
                for name in MASS_VALUES:
                    stats[name].add(row[name])
                if viewer is not None:
                    viewer['masses'][query] = {name: row[name] for name in MASS_VALUES}
            else:
                raise ValueError(f'Unexpected attention row: {kind}')

    def finish(self, record):
        if not self.score_count:
            raise ValueError('No MOS attention rows were observed')
        summary = []
        means = {}
        for group, stats in self.groups.items():
            row = dict(zip(GROUP_FIELDS, group))
            row.update(samples=stats['attention_weight'].n, unmasked_samples=stats['attention_logit'].n)
            for name, values in stats.items():
                for suffix, value in (('mean', values.mean), ('std', values.std),
                                      ('min', values.minimum), ('max', values.maximum)):
                    row[name + '_' + suffix] = value if values.n else None
            summary.append(row)
            if row['head'] == 'mean':
                identity = tuple(row[key] for key in ('model', 'pe_layer', 'device_id'))
                if identity not in means:
                    means[identity] = new_matrix(dict(row, sample_id=f'mean of {row["samples"]} samples'))
                query, key = ROLES.index(row['query_role']), ROLES.index(row['key_role'])
                for name in VALUES:
                    means[identity][name][query][key] = row[name + '_mean']
        writer = self._writer('attention_summary_pin.csv', SUMMARY_FIELDS)
        writer.writerows(summary)
        mass_summary = []
        for group, stats in self.mass_groups.items():
            mass_summary.append(dict(zip(MASS_GROUP_FIELDS, group),
                samples=stats['total_mass'].n, **{name + '_mean': value.mean for name, value in stats.items()}))
        writer = self._writer('attention_mass_summary_pin.csv', MASS_SUMMARY_FIELDS)
        writer.writerows(mass_summary)
        record.update(score_rows=self.score_count, mass_rows=self.mass_count,
                      viewer_matrices=len(self.viewer), excel_detail_preview_rows=len(self.preview))
        overview = [
            ['Scope', 'Downstream PE Self-Attention only; natively ungated checkpoints; no training'],
            ['Layer count', 'PE = cross_layers; SE = encoder_layers - cross_layers; read from saved checkpoint'],
            ['PE block order', 'Self-Attention precedes Cross-Attention. For a fixed ungated topology, PE1 sees pure structure and can be identical across designs.'],
            ['Matrix order', 'D/G/S/B; row = query, column = key; directional entries need not match'],
            ['Normalization', 'Softmax over ALL model keys before slicing the MOS matrix. No within-MOS renormalization.'],
            ['Head mean', 'Mean of head probabilities, not softmax of averaged logits'],
            ['Masked entries', 'Blank logits/bias mean hard -inf; corresponding attention weight is zero'],
            ['Native outputs', 'Normal forward is preserved; replay of each MHA output is checked numerically'],
            ['SampleMatrices', 'Head means for the bounded HTML sample selection'],
            ['MeanMatrices', 'Head means averaged over ALL selected samples; no row renormalization'],
            ['All heads / all samples', 'attention_scores_pin.csv and attention_mass_pin.csv'],
            ['ScoreSummary / std', 'Per-device/layer/head/role statistics across selected samples; population std (ddof=0)'],
            ['Preview', f'First {len(self.preview)} of {self.score_count} raw score rows; CSV contains all'],
            ['Full probe metadata', 'probe_pin.json contains source fingerprint, checkpoint SHA-256 and exact config'],
        ]
        sheets = [table_sheet('ReadMe', ('Item', 'Meaning'), overview, 'Inspect attention redistribution; scores are not physical conduction probabilities.'),
                  matrix_sheet('SampleMatrices', [v for v in self.viewer.values() if v['head'] == 'mean'],
                               'Sample-level head means for viewer_samples; use the offline HTML to inspect individual heads.'),
                  matrix_sheet('MeanMatrices', means.values(), 'All selected samples averaged; head mean; no row renormalization.'),
                  table_sheet('ScoreSummary', SUMMARY_FIELDS, ([r[k] for k in SUMMARY_FIELDS] for r in summary),
                              'Statistics across selected samples, preserving layer/device/head/terminal identities.'),
                  table_sheet('MassSummary', MASS_SUMMARY_FIELDS, ([r[k] for k in MASS_SUMMARY_FIELDS] for r in mass_summary),
                              'Each query partitions its probability mass into same device / other devices / external ports / graph / padding.'),
                  table_sheet('DetailPreview', SCORE_FIELDS, self.preview, 'Bounded preview; full details are in attention_scores_pin.csv.'),
                  table_sheet('Checkpoints', ('model', 'checkpoint', 'SHA256', 'PE layers', 'heads', 'Requires measured OP', 'Max replay output error'),
                              ([mode, value['checkpoint'], value['sha256'], value['pe_layers'], value['heads'],
                                value['requires_measured_op'], value['native_output_max_abs_error']] for mode, value in self.models.items()),
                              'Original checkpoint weights and fitted preprocessing are reused without modification.')]
        _write_workbook(self.output / 'attention_scores_pin.xlsx', sheets)
        payload = dict(records=list(self.viewer.values()), models=self.models,
                       selected_sample_count=len(record['selected_sample_ids']), viewer_samples=record['viewer_samples'])
        write_viewer(self.output / 'attention_viewer_pin.html', payload)
        for stream in self.handles:
            stream.flush()


def write_viewer(path, payload):
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    # Both fresh probes and report-only refreshes keep existing artifacts intact.
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(HTML.replace('__PAYLOAD__', encoded))


def refresh_viewer(source, output=None):
    """Re-render a saved probe's bounded viewer data without loading any model."""
    source = Path(source).expanduser()
    if source.is_dir():
        source = source / 'attention_viewer_pin.html'
    text = source.read_text(encoding='utf-8')
    marker = 'const payload='
    if marker not in text:
        raise ValueError('Source must be an attention probe HTML containing saved viewer data')
    payload, _ = json.JSONDecoder().raw_decode(text.split(marker, 1)[1].lstrip())
    if not isinstance(payload, dict) or not payload.get('records'):
        raise ValueError('Source contains no saved attention matrices')
    output = Path(output).expanduser() if output else source.with_name('attention_pe_overview_pin.html')
    write_viewer(output, payload)
    return output


HTML = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PE Self-Attention · MOS 引脚得分</title>
<style>
body{font:15px system-ui,sans-serif;background:#f3f6fb;color:#182b40;margin:0}main{max-width:1320px;margin:auto;padding:30px}
h1{font-size:26px;margin-bottom:8px}p{line-height:1.6;color:#4a5e74}.controls{display:flex;flex-wrap:wrap;gap:14px;background:white;padding:20px;border-radius:12px}
label{display:flex;flex-direction:column;gap:6px;font-size:12px;color:#41576f}select{max-width:500px;padding:8px;border:1px solid #c2ccda;border-radius:5px;color:#182b40}
section{background:white;border-radius:12px;padding:22px;margin-top:18px;overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:14px;text-align:center;border:1px solid #dae2ed}th{background:#edf2f8}td.same{outline:2px dashed #8599b2;outline-offset:-4px}
.sub{font-size:12px;color:#6a7d92;display:block;margin-top:4px}.note{font-size:13px}.value{font-weight:650}.mass th,.mass td{padding:9px;font-size:13px}
.pe-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px;margin-top:18px}
.pe-card{border:1px solid #dae2ed;border-radius:10px;padding:16px;min-width:0}.pe-card h3{margin:0 0 8px;font-size:18px}
.pe-card th,.pe-card td{padding:10px 5px;font-size:12px}.pe-card table{table-layout:fixed}.pe-card .sub{overflow-wrap:anywhere;font-size:10px}
.pe-card button{margin-top:12px;border:1px solid #c2ccda;border-radius:5px;background:#f3f6fb;color:#244b72;padding:7px 12px;cursor:pointer}
.scale{display:inline-block;width:150px;height:12px;border-radius:3px;vertical-align:middle;margin:0 8px}
@media(max-width:600px){main{padding:14px}section{padding:14px}.pe-grid{grid-template-columns:1fr}select{max-width:100%}.controls{padding:14px}}
</style><main><h1>PE Self-Attention · MOS 引脚得分</h1>
<p>无门控 checkpoint 的下游 PE 自注意力。行是 Query，列是 Key。Softmax 在完整序列上计算，再截取器件内子矩阵。</p>
<p class="note">每个 PE 块先 Self-Attention，后 Cross-Attention。固定拓扑下第一个 PE 的自注意力可能跨样本完全相同；后续层才接收到此前融合的信息。</p>
<div class="controls"><label>模型<select id="model"></select></label><label>样本<select id="sample"></select></label>
<label>MOS<select id="device"></select></label><label>Head<select id="head"></select></label>
<label>得分<select id="value"><option value="attention_weight">Softmax 注意力权重</option><option value="attention_logit">Softmax 前 logit</option>
<option value="qk_scaled">QK / √head_dim</option><option value="additive_bias">原始偏置 / 掩码</option></select></label></div>
<section><h2>所有 PE 层 · Self-Attention 热图</h2><p class="note" id="overview-meta"></p>
<p class="note" id="overview-legend"></p><div id="pe-overview" class="pe-grid"></div>
<p class="note">所有图对应同一个模型、样本、MOS 和 head，每个 PE 独立显示一张热图。各层共用色标；数值保持原始得分，不做器件内重新归一化。</p></section>
<section><label>PE 层（详细查看）<select id="layer"></select></label><h2 id="title"></h2><p class="note" id="meta"></p><table id="matrix"></table>
<p class="note" id="legend"></p></section><section><h2>每个 Query 的权重去向</h2><table id="mass" class="mass"></table>
<p class="note">器件内权重＋其他器件＋外部端口＋图 token＋padding ≈ 1。器件内 4×4 子矩阵的行和通常小于 1。Head mean 是各 head 权重的算术平均。</p></section>
<p class="note" id="coverage"></p></main><script>
const payload=__PAYLOAD__;
const byId=id=>document.getElementById(id), roles=['D','G','S','B'];
const fmt=x=>x===null?'−∞':Number(x).toPrecision(6);
function options(id, values, preferred){const el=byId(id), old=preferred??el.value;el.replaceChildren();for(const v of values){const op=document.createElement('option');op.value=String(v);op.textContent=String(v);el.appendChild(op)}if(values.map(String).includes(old))el.value=old;}
function cell(row,text,tag='td'){const node=document.createElement(tag);node.textContent=text;row.appendChild(node);return node;}
function selectedRows(){return payload.records.filter(r=>r.model===byId('model').value&&String(r.sample_id)===byId('sample').value&&r.device_id===byId('device').value&&String(r.head)===byId('head').value).sort((a,b)=>Number(a.pe_layer)-Number(b.pe_layer));}
function sharedScale(rows,field){let scale=0;for(const r of rows)for(const line of r[field])for(const v of line)if(v!==null&&Number.isFinite(v))scale=Math.max(scale,Math.abs(v));return scale||1;}
function color(v,field,scale){if(v===null)return '#e4e8ee';if(field==='attention_weight')return `rgba(30,116,194,${.05+.7*Math.max(0,Math.min(1,v))})`;return v>=0?`rgba(25,123,194,${.05+.5*Math.abs(v)/scale})`:`rgba(220,115,57,${.05+.5*Math.abs(v)/scale})`;}
function drawMatrix(table,r,field,scale){const values=r[field];table.replaceChildren();let tr=table.insertRow();cell(tr,'Q ↓ / K →','th');
for(let k=0;k<4;k++){const c=cell(tr,roles[k],'th');const sub=document.createElement('span');sub.className='sub';sub.textContent=r.nets[k];c.appendChild(sub)}
for(let q=0;q<4;q++){tr=table.insertRow();cell(tr,roles[q],'th');for(let k=0;k<4;k++){const v=values[q][k],c=cell(tr,fmt(v));c.className='value';if(r.nets[q]===r.nets[k])c.classList.add('same');c.style.background=color(v,field,scale);}}}
function drawDetail(){const rows=selectedRows(),r=rows.find(r=>String(r.pe_layer)===byId('layer').value);if(!r)return;const field=byId('value').value,scale=sharedScale(rows,field);
byId('title').textContent=`${r.device_id} · PE ${r.pe_layer} · head ${r.head}`;
byId('meta').textContent=`${r.device_type} | encoder 第 ${r.encoder_layer} 层 | ${r.mask_scope} 掩码 | 样本 ${r.sample_id}`;
drawMatrix(byId('matrix'),r,field,scale);
byId('legend').textContent='与上方所有 PE 热图使用相同色标。虚线边框表示同网引脚，仍保留独立 token；−∞ 表示硬掩码。';
const mass=byId('mass');mass.replaceChildren();let tr=mass.insertRow();for(const label of ['Query','本器件','其他器件','外部端口','图 token','Padding','总和'])cell(tr,label,'th');
for(let q=0;q<4;q++){tr=mass.insertRow();cell(tr,roles[q],'th');for(const f of ['intra_device_mass','other_device_mass','external_port_mass','graph_token_mass','padding_mass','total_mass'])cell(tr,fmt(r.masses[q]?.[f]??null));}}
function draw(){const rows=selectedRows(),field=byId('value').value,scale=sharedScale(rows,field),grid=byId('pe-overview');grid.replaceChildren();
byId('overview-meta').textContent=`${byId('model').value} | 样本 ${byId('sample').value} | ${byId('device').value} | head ${byId('head').value} | 共 ${rows.length} 个 PE 层`;
const legend=byId('overview-legend');legend.replaceChildren();const ramp=document.createElement('span');ramp.className='scale';
if(field==='attention_weight'){legend.append('共同色标：0');ramp.style.background='linear-gradient(to right,rgba(30,116,194,.05),rgba(30,116,194,.75))';legend.append(ramp,'1（完整序列 softmax 权重）');}
else{legend.append(`共同色标：${fmt(-scale)}`);ramp.style.background='linear-gradient(to right,rgba(220,115,57,.55),#f8fafc,rgba(25,123,194,.55))';legend.append(ramp,`${fmt(scale)}；蓝色非负、橙色负、灰色硬掩码。`);}
for(const r of rows){const card=document.createElement('article');card.className='pe-card';card.dataset.peLayer=String(r.pe_layer);const title=document.createElement('h3');title.textContent=`PE ${r.pe_layer} · Self-Attention`;card.appendChild(title);
const meta=document.createElement('p');meta.className='note';meta.textContent=`encoder 第 ${r.encoder_layer} 层 · ${r.mask_scope} · head ${r.head}`;card.appendChild(meta);
const table=document.createElement('table');table.setAttribute('aria-label',`${r.device_id} PE ${r.pe_layer} head ${r.head} ${field}`);drawMatrix(table,r,field,scale);card.appendChild(table);
const button=document.createElement('button');button.type='button';button.textContent='查看该层详情';button.addEventListener('click',()=>{byId('layer').value=String(r.pe_layer);drawDetail();byId('title').scrollIntoView?.({behavior:'smooth',block:'start'});});card.appendChild(button);grid.appendChild(card);}
drawDetail();}
function refresh(level){let rows=payload.records;const controls=[['model','model'],['sample','sample_id'],['device','device_id'],['head','head'],['layer','pe_layer']];for(let i=0;i<controls.length;i++){const [id,key]=controls[i];if(i>=level){let values=[...new Set(rows.map(r=>r[key]))];if(id==='layer')values.sort((a,b)=>Number(a)-Number(b));options(id,values,id==='head'&&level===0?'mean':undefined);}rows=rows.filter(r=>String(r[key])===byId(id).value)}draw();}
['model','sample','device','head'].forEach((id,i)=>byId(id).addEventListener('change',()=>refresh(i+1)));byId('layer').addEventListener('change',drawDetail);byId('value').addEventListener('change',draw);
byId('coverage').textContent=`HTML 展示 ${payload.viewer_samples}/${payload.selected_sample_count} 个抽取样本，所有 PE 层和 heads；完整采样明细见 attention_scores_pin.csv。`;refresh(0);
</script></html>'''


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Add the all-PE heatmap overview to an existing probe HTML; no inference.')
    parser.add_argument('--source', required=True, help='Saved probe directory or attention_viewer_pin.html')
    parser.add_argument('--output', help='New HTML path; defaults to attention_pe_overview_pin.html beside the source')
    args = parser.parse_args()
    try:
        print(refresh_viewer(args.source, args.output))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
