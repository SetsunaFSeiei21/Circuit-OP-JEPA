"""Full-circuit PE attention archives and an offline canvas heatmap view.

Keep every valid key through softmax, including graph and external ports. The
viewer may hide auxiliary tokens or reorder physical pins, never renormalize.
Only float32 softmax matrices enter the HTML; all four scores stay in NPZ files.
"""
import base64
import json
from pathlib import Path

FIELDS = ('qk_scaled', 'additive_bias', 'attention_logit', 'attention_weight')


def full_circuit_records(batch, samples, captured, *, mode, seed, terminal_names):
    """Recover token identity from the observed batch, including non-MOS pins."""
    import numpy as np
    import torch
    layout = {key: batch[key].detach().cpu() for key in (
        'owner_index', 'pin_terminal_index', 'pin_net_index', 'device_mask')}
    if 'device_ids' not in batch:
        raise ValueError('Full-circuit inspection requires native token IDs from metadata=True')
    for b, sample in enumerate(samples):
        devices = sorted(sample['devices'], key=lambda row: row['device_id'])
        expected = {f'pin:{d["device_id"]}:{role}' for d in devices for role in d['terminals']}
        ports = {f'port:{name}': (name, net) for name, net in sample.get('ports', {}).items()}
        expected.update(ports)
        tokens = [dict(kind='graph', label='GRAPH', token_id='graph', sequence_index=0,
                       pin_index=None, owner_index=None, device_id=None,
                       device_type=None, terminal=None, net=None)]
        seen, net_ids, net_names = set(), {}, {}
        for pin in torch.where(layout['device_mask'][b])[0].tolist():
            token_id = batch['device_ids'][b][pin]
            owner = int(layout['owner_index'][b, pin])
            terminal = int(layout['pin_terminal_index'][b, pin])
            if 0 <= owner < len(devices):
                dev = devices[owner]
                if not 0 <= terminal < len(terminal_names):
                    raise ValueError('Owned full-circuit pin has no terminal identity')
                role = terminal_names[terminal]
                if token_id != f'pin:{dev["device_id"]}:{role}' or role not in dev['terminals']:
                    raise ValueError('Native token ID and physical device/terminal disagree')
                node = dict(kind='device', label=f'{dev["device_id"]}.{role.upper()}',
                            device_id=dev['device_id'], device_type=dev['device_type'],
                            terminal=role, net=dev['terminals'][role])
            elif owner == -1 and terminal == -1 and token_id in ports:
                port, net = ports[token_id]
                node = dict(kind='port', label=f'PORT:{port}', device_id=None,
                            device_type=None, terminal=port, net=net)
            else:
                raise ValueError('Valid full-circuit token has an invalid owner or external port')
            if token_id in seen:
                raise ValueError('Repeated physical pin/port token')
            seen.add(token_id)
            net_id = int(layout['pin_net_index'][b, pin])
            net = node['net']
            if net_id < 0 or net_ids.get(net, net_id) != net_id or net_names.get(net_id, net) != net:
                raise ValueError('Native tensor net identity differs from physical pin connectivity')
            net_ids[net], net_names[net_id] = net_id, net
            tokens.append(dict(node, token_id=token_id, sequence_index=pin + 1,
                               pin_index=pin, owner_index=owner))
        if seen != expected:
            raise ValueError('Full-circuit token mapping does not cover every device terminal and port')
        sequence = torch.tensor([node['sequence_index'] for node in tokens], dtype=torch.long)
        matrices = {name: [] for name in FIELDS}
        layers, heads = [], None
        for item in captured:
            layers.append({key: item[key] for key in ('pe_layer', 'encoder_layer', 'mask_scope')})
            count = item['attention_weight'].shape[1]
            if heads is not None and heads != count:
                raise ValueError('PE layers have inconsistent attention head counts')
            heads = count
            for name in FIELDS:
                # Captures are already on CPU. Append the arithmetic head mean
                # AFTER the native full-key softmax, before any display slicing.
                value = item[name][b].index_select(-2, sequence).index_select(-1, sequence)
                value = torch.cat((value, value.mean(0, keepdim=True)), dim=0).float().numpy()
                matrices[name].append(value)
        if not layers:
            raise ValueError('No PE self-attention layers available for a full-circuit heatmap')
        arrays = {name: np.stack(values) for name, values in matrices.items()}
        weights = arrays['attention_weight']
        if not np.isfinite(weights).all() or not np.allclose(weights.sum(-1), 1., atol=3e-5, rtol=3e-5):
            raise ValueError('Valid full-token attention rows must retain unit probability mass')
        meta = dict(model=mode, seed=seed, circuit=sample['circuit_id'], sample_id=sample['sample_id'],
                    tokens=tokens, layers=layers, heads=[*range(1, heads + 1), 'mean'],
                    shape=list(weights.shape), axes=['pe_layer', 'head', 'query', 'key'],
                    dtype='float32', padding_excluded=True, normalized_over='complete native key sequence',
                    graph_and_ports_retained=True, within_device_renormalization=False)
        yield meta, arrays


class FullCircuitReport:
    def __init__(self, output, viewer_ids):
        self.root = Path(output) / 'full_attention_pin'
        self.viewer_ids = set(viewer_ids)
        self.entries, self.viewer = [], []

    def add(self, records):
        import numpy as np
        for meta, arrays in records:
            if not self.entries:
                self.root.mkdir(exist_ok=False)
            filename = f'{len(self.entries) + 1:06d}_pin.npz'
            # String arrays only for metadata: loading never requires pickle.
            with (self.root / filename).open('xb') as stream:
                np.savez_compressed(stream, metadata_json=np.asarray(json.dumps(meta, ensure_ascii=False)), **arrays)
            self.entries.append(dict(model=meta['model'], sample_id=meta['sample_id'],
                                     file=filename, shape=meta['shape']))
            if meta['sample_id'] in self.viewer_ids:
                compact = np.asarray(arrays['attention_weight'], dtype='<f4', order='C')
                self.viewer.append(dict(meta, weights_f32le_b64=base64.b64encode(compact.tobytes()).decode('ascii')))

    def finish(self):
        if self.entries:
            with (self.root / 'manifest_pin.json').open('x', encoding='utf-8') as stream:
                json.dump(dict(format_version=1, fields=FIELDS, entries=self.entries,
                               note='Each NPZ retains graph + external ports + every device pin; no padding or row renormalization.'),
                          stream, ensure_ascii=False, indent=2, allow_nan=False)


FULL_CIRCUIT_SECTION = r'''
<section id="full-circuit-section"><h2>全电路 · 各 PE 的 Self-Attention 大热力图</h2>
<p class="note">每张图同时包含全部器件引脚及跨器件注意力，行是 Query、列是 Key。一层 PE 一张图；MOS 的 D/G/S/B 和电阻、电容等端口按器件分组。</p>
<div class="controls" id="full-controls"><label>矩阵范围<select id="full-scope"><option value="devices">全部器件引脚</option><option value="all">全部有效 token（含外部端口和 GRAPH）</option></select></label>
<label>各 PE 共同色标<select id="full-range"><option value="auto">0 至当前各层共同最大值</option><option value="fixed">固定 0–1</option></select></label></div>
<p class="note" id="full-meta"></p><p class="note" id="full-status" role="status"></p><div id="full-pe-grid"></div>
<p class="note">展示的是完整序列 softmax 后的权重，隐藏 GRAPH/外部端口只改变显示范围，不重新归一化。移动鼠标读取任意跨器件引脚对的得分；按钮可下载每层的大图。</p></section>
'''


FULL_CIRCUIT_JS = r'''
let fullCache=null,fullDrawKey=null;
const fullPalette=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
function fullColor(value,max){const t=Math.max(0,Math.min(1,value/max))*4,i=Math.min(3,Math.floor(t)),f=t-i;return 'rgb('+fullPalette[i].map((v,k)=>Math.round(v+(fullPalette[i+1][k]-v)*f)).join(',')+')';}
function fullDecode(record){if(fullCache?.record===record)return fullCache.values;const raw=atob(record.weights_f32le_b64),bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);const count=record.shape.reduce((a,b)=>a*b,1);if(bytes.length!==count*4)throw new Error('Full matrix binary size does not match metadata');const data=new DataView(bytes.buffer),values=new Float32Array(count);for(let i=0;i<count;i++)values[i]=data.getFloat32(i*4,true);fullCache={record,values};return values;}
function fullOrder(record,all){const priority={d:0,g:1,s:2,b:3};const indices=record.tokens.map((_,i)=>i).filter(i=>all||record.tokens[i].kind==='device');
return indices.sort((ia,ib)=>{const a=record.tokens[ia],b=record.tokens[ib],rank={graph:0,port:1,device:2};if(rank[a.kind]!==rank[b.kind])return rank[a.kind]-rank[b.kind];if(a.kind!=='device')return a.sequence_index-b.sequence_index;const compare=a.device_id<b.device_id?-1:a.device_id>b.device_id?1:0;return compare||(priority[a.terminal]??10)-(priority[b.terminal]??10)||a.sequence_index-b.sequence_index;});}
function fullGroup(token){return token.kind==='device'?'device:'+token.device_id:token.token_id;}
function fullHeatmap(canvas,record,values,layer,head,indices,max){const n=indices.length,N=record.tokens.length,H=record.heads.length,offset=(layer*H+head)*N*N;
const side=Math.max(480,Math.min(1400,n*10)),step=side/n,font=Math.max(6,Math.min(10,step*.85)),margin=190,top=80,bottom=250;
canvas.width=margin+side+38;canvas.height=top+side+bottom;canvas.style.width='100%';canvas.style.maxWidth=canvas.width+'px';canvas.style.height='auto';
const ctx=canvas.getContext('2d');if(!ctx)throw new Error('Browser canvas is unavailable');ctx.fillStyle='#fff';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.font='bold 17px sans-serif';ctx.fillStyle='#182b40';
ctx.fillText(`${record.model} | sample ${record.sample_id}`,margin,24);ctx.font='14px sans-serif';const pe=record.layers[layer];ctx.fillText(`PE ${pe.pe_layer} / encoder ${pe.encoder_layer} / ${pe.mask_scope} | head ${record.heads[head]} | ${n} x ${n}`,margin,49);
for(let q=0;q<n;q++)for(let k=0;k<n;k++){ctx.fillStyle=fullColor(values[offset+indices[q]*N+indices[k]],max);ctx.fillRect(margin+k*step,top+q*step,step+.2,step+.2);}
ctx.fillStyle='#182b40';ctx.font=font+'px sans-serif';ctx.textAlign='right';ctx.textBaseline='middle';
for(let i=0;i<n;i++){const label=record.tokens[indices[i]].label;ctx.fillText(label,margin-8,top+(i+.5)*step);ctx.save();ctx.translate(margin+(i+.5)*step,top+side+8);ctx.rotate(-Math.PI/2);ctx.fillText(label,0,0);ctx.restore();}
ctx.strokeStyle='rgba(255,255,255,.7)';ctx.lineWidth=1;for(let i=1;i<n;i++)if(fullGroup(record.tokens[indices[i]])!==fullGroup(record.tokens[indices[i-1]])){ctx.beginPath();ctx.moveTo(margin+i*step,top);ctx.lineTo(margin+i*step,top+side);ctx.moveTo(margin,top+i*step);ctx.lineTo(margin+side,top+i*step);ctx.stroke();}
ctx.strokeStyle='#182b40';ctx.strokeRect(margin,top,side,side);ctx.textAlign='left';ctx.textBaseline='alphabetic';ctx.font='12px sans-serif';const y=top+side+195,w=Math.min(360,side);for(let x=0;x<w;x++){ctx.fillStyle=fullColor(x/(w-1)*max,max);ctx.fillRect(margin+x,y,1,14);}ctx.fillStyle='#182b40';ctx.fillText('0',margin,y+31);ctx.textAlign='right';ctx.fillText(Number(max).toPrecision(6),margin+w,y+31);ctx.textAlign='left';ctx.fillText('Query = rows; Key = columns. Native softmax; no row renormalization.',margin,y+49);
return {left:margin,top,side,step,offset,N};}
function drawFullCircuit(){const status=byId('full-status'),grid=byId('full-pe-grid'),records=payload.full_circuit||[];
const record=records.find(r=>r.model===byId('model').value&&String(r.sample_id)===byId('sample').value);
if(!record){grid.replaceChildren();byId('full-controls').hidden=true;byId('full-meta').textContent='';status.textContent='这份旧报告仅保存 MOS 内部 4×4 子矩阵，没有跨器件数据。请更新后重新运行 inspect_pe_attention_pin.sh（只推理，不重训），即可生成全电路大图。';fullDrawKey=null;return;}
byId('full-controls').hidden=false;const head=record.heads.map(String).indexOf(byId('head').value);if(head<0){status.textContent='Saved full matrix has no matching head';return;}
const all=byId('full-scope').value==='all',fixed=byId('full-range').value==='fixed',key=JSON.stringify([record.model,record.sample_id,head,all,fixed]);if(key===fullDrawKey)return;
try{const values=fullDecode(record),indices=fullOrder(record,all),N=record.tokens.length,H=record.heads.length;let max=fixed?1:0;
if(!fixed)for(let layer=0;layer<record.layers.length;layer++){const offset=(layer*H+head)*N*N;for(const q of indices)for(const k of indices)max=Math.max(max,values[offset+q*N+k]);}max=max||1;
grid.replaceChildren();byId('full-meta').textContent=`${record.model} | 样本 ${record.sample_id} | head ${record.heads[head]} | ${indices.length} 个${all?'有效 token':'器件引脚'} | ${record.layers.length} 个 PE = ${record.layers.length} 张大图 | 共同色标 0–${Number(max).toPrecision(6)}`;
status.textContent='';for(let layer=0;layer<record.layers.length;layer++){const pe=record.layers[layer],card=document.createElement('article');card.className='full-pe-card';card.dataset.peLayer=String(pe.pe_layer);const title=document.createElement('h3');title.textContent=`PE ${pe.pe_layer} · 全电路 ${indices.length} × ${indices.length}`;card.appendChild(title);
const canvas=document.createElement('canvas');canvas.setAttribute('aria-label',`${record.model} PE ${pe.pe_layer} all circuit pins attention`);card.appendChild(canvas);const geometry=fullHeatmap(canvas,record,values,layer,head,indices,max);
const hover=document.createElement('p');hover.className='note';hover.textContent='鼠标移动到矩阵内，查看 Query / Key 引脚、网络和实际注意力权重。';card.appendChild(hover);
canvas.addEventListener('mousemove',event=>{const rect=canvas.getBoundingClientRect(),x=(event.clientX-rect.left)*canvas.width/rect.width,y=(event.clientY-rect.top)*canvas.height/rect.height,q=Math.floor((y-geometry.top)/geometry.step),k=Math.floor((x-geometry.left)/geometry.step);if(q<0||k<0||q>=indices.length||k>=indices.length)return;const qt=record.tokens[indices[q]],kt=record.tokens[indices[k]],v=values[geometry.offset+indices[q]*N+indices[k]];hover.textContent=`Query ${qt.label} [${qt.net??'graph'}] ← Key ${kt.label} [${kt.net??'graph'}] | attention = ${v.toPrecision(9)} | native indices ${qt.sequence_index}, ${kt.sequence_index}`;});
const button=document.createElement('button');button.type='button';button.textContent=`下载 PE ${pe.pe_layer} 大图 PNG`;button.addEventListener('click',()=>{canvas.toBlob(blob=>{if(!blob){status.textContent='PNG export failed';return;}const a=document.createElement('a'),url=URL.createObjectURL(blob);a.href=url;a.download=`${record.model}_${record.sample_id}_PE${pe.pe_layer}_head${record.heads[head]}_${all?'all_tokens':'device_pins'}.png`.replace(/[^A-Za-z0-9_.-]/g,'_');a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);},'image/png');});card.appendChild(button);grid.appendChild(card);}
fullDrawKey=key;
}catch(error){status.textContent='全电路热图显示失败：'+error.message;fullDrawKey=null;}}
byId('full-scope').addEventListener('change',drawFullCircuit);byId('full-range').addEventListener('change',drawFullCircuit);
'''
